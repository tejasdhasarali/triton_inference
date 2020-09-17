import tritongrpcclient
import tritongrpcclient.model_config_pb2 as mc
import tritonhttpclient
from tritonclientutils import triton_to_np_dtype
from tritonclientutils import InferenceServerException
import sys
import os
from PIL import Image
import numpy as np
np.set_printoptions(suppress=True)
import queue

# Tensorflow
# file_path = "/workspace/images"
# model_name = "unet_savedmodel"
# input_name = "input_1"
# output_name = "softmax"
# dtype = "FP32"
# classes = 2
# model_version = "1"
# format = "NHWC"

# PyTorch
file_path = "/workspace/images"
model_name = "production_model"
input_name = "INPUT__0"
output_name = "OUTPUT__0"
dtype = "FP32"
classes = 2
model_version = "1"
format = "NCHW"

class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()

# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))

def preprocess(img):
    print(img.size)
    img.load()
    # sample_img = img.convert('RGB')
    print(img.size)
    resized_img = img.resize((512, 512), Image.LANCZOS)
    print(resized_img.size)
    npdtype = triton_to_np_dtype(dtype)
    resized = np.asarray(resized_img, dtype = npdtype)
    print("Numpy shape",resized.shape)
    if format == "NCHW":
        resized = np.rollaxis(resized, 2, 0)
    print(resized.shape)
    resized = resized[np.newaxis, :,:,:]
    print(resized.shape)
    np.save("image_array", resized)
    return resized

def postprocess(results, output_name):
    # print(results)
    output_array = results.as_numpy(output_name)
    print(output_array.shape)
    for results in output_array:
        print(results.shape)
        if format == "NCHW":
            results = np.rollaxis(results, 0, 3)
        print(results.shape)        
        # results = np.reshape(results,results.shape[:-1])
        results = np.amax(results,axis=-1).astype(np.uint8)
        print(results.shape)
        # print(results)
        img = Image.fromarray(results, 'P')
        img.save(str(output_name)+'.png')
        img.show()
            

if __name__=="__main__":

    # try:    
    #     model_metadata = triton_client.get_model_metadata(
    #         model_name="localhost:8000", model_version="")
    # except InferenceServerException as e:
    #     print("failed to retrieve the metadata: " + str(e))
    #     sys.exit(1)
    
    # try:
    #     model_config = triton_client.get_model_config(
    #         model_name="localhost:8000", model_version="")
    # except InferenceServerException as e:
    #     print("failed to retrieve the config: " + str(e))
    #     sys.exit(1)

    try:
        # Specify large enough concurrency to handle the
        # the number of requests.
        triton_client = tritonhttpclient.InferenceServerClient(
            url="localhost:8000")
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    
    filenames = []
    if os.path.isdir(file_path):
        filenames = [
            os.path.join(file_path, f)
            for f in os.listdir(file_path)
            if os.path.isfile(os.path.join(file_path, f))
        ]
    else:
        filenames = [
           file_path,
        ]

    filenames.sort()

    # Preprocess the images into input data according to model
    # requirements
    image_data = []
    for filename in filenames:
        img = Image.open(filename)
        image_data.append(
            preprocess(img))
    
    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    responses = []
    result_filenames = []
    request_ids = []
    image_idx = 0
    last_request = False
    user_data = UserData()

    # Holds the handles to the ongoing HTTP async requests.
    async_requests = []

    sent_count = 0
    try:
        for image in image_data:
            sent_count += 1
            inputs = [tritonhttpclient.InferInput(input_name, image.shape, dtype)]
            outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

            inputs[0].set_data_from_numpy(image, binary_data=True)
            responses.append(
                triton_client.infer(model_name,
                                    inputs,
                                    request_id=str(sent_count),
                                    model_version=model_version,
                                    outputs=outputs))
    except InferenceServerException as e:
            print("inference failed: " + str(e))
            sys.exit(1)
    
    for response in responses:
        this_id = response.get_response()["id"]
        print("Request ",this_id)
        postprocess(response, output_name)
    print("PASS")

