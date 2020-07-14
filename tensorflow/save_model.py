import segmentation_models as sm
import tensorflow as tf
import os
if __name__ == "__main__":
    model = sm.Unet(input_shape=(256,256,3), classes=2, activation='softmax')
    print(model.summary())
    save_path = os.getcwd()+"/unet_savedmodel/1/model.savedmodel"
    tf.keras.models.save_model(model, save_path)
    # model.save("temp", save_format='h5')
    # tf.saved_model.save(model, save_path)
    print("Model saved in", save_path)