import segmentation_models as sm
import tensorflow as tf
model = sm.Unet(input_shape=(256,256,3), classes=2)
print(model.summary())
# model.save("temp", save_format='h5')
# model = tf.saved_model.load("model.savedmodel")
# tf.saved_model.save(model, "/home/tejas/Compgeom/triton/scripts/")
# tf.keras.models.save_model(model, "/home/tejas/Compgeom/triton/scripts/")