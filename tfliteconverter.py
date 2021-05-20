import tensorflow as tf
from keras.models import load_model


# Convert the model

classifier = load_model('./model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(classifier) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)