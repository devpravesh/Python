import tensorflow as tf

# Load the model
model_path = "/Users/justmac/Downloads/mobilenet-v2-tensorflow2-tf2-preview-classification-v4"
model = tf.saved_model.load(model_path)

# Define a function to wrap the model and specify the input signature
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)])
def serving_fn(input_tensor):
    return model(input_tensor)

# Specify a valid, writable directory for saving the model
new_model_path = "/Users/justmac/Documents/Python/saved_model_with_signature"  # Use your own path here

# Save the model with the signature
tf.saved_model.save(model, new_model_path, signatures={'serving_default': serving_fn})

# Now try converting the model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(new_model_path)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model saved and converted successfully!")
