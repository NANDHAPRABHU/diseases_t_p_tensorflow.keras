# import numpy as np
# import tensorflow as tf
# from keras_preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
#
# # Load the pre-trained model
# model = load_model('cow_livestock_weights.h5')  # Replace with your model file path
#
# # Path to the single image
# image_path = r"C:\Users\GOWTHAM.S\Desktop\IMG-20241203-WA0006.jpg"
# # Load and preprocess the image
# img = load_img(image_path, target_size=(256, 256))  # Ensure the target size matches your model's input size
# img_array = img_to_array(img)  # Convert to a NumPy array
# img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
# img_array = img_array / 255.0  # Normalize pixel values (same as training)
#
# # Predict the class probabilities
# predictions = model.predict(img_array)
#
# # Convert probabilities to class indices
# predicted_class_index = np.argmax(predictions, axis=1)[0]
#
# validation_dir = r"E:\cow_diseases_dataset" # Update with the path to your validation dataset
#
# # Load validation data
# datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values
# validation_generator = datagen.flow_from_directory(
#     validation_dir,
#     target_size=(256, 256),  # Adjust to your model's input size
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=False  # Important to maintain order for confusion matrix
# )
#
# # Get class labels (mapping indices to class names)
# class_indices = {
#     v: k for k, v in validation_generator.class_indices.items()
# }  # Reverse the mapping
# print(class_indices)
# predicted_class_label = class_indices[predicted_class_index]
#
# #  Print the prediction
# print(f"Predicted Class Index: {predicted_class_index}")
# print(f"Predicted Class Label: {predicted_class_label}")
# # import tensorflow as tf
# # from tensorflow.keras.preprocessing import image
# # from tensorflow.keras.applications import InceptionV3
# # from tensorflow.keras.applications.inception_v3 import preprocess_input
# # import numpy as np
# # from PIL import Image
#
#
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Load the trained Inception V3 model
model = tf.keras.models.load_model("cow_model_new.keras")  # Replace with your model path

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((256, 256))  # Resize to 299x299 for InceptionV3
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image for InceptionV3
    return img_array


def predict_disease(image_path, threshold=0.8, top_n_classes=3):
    # Use the dynamic class indices from the validation set
    validation_dir = r"E:\cow_diseases_dataset"  # Update with the path to your validation dataset

    # Load validation data
    datagen = ImageDataGenerator(rescale=1. / 255)  # Normalize pixel values
    validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(256, 256),  # Adjust to your model's input size
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # Important to maintain order for confusion matrix
    )

    # Get class labels (mapping indices to class names)
    class_indices = validation_generator.class_indices
    print("Class Labels:")
    for index, label in class_indices.items():
        print(f"{index}: {label}")

    # Preprocess the input image
    input_image = preprocess_image(image_path)

    # Get predictions from the model
    predictions = model.predict(input_image)

    # Calculate softmax probabilities
    probabilities = tf.nn.softmax(predictions[0])  # Apply softmax to get probabilities
    max_confidence = np.max(probabilities)  # Get the max confidence
    predicted_class = np.argmax(probabilities)  # Get the predicted class index

    # Get the top N classes with their probabilities
    top_n_classes_indices = tf.argsort(probabilities, direction='DESCENDING')[:top_n_classes]
    top_n_probs = tf.gather(probabilities, top_n_classes_indices)

    # Check for OOD (confidence below the threshold)
    if any(prob > threshold for prob in top_n_probs.numpy()):
        # Directly fetch the predicted class label using the class index
        predicted_class_label = list(class_indices.keys())[list(class_indices.values()).index(predicted_class)]

        # Print the prediction
        print(f"Predicted Class Index: {predicted_class}")
        # print(f"Confidence: {max_confidence:.2f}")

        # Return the predicted class label and confidence
        return f"Predicted Class Label: {predicted_class_label}"
    else:
        return "Irrelevant Image"
# Example of using the function
image_path =r"E:\cow_diseases_dataset\Foot and Mouth disease\Foot and Mouth disease\mouthfibrinWM.jpg" # Replace with the new image path
result = predict_disease(image_path, threshold=0.3)
print(result)



#.....................................
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras.applications.inception_v3 import preprocess_input
# import numpy as np
# from PIL import Image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # Load the trained Inception V3 model
# model = tf.keras.models.load_model("cow_model_new.keras")  # Replace with your model path
#
# # Function to preprocess the image
# def preprocess_image(image_path):
#     img = Image.open(image_path).resize((256, 256))  # Ensure consistent resizing
#     img_array = image.img_to_array(img)  # Convert image to array
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = preprocess_input(img_array)  # Preprocess for InceptionV3
#     return img_array
#
# # Function to predict disease
# def predict_disease(image_path, threshold=0.3, top_n_classes=3):
#     # Path to the validation dataset
#     validation_dir = r"E:\cow_diseases_dataset"# Update with the path to your validation dataset
#
#     # Load validation data to get class labels
#     datagen = ImageDataGenerator(rescale=1./255)
#     validation_generator = datagen.flow_from_directory(
#         validation_dir,
#         target_size=(256, 256),  # Model's input size
#         batch_size=32,
#         class_mode='categorical',
#         shuffle=False
#     )
#
#     # Map indices to class labels
#     class_indices = validation_generator.class_indices
#     index_to_label = {v: k for k, v in class_indices.items()}
#
#     # Preprocess the input image
#     input_image = preprocess_image(image_path)
#
#     # Get predictions
#     predictions = model.predict(input_image)
#
#     # Apply softmax to get probabilities
#     probabilities = tf.nn.softmax(predictions[0]).numpy()
#
#     # Get the top N predicted classes
#     top_n_indices = np.argsort(probabilities)[::-1][:top_n_classes]
#     top_n_probs = probabilities[top_n_indices]
#
#     # Print class labels and probabilities
#     for i, class_index in enumerate(top_n_indices):
#         print(f"Class: {index_to_label[class_index]}, Probability: {top_n_probs[i]:.2f}")
#
#     # Check if the highest probability exceeds the threshold
#     if top_n_probs[0] >= threshold:
#         predicted_label = index_to_label[top_n_indices[0]]
#         confidence = top_n_probs[0]
#         return f"Predicted: {predicted_label} with confidence {confidence:.2f}"
#     else:
#         return "Irrelevant Image"
#
# # Example usage
# image_path = r"f.jpg"  # Replace with the new image path
# result = predict_disease(image_path, threshold=0.3)
# print(result)