import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score
# Load the model
model = load_model('transfer_learning_trained_face.h5')


def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Resize the image to 224x224
    img = cv2.resize(img, (224, 224))

    # Convert to float32 and scale pixel values
    img = img.astype('float32') / 255.0

    # Expand dimensions to match the model's input shape (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)

    return img


# Prepare the input image
image_path = r"C:\Users\GOWTHAM.S\Desktop\mmmm.jpg"
input_image = preprocess_image(image_path)

# Make predictions
predictions = model.predict(input_image)
print(predictions)
# Get the predicted class index
predicted_class_index = np.argmax(predictions, axis=1)[0]

# Output the predicted class index
print(f'Predicted class index: {predicted_class_index}')
print(f'Predicted class accuracy: {predictions[0][predicted_class_index]}')
class_names = ['24MCR067', '24MCR072', '24MCR090', '24MCR096', '24MCR115']

# Output the predicted class name
predicted_class_name = class_names[predicted_class_index]
print(f'Predicted class: {predicted_class_name}')

