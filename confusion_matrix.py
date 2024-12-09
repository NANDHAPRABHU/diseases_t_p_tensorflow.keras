import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
model = load_model('best_tomato_weights.h5')  # Replace 'model.h5' with your actual model file path

# Path to validation folder
validation_dir = r"C:\Users\GOWTHAM.S\Desktop\kaggle\tomato\train"  # Update with the path to your validation dataset

# Load validation data
datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(256, 256),  # Adjust to your model's input size
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important to maintain order for confusion matrix
)

# Predict on validation data
predictions = model.predict(validation_generator)

# Convert probabilities to class indices
predicted_classes = np.argmax(predictions, axis=1)

# True class labels
true_classes = validation_generator.classes
print(true_classes)
# Generate confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(),
            yticklabels=validation_generator.class_indices.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Optional: Print classification report
print(classification_report(true_classes, predicted_classes, target_names=validation_generator.class_indices.keys()))
