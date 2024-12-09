import tensorflow as tf
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Configuration
IMG_SIZE = (256, 256)  # InceptionV3 input size
BATCH_SIZE = 32
EPOCHS = 10  # Adjust epochs for incremental training
LEARNING_RATE = 0.0001
TRAIN_DIR = r"E:\cucumber_dataset\Cucumber Disease Recognition Dataset\train"  # Replace with the path to new training data
VAL_DIR = r"E:\cucumber_dataset\Cucumber Disease Recognition Dataset\val"  # Replace with the path to validation data
NUM_CLASSES = 8  # Ensure this matches the new dataset's number of classes

# Path to save/load the model
MODEL_PATH = "cucumber_vegetable_weights.h5"

# Check if the model exists for incremental training
if os.path.exists(MODEL_PATH):
    print("Loading pre-trained model...")
    model = load_model(MODEL_PATH)

    # Modify the last layer if new classes are introduced
    if model.layers[-1].output_shape[-1] != NUM_CLASSES:
        print("Updating the output layer for new classes...")
        x = model.layers[-2].output  # Get the second-to-last layer
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=model.input, outputs=predictions)
else:
    print("No pre-trained model found. Creating a new model...")

    # Load the InceptionV3 base model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # Combine base model and custom layers
    model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Callbacks for saving the best model and early stopping
checkpoint = ModelCheckpoint(MODEL_PATH,
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max')

early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=5,
                               restore_best_weights=True)

# Unfreeze some layers for fine-tuning if applicable
if os.path.exists(MODEL_PATH):
    print("Unfreezing some layers for fine-tuning...")
    for layer in model.layers[:100]:  # Adjust based on your model's layer count
        layer.trainable = True

# Compile again if layers are unfrozen
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE / 10),  # Lower LR for fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Incremental training
print("Starting incremental training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

# Evaluate the updated model
loss, accuracy = model.evaluate(val_generator, steps=val_generator.samples // BATCH_SIZE)
print(f"Updated Validation Loss: {loss}")
print(f"Updated Validation Accuracy: {accuracy}")

# Save the updated model
model.save(MODEL_PATH)
print("Model saved after incremental training.")
