import tensorflow as tf
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Configuration
IMG_SIZE = (256, 256)  # InceptionV3 input size
BATCH_SIZE = 36

EPOCHS = 10
LEARNING_RATE = 0.0001
TRAIN_DIR = r"E:\maize_crop_dataset\train"# Replace with the path to your training data
VAL_DIR = r"E:\maize_crop_dataset\test"# Replace with the path to your validation data
NUM_CLASSES =6 # Replace with the number of classes in your dataset

# Load the InceptionV3 model pre-trained on ImageNet, excluding the top layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
print(base_model.summary())
# Freeze the base model layers to use for feature extraction

for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global Average Pooling
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)   # Fully connected layer
predictions = Dense(NUM_CLASSES, activation='softmax')(x)  # Output layer

# Combine base model and custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

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
# checkpoint = ModelCheckpoint('cow_livestock_weights.h5',
#                              monitor='val_accuracy',
#                              save_best_only=True,
#                              mode='max')

# Callbacks for saving the best model and early stopping
checkpoint = ModelCheckpoint(
    'maize_crop_model.keras',  # Correct extension for full model saving
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)





early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=5,
                               restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',

              metrics=['accuracy'])



#model.save("cow_livestock_weights.h5")
model.save("maize_crop_model.keras")
# Final evaluation
loss, accuracy = model.evaluate(val_generator)
print(f"Final Validation Loss: {loss}")
print(f"Final Validation Accuracy: {accuracy}")


#.......................................................................................................................................


