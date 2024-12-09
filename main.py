from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import io
from PIL import Image
import uvicorn  # Import Uvicorn

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust for your frontend's domain in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Directory where models are stored
MODEL_DIR = './'

# Supported crop models
CROP_MODELS = {
    "guava": "guava_fruit_weights.h5",
    "watermelon": "watermelon_fruit_weights.h5",
    "apple": "apple_fruit_weights.h5",
    "maize": "maize_crop_weights.h5",
    "brinjal": "brinjal_vegetable_weights.h5",
    "cucumber": "cucumber_vegetable_weights.h5",
    "carrot": "carrot_vegetable.h5",
    "tomato": "tomato_leaf_weights.h5"

    # Add other crop models here...
}

DIRECTORY = {
    "guava": r"E:\guava_split_diseases\val",
    "watermelon": r"E:\watermelon_dataset\val",
    "apple": r"E:\apple dataset\datasets\test",
    "maize": r"E:\maize_crop_dataset\train",
    "brinjal": r"E:\brinjal_split_diseases\val",
    "cucumber": r"E:\cucumber_split_dataset\val",
    "carrot": r"E:\carrot\Dataset\Train",
    "tomato": r"C:\Users\GOWTHAM.S\Desktop\kaggle\tomato\train"
}





@app.post("/predict")
async def predict(
        crop: str = Form(...),  # Crop name
        image: UploadFile = File(...)  # Image file
):
    try:
        print(crop)
        # Check if crop is supported
        if crop not in CROP_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported crop: {crop}")

        # Load the corresponding model
        model_path = CROP_MODELS[crop]
        print(os.path.exists(model_path))
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail=f"Model for crop {crop} not found")

        model = load_model(model_path)

        # Load and preprocess the image
        contents = await image.read()


        # Convert bytes to a file-like object
        image_stream = io.BytesIO(contents)

        try:
            img = Image.open(image_stream)
            img = img.resize((256,256))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            print("Image preprocessing successful.")
        except Exception as e:
            print(f"Error during image preprocessing: {e}")
            raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

        # Predict the class probabilities
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Map class index to label (example mapping; update with actual labels)
        validation_dir = DIRECTORY[crop]

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
        class_indices = {
            v: k for k, v in validation_generator.class_indices.items()
        }  # Reverse the mapping
        print(class_indices)
        predicted_class_label = class_indices[predicted_class_index]

        return {
            "crop": crop,
            "predicted_class_index": int(predicted_class_index),
            "predicted_class_label": predicted_class_label,
            "confidence": float(np.max(predictions))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run Uvicorn programmatically
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)


