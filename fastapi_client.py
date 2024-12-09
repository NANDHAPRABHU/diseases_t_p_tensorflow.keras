import requests

# Define the server URL
url = "http://127.0.0.1:8080/predict"

# Define the parameters
crop = "guava"
image_path = r"E:\guava_split_diseases\val\rust\aug_3.jpg"

# Open the image file
with open(image_path, "rb") as image_file:
    # Prepare the data for the POST request
    files = {
        "image": image_file,
    }
    data = {
        "crop": crop
    }

    # Send the POST request
    response = requests.post(url, files=files, data=data)

    # Handle the response
    if response.status_code == 200:
        prediction = response.json()
        print("Prediction result:", prediction)
    else:
        print(f"Error: {response.status_code} - {response.text}")
