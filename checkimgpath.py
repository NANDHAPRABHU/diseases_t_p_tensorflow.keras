import os

image_path = r"C:\Users\GOWTHAM.S\Desktop\Normal_Skin_73.png"  # Replace with your path
if os.path.exists(image_path):
    print("Path is valid!")
else:
    print("Invalid path. Please check the file location.")
