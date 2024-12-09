import os
import pandas as pd
import shutil

# Step 1: Load the first sheet of the Excel file
excel_file_path = r'E:\Tomato Data set\Dataset anotations.xls'  # Path to your Excel file

# Load the first sheet explicitly
df = pd.read_excel(excel_file_path, sheet_name="Annotations")

# Unmerge cells by filling down missing values
df = df.ffill()  # Forward-fill to handle merged cells

# Optional: Check the first few rows of the dataframe to understand the structure
print(df.head())

# Step 2: Set the directory paths
image_directory = r"E:\Tomato Data set\Manualy Segmented"  # Folder containing the images
output_directory = r"E:\tomato_fruit_dataset"  # Directory where images will be segregated into class folders

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Step 3: Loop through the dataframe and move images into respective class folders
# Assuming column names in the Excel file are "Image Name" and "Class Name"
for index, row in df.iterrows():
    try:
         if(index>3):   # Replace column names below with actual column names from your Excel file
            image_name = row['Unnamed: 3']  # Replace 'Image Name' with the correct column header
            class_name = row['Unnamed: 9']  # Replace 'Class Name' with the correct column header

            # Full path of the image
            image_path = os.path.join(image_directory, f"{image_name}.png")  # Update extension if needed
            class_folder = os.path.join(output_directory, class_name)

            # Create a directory for the class if it doesn't exist
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            # Move the image to the corresponding class folder
            if os.path.exists(image_path):
                shutil.move(image_path, os.path.join(class_folder, f"{image_name}.png"))
            else:
                print(f"Image {image_name}.png not found in the source directory.")
    except KeyError as e:
        print(f"Column missing: {e}")
    except Exception as e:
        print(f"Error processing row {index}: {e}")

print("Image segregation completed!")
