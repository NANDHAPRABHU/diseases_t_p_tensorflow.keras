import os
import shutil
from sklearn.model_selection import train_test_split

# Step 1: Define paths
dataset_dir = r"E:\cucumber_dataset\Cucumber Disease Recognition Dataset"  # Path to the dataset
output_dir = r"E:\cucumber_split_dataset"  # Output directory for the split dataset

# Create training and validation directories
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Step 2: Iterate over all subdirectories to find images
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

for root, dirs, files in os.walk(dataset_dir):
    # Filter image files
    images = [f for f in files if f.lower().endswith(valid_extensions)]

    # Skip folders with no images
    if not images:
        continue

    # Get the relative path to identify the class (e.g., `augmented/class_name`)
    relative_path = os.path.relpath(root, dataset_dir)
    class_name = os.path.basename(relative_path)  # Use the deepest folder as the class name

    # Split images into 80% training and 20% validation
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    # Create class subdirectories in train and validation directories
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # Copy training images
    for img in train_images:
        src = os.path.join(root, img)
        dst = os.path.join(train_class_dir, img)
        shutil.copy(src, dst)

    # Copy validation images
    for img in val_images:
        src = os.path.join(root, img)
        dst = os.path.join(val_class_dir, img)
        shutil.copy(src, dst)

    print(f"Processed class '{class_name}' from '{relative_path}'")

print("Dataset split completed!")
