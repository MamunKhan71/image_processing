import cv2
import os

# Function to apply high-pass filter
def apply_high_pass_filter(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Laplacian filter
    high_pass = cv2.Laplacian(gray, cv2.CV_64F)

    # Normalize the Laplacian to the 0-255 range
    high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8 type
    return cv2.convertScaleAbs(high_pass)

# Function to apply low-pass filter
def apply_low_pass_filter(image):
    low_pass = cv2.GaussianBlur(image, (5, 5), 0)
    return low_pass

# Function to apply histogram equalization
def apply_histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

# Function to apply contrast stretching
def apply_contrast_stretching(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply contrast stretching
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    stretched = cv2.convertScaleAbs(gray, alpha=255.0/(max_val-min_val), beta=-min_val * 255.0/(max_val-min_val))

    # Convert to BGR type
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)

# Function to process images in the dataset
def process_images(dataset_folder, output_folder):
    for class_folder in os.listdir(dataset_folder):
        if class_folder in ['.idea', 'venv']:
            continue  # Skip specified folders

        class_folder_path = os.path.join(dataset_folder, class_folder)

        # Ensure the path is a directory
        if os.path.isdir(class_folder_path):
            # Create output folders if they don't exist for each class
            os.makedirs(os.path.join(output_folder, "high_pass", class_folder), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "low_pass", class_folder), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "histogram_equalization", class_folder), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "contrast_stretching", class_folder), exist_ok=True)

            # Target the "image" subfolder in each class folder
            image_folder_path = os.path.join(class_folder_path, "image")

            # Counter for processing only the first 1000 images
            count = 0

            for filename in os.listdir(image_folder_path):
                file_path = os.path.join(image_folder_path, filename)

                # Read the image
                image = cv2.imread(file_path)

                # Check if the image is read correctly
                if image is None:
                    print("Error: Unable to read the image at", file_path)
                    continue  # Skip to the next iteration if there's an issue with the image

                # Apply high-pass filter for the first 1000 images
                if count < 1000:
                    high_pass_result = apply_high_pass_filter(image)
                    low_pass_result = apply_low_pass_filter(image)
                    contrast_stretching_result = apply_contrast_stretching(image)
                else:
                    high_pass_result = None  # No filtering for remaining images
                    low_pass_result = None
                    contrast_stretching_result = None

                # Apply histogram equalization for all images
                equalized_result = apply_histogram_equalization(image)

                # Save the results sequentially
                if high_pass_result is not None:
                    cv2.imwrite(os.path.join(output_folder, "high_pass", class_folder, filename),
                                high_pass_result)
                if low_pass_result is not None:
                    cv2.imwrite(os.path.join(output_folder, "low_pass", class_folder, filename),
                                low_pass_result)
                if contrast_stretching_result is not None:
                    cv2.imwrite(os.path.join(output_folder, "contrast_stretching", class_folder, filename),
                                contrast_stretching_result)

                cv2.imwrite(os.path.join(output_folder, "histogram_equalization", class_folder, filename),
                            equalized_result)

                count += 1

# Main function
if __name__ == "__main__":
    dataset_folder = 'D:/ChestXray'
    output_folder = 'D:/ChestXray/output'

    # Process images in the dataset
    process_images(dataset_folder, output_folder)
