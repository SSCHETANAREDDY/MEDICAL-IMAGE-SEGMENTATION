import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to avoid PyCharm issues
import matplotlib.pyplot as plt
import os

def highlight_lung_part(image, mask, alpha=0.5, color=(255, 0, 0)):  # Default: Red for COVID
    """
    Highlight specific parts of the lung in the image using a transparent mask.
    """
    # Create a colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color  # Apply the specified color to the mask

    # Blend the colored mask with the original image
    highlighted_image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    return highlighted_image

def load_images_from_folder(folder, num_images=5):
    """
    Load a specified number of images from a folder.
    """
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            if len(images) >= num_images:
                break
    return images

def create_lung_masks(images):
    """
    Create masks to simulate highlighted lung parts.
    """
    masks = []
    for img in images:
        mask = np.zeros_like(img[:, :, 0])  # Create a blank mask
        h, w = mask.shape

        # Simulate lung-like shapes (elliptical regions)
        cv2.ellipse(mask, (w // 3, h // 2), (100, 150), 0, 0, 360, 255, -1)  # Left lung
        cv2.ellipse(mask, (2 * w // 3, h // 2), (100, 150), 0, 0, 360, 255, -1)  # Right lung

        masks.append(mask)
    return masks

def display_comparison(covid_images, non_covid_images, covid_masks, non_covid_masks):
    """
    Display COVID and Non-COVID images side by side with highlighted lung parts.
    """
    plt.figure(figsize=(15, 10))
    for i in range(5):
        # Display COVID-19 image with red mask
        plt.subplot(2, 5, i + 1)
        covid_highlighted = highlight_lung_part(covid_images[i], covid_masks[i], color=(255, 0, 0))  # Red for COVID
        plt.imshow(cv2.cvtColor(covid_highlighted, cv2.COLOR_BGR2RGB))
        plt.title(f"COVID-19 {i + 1}")
        plt.axis('off')

        # Display Non-COVID image with light blue mask
        plt.subplot(2, 5, i + 6)
        non_covid_highlighted = highlight_lung_part(non_covid_images[i], non_covid_masks[i], color=(173, 216, 230))  # Light blue for Non-COVID
        plt.imshow(cv2.cvtColor(non_covid_highlighted, cv2.COLOR_BGR2RGB))
        plt.title(f"Non-COVID {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main(covid_folder, non_covid_folder):
    # Load 5 COVID-19 images
    covid_images = load_images_from_folder(covid_folder, num_images=5)
    # Load 5 Non-COVID images
    non_covid_images = load_images_from_folder(non_covid_folder, num_images=5)

    # Create lung masks for COVID-19 and Non-COVID images
    covid_masks = create_lung_masks(covid_images)
    non_covid_masks = create_lung_masks(non_covid_images)

    # Display the comparison
    display_comparison(covid_images, non_covid_images, covid_masks, non_covid_masks)

if __name__ == "__main__":
    # Provide the paths to the folders containing COVID-19 and Non-COVID images
    covid_folder = "COVID-QU-Ex/Lung_Segmentation_Data/Lung Segmentation Data/Val/COVID-19/images"
    non_covid_folder = "COVID-QU-Ex/Lung_Segmentation_Data/Lung Segmentation Data/Val/Non-COVID/images"
    main(covid_folder, non_covid_folder)
