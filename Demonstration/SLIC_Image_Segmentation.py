import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.segmentation import slic, mark_boundaries

def load_image(path):
    img = io.imread(path)
    if img.ndim == 2:
        img = color.gray2rgb(img)
    return img_as_float(img)

def visualize_slic(image_path, n_segments=200, compactness=20):
    img = load_image(image_path)

    # Apply SLIC
    segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=1)

    # Draw boundaries
    boundary_img = mark_boundaries(img, segments, color=(0, 0, 1))  # blue boundaries

    # Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(boundary_img)
    plt.title(f"SLIC Superpixels (k = {np.max(segments)})")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('Pictures\\Results\\Demonstration Images\\SLIC\\SLIC_200.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    visualize_slic("Pictures\\Images\\kodim06.png", n_segments=200)