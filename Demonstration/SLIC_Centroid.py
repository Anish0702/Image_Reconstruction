import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.segmentation import slic
from skimage.color import label2rgb

def load_image(path):
    img = io.imread(path)
    if img.ndim == 2:
        img = color.gray2rgb(img)
    return img_as_float(img)

def visualize_centroids(image_path, n_segments=50, compactness=20):
    img = load_image(image_path)

    # Apply SLIC
    segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=1)

    # Create colored superpixel image
    segmented_img = label2rgb(segments, img, kind='avg')

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_img)
    plt.title(f"Superpixels with Centroids (k = {np.max(segments)})")
    plt.axis('off')

    # Compute and plot centroids
    num_segments = np.max(segments)

    for i in range(1, num_segments + 1):
        coords = np.argwhere(segments == i)
        if len(coords) == 0:
            continue

        centroid = np.mean(coords, axis=0)

        # Plot centroid as black star
        plt.plot(centroid[1], centroid[0], 'k*', markersize=6)

    plt.tight_layout()
    plt.savefig('Pictures\\Results\\Demonstration Images\\SLIC_Centroid\\SLIC_Centroid_50.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    visualize_centroids("Pictures\\Images\\kodim06.png", n_segments=50)