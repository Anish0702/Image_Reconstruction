import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.segmentation import slic

np.random.seed(42)

def load_image(path):
    img = io.imread(path)
    if img.ndim == 2:
        img = color.gray2rgb(img)
    return img_as_float(img)

# -------------------------
# Centroid Sampling
# -------------------------
def centroid_sampling(segments):
    h, w = segments.shape
    mask = np.zeros((h, w), dtype=bool)

    for i in range(1, np.max(segments) + 1):
        coords = np.argwhere(segments == i)
        if len(coords) == 0:
            continue

        centroid = np.mean(coords, axis=0).astype(int)
        mask[centroid[0], centroid[1]] = True

    return mask

# -------------------------
# Hybrid Sampling
# -------------------------
def hybrid_sampling(segments, ratio=0.5):
    h, w = segments.shape
    mask = centroid_sampling(segments)

    remaining = int(h * w * ratio) - np.sum(mask)

    if remaining > 0:
        indices = np.random.choice(h * w, remaining, replace=False)
        mask.flat[indices] = True

    return mask

# -------------------------
# Apply mask
# -------------------------
def apply_mask(image, mask):
    masked = np.zeros_like(image)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked[mask_3d] = image[mask_3d]
    return masked

# -------------------------
# Visualization
# -------------------------
def compare_sampling(image_path, ratio=0.5, n_segments=120000):
    img = load_image(image_path)

    segments = slic(img, n_segments=n_segments, compactness=20, start_label=1)

    mask_centroid = centroid_sampling(segments)
    mask_hybrid = hybrid_sampling(segments, ratio)

    centroid_img = apply_mask(img, mask_centroid)
    hybrid_img = apply_mask(img, mask_hybrid)

    r1 = np.sum(mask_centroid) / (img.shape[0]*img.shape[1])
    r2 = np.sum(mask_hybrid) / (img.shape[0]*img.shape[1])

    # Plot
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(centroid_img)
    plt.title(f"{r1*100:.2f}% Centroid Sampling")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(hybrid_img)
    plt.title(f"{r2*100:.2f}% Hybrid Sampling")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("Pictures\\Results\\Demonstration Images\\SLIC_Sampling\\centroid_vs_hybrid.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    compare_sampling("Pictures\\Images\\kodim06.png", ratio=0.5, n_segments=120000)