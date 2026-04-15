import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.segmentation import slic, mark_boundaries
from skimage.color import label2rgb
import os

np.random.seed(42)

# -------------------------
# Load Image
# -------------------------
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
# Hybrid Sampling (Improved)
# -------------------------
def hybrid_sampling_local(segments, ratio=0.01):
    h, w = segments.shape
    mask = np.zeros((h, w), dtype=bool)

    total_pixels = h * w
    total_samples = int(total_pixels * ratio)

    num_segments = np.max(segments)

    # Step 1: Centroids
    for i in range(1, num_segments + 1):
        coords = np.argwhere(segments == i)
        if len(coords) == 0:
            continue

        centroid = np.mean(coords, axis=0).astype(int)
        mask[centroid[0], centroid[1]] = True

    remaining = total_samples - np.sum(mask)

    # Step 2: Controlled random per segment
    max_points_per_segment = 10  

    for i in range(1, num_segments + 1):
        coords = np.argwhere(segments == i)
        if len(coords) == 0:
            continue

        num_points = min(max_points_per_segment, len(coords))

        indices = np.random.choice(len(coords), num_points, replace=False)
        selected = coords[indices]

        for pt in selected:
            mask[pt[0], pt[1]] = True

    return mask

# -------------------------
# Visualization
# -------------------------
def visualize_hybrid(image_path, n_segments=50, ratio=0.01):
    img = load_image(image_path)

    # SLIC segmentation
    segments = slic(img, n_segments=n_segments, compactness=20, start_label=1)

    # Colored superpixels
    segmented_img = label2rgb(segments, img, kind='avg')

    # 🔥 Add boundaries (optional but recommended)
    segmented_img = mark_boundaries(segmented_img, segments, color=(0, 1, 1), mode='inner')

    # Sampling
    centroid_mask = centroid_sampling(segments)
    hybrid_mask = hybrid_sampling_local(segments, ratio)

    random_mask = hybrid_mask & (~centroid_mask)

    centroids = np.argwhere(centroid_mask)
    random_pts = np.argwhere(random_mask)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_img)
    plt.title(f"Hybrid Sampling (k = {np.max(segments)}, {ratio*100:.0f}%)")
    plt.axis('off')

    # Centroids (highlighted)
    plt.scatter(centroids[:, 1], centroids[:, 0],
                c='red', s=80, marker='*',
                edgecolors='black', label='Centroids')

    # Random points (light + clean)
    plt.scatter(random_pts[:, 1], random_pts[:, 0],
                c='black', s=3, alpha=0.2, label='Random')

    plt.legend()

    plt.tight_layout()

    # Save
    save_path = 'Pictures\\Results\\Demonstration Images\\SLIC_hybrid'
    os.makedirs(save_path, exist_ok=True)

    plt.savefig(f'{save_path}\\SLIC_hybrid_final.png', dpi=300)
    plt.show()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    visualize_hybrid("Pictures\\Images\\kodim06.png", n_segments=50, ratio=0.01)