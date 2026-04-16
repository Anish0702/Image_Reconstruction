import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.segmentation import slic
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle

np.random.seed(42)

# -------------------------
# Load Image
# -------------------------
def load_image(filepath):
    img = io.imread(filepath)
    if img.ndim == 2:
        img = color.gray2rgb(img)
    return img_as_float(img)

# -------------------------
# Segmentation
# -------------------------
def segment_slic(image, n_segments=10000, compactness=10):
    return slic(image, n_segments=n_segments, compactness=compactness, start_label=1)

# -------------------------
# Hybrid Sampling (clean)
# -------------------------
def sample_pixels(image, segments, ratio=0.2):
    h, w = segments.shape
    mask = np.zeros((h, w), dtype=bool)

    total_pixels = h * w
    total_samples = int(total_pixels * ratio)

    num_segments = np.max(segments)

    # Centroids
    for i in range(1, num_segments + 1):
        coords = np.argwhere(segments == i)
        if len(coords) == 0:
            continue
        centroid = np.mean(coords, axis=0).astype(int)
        mask[centroid[0], centroid[1]] = True

    remaining = total_samples - np.sum(mask)

    # Random fill (global)
    if remaining > 0:
        available = np.where(~mask.flatten())[0]
        chosen = np.random.choice(available, min(remaining, len(available)), replace=False)
        mask.flat[chosen] = True

    return mask

# -------------------------
# Reconstruction
# -------------------------
def reconstruct_cluster_aware(image, mask, segments):
    reconstructed = np.zeros_like(image)
    num_segments = np.max(segments)

    segment_colors = {}

    # Pass 1: mean color
    for i in range(1, num_segments + 1):
        seg_mask = (segments == i)
        sampled = seg_mask & mask

        if np.any(sampled):
            mean_color = np.mean(image[sampled], axis=0)
            reconstructed[seg_mask] = mean_color
            segment_colors[i] = mean_color
        else:
            segment_colors[i] = None

    # Pass 2: fill missing
    filled = {k: v for k, v in segment_colors.items() if v is not None}
    unfilled = [k for k, v in segment_colors.items() if v is None]

    if filled:
        keys = list(filled.keys())
        centroids = []

        for k in keys:
            coords = np.argwhere(segments == k)
            centroids.append(np.mean(coords, axis=0))

        centroids = np.array(centroids)

        for k in unfilled:
            coords = np.argwhere(segments == k)
            if len(coords) == 0:
                continue

            c = np.mean(coords, axis=0)
            dists = np.sum((centroids - c)**2, axis=1)
            nearest = keys[np.argmin(dists)]

            reconstructed[segments == k] = filled[nearest]

    return np.clip(reconstructed, 0, 1)

# -------------------------
# Metrics
# -------------------------
def evaluate(original, reconstructed):
    mse = mean_squared_error(original, reconstructed)
    psnr = peak_signal_noise_ratio(original, reconstructed, data_range=1.0)
    ssim = structural_similarity(original, reconstructed, data_range=1.0, channel_axis=2)
    return mse, psnr, ssim

# -------------------------
# Pipeline
# -------------------------
def run_pipeline(image_path, ratio=0.2):
    img = load_image(image_path)
    h, w = img.shape[:2]

    segments = segment_slic(img)

    mask = sample_pixels(img, segments, ratio)
    actual_ratio = np.sum(mask) / (h * w)

    sampled_img = np.zeros_like(img)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    sampled_img[mask_3d] = img[mask_3d]

    # Reconstruction
    reconstructed = reconstruct_cluster_aware(img, mask, segments)


    mse, psnr, ssim = evaluate(img, reconstructed)

    print("=== HYBRID + RECONSTRUCTION ===")
    print(f"Sampling: {actual_ratio*100:.2f}%")
    print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    # Plot
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(color.label2rgb(segments, img, kind='avg'))
    axes[1].set_title(f"Segments ({np.max(segments)})")
    axes[1].axis('off')

    axes[2].imshow(sampled_img)
    axes[2].set_title(f"Sampled ({actual_ratio*100:.1f}%)")
    axes[2].axis('off')

    axes[3].imshow(reconstructed)
    axes[3].set_title(f"Hybrid+Reconstructed\nPSNR: {psnr:.2f} | SSIM: {ssim:.3f}")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('Pictures\\Results\\Method 2\\results_hybrid_smoothed.png',
                dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run_pipeline("Pictures\\Images\\kodim04.png", ratio=0.2)