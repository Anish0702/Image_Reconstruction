import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import gaussian
import os

np.random.seed(42)

# =========================
# LOAD IMAGE
# =========================
def load_image(path):
    return img_as_float(io.imread(path))

# =========================
# HYBRID SAMPLING (YOUR STYLE)
# =========================
def hybrid_sampling(image, ratio=0.15):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    # Superpixel centroids
    segments = slic(image, n_segments=500, compactness=20, start_label=1)

    for i in range(1, np.max(segments) + 1):
        coords = np.argwhere(segments == i)
        if len(coords) == 0:
            continue
        centroid = np.mean(coords, axis=0).astype(int)
        mask[centroid[0], centroid[1]] = True

    # Fill remaining randomly
    remaining = int(h * w * ratio) - np.sum(mask)

    if remaining > 0:
        indices = np.random.choice(h * w, remaining, replace=False)
        mask.flat[indices] = True

    return mask

# =========================
# SUPERPIXEL DENOISER (YOUR CORE IDEA)
# =========================
def superpixel_denoiser(img, n_segments=800, strength=0.25):
    segments = slic(img, n_segments=n_segments, compactness=10, start_label=1)
    smooth = label2rgb(segments, img, kind='avg')

    # Blend instead of replacing (VERY IMPORTANT)
    return strength * smooth + (1 - strength) * img



def better_denoiser(img):
    sp = superpixel_denoiser(img, n_segments=600, strength=0.25)
    return gaussian(sp, sigma=0.6, channel_axis=2)
# =========================
# ADMM (PLUG-AND-PLAY)
# =========================
def admm_pnp(sparse, mask, iterations=120, rho=0.5):
    x = sparse.copy()
    v = x.copy()
    u = np.zeros_like(x)

    mask3 = np.repeat(mask[:, :, None], 3, axis=2)

    for i in range(iterations):

        # Data consistency (KEEP KNOWN PIXELS)
        x = v - u
        x[mask3] = sparse[mask3]

        # Denoising (controlled)
        v = superpixel_denoiser(x + u, n_segments=400, strength=0.4)

        # Dual update
        u = u + x - v

    return np.clip(x, 0, 1)

# =========================
# PIPELINE
# =========================
def run_pipeline(image_path, ratio=0.15):

    img = load_image(image_path)
    h, w = img.shape[:2]

    # Sampling
    mask = hybrid_sampling(img, ratio)
    actual_ratio = np.sum(mask) / (h * w)

    sparse = np.zeros_like(img)
    sparse[mask] = img[mask]

    # ADMM Reconstruction
    recon = admm_pnp(sparse, mask, iterations=60, rho=0.6)

    # Metrics
    psnr = peak_signal_noise_ratio(img, recon, data_range=1.0)
    ssim = structural_similarity(img, recon, channel_axis=2, data_range=1.0)

    print("=== METHOD 4: ADMM + SUPERPIXEL ===")
    print(f"Sampling: {actual_ratio*100:.2f}%")
    print(f"PSNR: {psnr:.2f}")
    print(f"SSIM: {ssim:.3f}")

    # =========================
    # SAVE RESULTS
    # =========================
    os.makedirs("Pictures/Results/Method 4", exist_ok=True)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(sparse)
    axes[1].set_title(f"Sampled ({actual_ratio*100:.1f}%)")
    axes[1].axis('off')

    axes[2].imshow(recon)
    axes[2].set_title(f"ADMM + Superpixel\nPSNR: {psnr:.2f} | SSIM: {ssim:.3f}")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("Pictures/Results/Method 4/admm_superpixel.png", dpi=300)
    plt.show()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_pipeline("Pictures/Images/kodim03.png", ratio=0.15)