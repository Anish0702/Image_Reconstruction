import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

np.random.seed(42)

# ========================
# SAMPLING (use your best one)
# ========================
def random_sampling(image, ratio=0.15):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    indices = np.random.choice(h * w, int(h * w * ratio), replace=False)
    mask.flat[indices] = True

    return mask

# ========================
# GRADIENT OPERATORS
# ========================
def gradient(x):
    gx = np.roll(x, -1, axis=1) - x
    gy = np.roll(x, -1, axis=0) - x
    return gx, gy

def divergence(gx, gy):
    dx = gx - np.roll(gx, 1, axis=1)
    dy = gy - np.roll(gy, 1, axis=0)
    return dx + dy

# ========================
# SHRINKAGE (TV)
# ========================
def shrink(gx, gy, lam):
    mag = np.sqrt(gx**2 + gy**2) + 1e-8
    scale = np.maximum(0, mag - lam) / mag
    return gx * scale, gy * scale

# ========================
# ADMM RECONSTRUCTION
# ========================
def admm_reconstruction(image, mask, lam=0.1, rho=1.0, iterations=100):
    h, w, c = image.shape

    # Initialize
    x = image.copy()
    z1 = np.zeros_like(image)
    z2 = np.zeros_like(image)
    u1 = np.zeros_like(image)
    u2 = np.zeros_like(image)

    mask3 = np.repeat(mask[:, :, None], 3, axis=2)

    for _ in range(iterations):

        # ---- x update ----
        div = divergence(z1 - u1, z2 - u2)
        x = (mask3 * image + rho * div) / (mask3 + rho)

        # ---- gradient ----
        gx, gy = gradient(x)

        # ---- z update (TV shrinkage) ----
        z1, z2 = shrink(gx + u1, gy + u2, lam / rho)

        # ---- dual update ----
        u1 += gx - z1
        u2 += gy - z2

    return np.clip(x, 0, 1)

# ========================
# MAIN PIPELINE
# ========================
img_path = "Pictures/Images/kodim03.png"
img = imread(img_path) / 255.0

mask = random_sampling(img, ratio=0.15)

# Sparse image
sparse = np.zeros_like(img)
sparse[mask] = img[mask]

# ADMM reconstruction
recon = admm_reconstruction(img, mask, lam=0.08, rho=1.0, iterations=100)

# ========================
# METRICS
# ========================
psnr = peak_signal_noise_ratio(img, recon, data_range=1.0)
ssim = structural_similarity(img, recon, channel_axis=2, data_range=1.0)

print("=== ADMM RECONSTRUCTION ===")
print("PSNR:", psnr)
print("SSIM:", ssim)

# ========================
# DISPLAY
# ========================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(sparse)
plt.title("Sampled")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(recon)
plt.title(f"ADMM\nPSNR:{psnr:.2f} SSIM:{ssim:.3f}")
plt.axis("off")

plt.tight_layout()
plt.show()