import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.segmentation import slic
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

np.random.seed(42)

# ========================
# SEGMENTATION
# ========================
def segment(img):
    return slic(img, n_segments=3000, compactness=10)

# ========================
# SAMPLING
# ========================
def uniform_sampling(img, ratio=0.15):
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=bool)

    step = int(1 / np.sqrt(ratio))  # spacing
    mask[::step, ::step] = True

    return mask

# ========================
# LOW PASS FILTER (IMPROVED)
# ========================
def low_pass(shape, r=120):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cx, cy = h//2, w//2
    mask = ((y - cx)**2 + (x - cy)**2 <= r**2).astype(float)
    return mask

# ========================
# DFT RECONSTRUCTION (FIXED)
# ========================
def dft_reconstruct(sparse, mask):
    out = np.zeros_like(sparse)
    lp_mask = low_pass(sparse.shape[:2])

    for c in range(3):
        channel = sparse[:,:,c]

        # FFT
        F = np.fft.fftshift(np.fft.fft2(channel))

        # Apply low-pass
        F_filtered = F * lp_mask

        # Inverse FFT
        recon = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))

        # NORMALIZATION (IMPORTANT)
        recon = recon - recon.min()
        recon = recon / (recon.max() + 1e-8)

        out[:,:,c] = recon

    return np.clip(out, 0, 1)

# ========================
# MAIN
# ========================
img_path = "Pictures\\512x512.2.jpg"
img = imread(img_path) / 255.0

seg = segment(img)
mask = uniform_sampling(img)

# Sparse image (NO FILLING)
sparse = np.zeros_like(img)
sparse[mask] = img[mask]

# DFT reconstruction directly
recon = dft_reconstruct(sparse, mask)

# ========================
# METRICS
# ========================
psnr = peak_signal_noise_ratio(img, recon, data_range=1.0)
ssim = structural_similarity(img, recon, channel_axis=2, data_range=1.0)

print("PSNR:", psnr)
print("SSIM:", ssim)

# ========================
# SAVE IMAGES
# ========================
os.makedirs("Pictures\\Results\\Method 3\\results_dft_uniform", exist_ok=True)

imsave("Pictures\\Results\\Method 3\\results_dft_uniform/original.png", (img*255).astype(np.uint8))
imsave("Pictures\\Results\\Method 3\\results_dft_uniform/sampled.png", (sparse*255).astype(np.uint8))
imsave("Pictures\\Results\\Method 3\\results_dft_uniform/reconstructed.png", (recon*255).astype(np.uint8))

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
plt.title("Sampled (15%)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(recon)
plt.title(f"DFT Recon\nPSNR:{psnr:.2f} SSIM:{ssim:.3f}")
plt.axis("off")

plt.tight_layout()
plt.show()