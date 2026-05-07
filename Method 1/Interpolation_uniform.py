import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from scipy.interpolate import griddata

np.random.seed(42)

def load_image(filepath):
    img = io.imread(filepath)
    if img.ndim == 2:
        img = color.gray2rgb(img)
    return img_as_float(img)

from skimage.segmentation import slic

def sample_pixels(image, ratio=0.2):
    """Uniform grid sampling"""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    step = round(np.sqrt(1 / ratio))

    for i in range(0, h, step):
        for j in range(0, w, step):
            mask[i, j] = True

    return mask

def reconstruct_interpolation(image, mask):
    """Interpolation-based reconstruction"""
    h, w, c = image.shape

    coords = np.indices((h, w)).reshape(2, -1).T
    sampled_coords = coords[mask.flatten()]
    missing_coords = coords[~mask.flatten()]

    reconstructed = image.copy()

    for ch in range(c):
        values = image[:, :, ch].flatten()
        sampled_values = values[mask.flatten()]

        interpolated = griddata(
            sampled_coords,
            sampled_values,
            missing_coords,
            method='linear',
            fill_value=0
        )

        reconstructed[:, :, ch].flat[~mask.flatten()] = interpolated

    return np.clip(reconstructed, 0, 1)

def evaluate_reconstruction(original, reconstructed):
    mse_val = mean_squared_error(original, reconstructed)
    psnr_val = peak_signal_noise_ratio(original, reconstructed, data_range=1.0)
    ssim_val = structural_similarity(original, reconstructed, data_range=1.0, channel_axis=2)
    return mse_val, psnr_val, ssim_val

def run_pipeline(image_path, ratio=0.2):
    img = load_image(image_path)
    h, w = img.shape[:2]

    mask = sample_pixels(img, ratio)
    actual_ratio = np.sum(mask) / (h * w)

    sampled_img = np.zeros_like(img)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    sampled_img[mask_3d] = img[mask_3d]

    reconstructed = reconstruct_interpolation(img, mask)
    mse_val, psnr_val, ssim_val = evaluate_reconstruction(img, reconstructed)

    print("=== INTERPOLATION METHOD ===")
    print(f"Sampling: {actual_ratio*100:.2f}%")
    print(f"MSE:  {mse_val:.6f}")
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print("============================")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(sampled_img)
    axes[1].set_title(f"Sampled ({actual_ratio*100:.1f}%)")
    axes[1].axis('off')

    axes[2].imshow(reconstructed)
    axes[2].set_title(f"Reconstructed\nPSNR: {psnr_val:.2f} | SSIM: {ssim_val:.3f}")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('Pictures\\Results\\Method 1\\interpolation_uniform.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    run_pipeline("C:\\Users\\anish\\Downloads\\Image_Reconstruction\\Pictures\\Images\\kodim03.png", ratio=0.2)