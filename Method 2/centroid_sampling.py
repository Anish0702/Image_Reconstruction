import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.segmentation import slic
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle

# Set random seed for reproducibility
np.random.seed(42)

def load_image(filepath):
    img = io.imread(filepath)
    if img.ndim == 2:
        img = color.gray2rgb(img)
    return img_as_float(img)

def segment_slic(image, n_segments, compactness=20):
    return slic(image, n_segments=n_segments, compactness=compactness, start_label=1)

def sample_pixels(image, segments, ratio=0.6):
    """Centroid Sampling
    Picks 1 pixel per superpixel (centroid).
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    num_segments = np.max(segments)

    for i in range(1, num_segments + 1):
        coords = np.argwhere(segments == i)
        if len(coords) == 0:
            continue
        centroid = np.mean(coords, axis=0).astype(int)
        mask[centroid[0], centroid[1]] = True

    return mask

def reconstruct_cluster_aware(image, mask, segments):
    """
    Cluster-aware reconstruction:
    Fills each superpixel with the mean color of its sampled pixels.
    If no sampled pixels fall within a superpixel, it uses the nearest neighboring filled superpixel's color.
    """
    reconstructed = np.zeros_like(image)
    num_segments = np.max(segments)
    
    segment_colors = {}
    mask_2d = mask[:,:,0] if mask.ndim == 3 else mask
    
    # Pass 1: Compute mean color from correctly sampled pixels
    for i in range(1, num_segments + 1):
        seg_mask = (segments == i)
        if not np.any(seg_mask):
            continue
            
        sampled_in_seg_mask = seg_mask & mask_2d
        
        if np.any(sampled_in_seg_mask):
            mean_color = np.mean(image[sampled_in_seg_mask], axis=0)
            reconstructed[seg_mask] = mean_color
            segment_colors[i] = mean_color
        else:
            segment_colors[i] = None
            
    # Pass 2: Fallback assignment for superpixels with 0 samples
    unfilled_segments = [k for k, v in segment_colors.items() if v is None]
    filled_segments = [k for k, v in segment_colors.items() if v is not None]
    
    if unfilled_segments and filled_segments:
        # Cache centroids of valid segments
        filled_centroids = []
        for k in filled_segments:
            seg_mask = (segments == k)
            coords = np.argwhere(seg_mask)
            filled_centroids.append(np.mean(coords, axis=0))
        filled_centroids = np.array(filled_centroids)
        
        # Extrapolate for each unfilled segment
        for k in unfilled_segments:
            seg_mask = (segments == k)
            coords = np.argwhere(seg_mask)
            if len(coords) == 0: continue
            
            c_k = np.mean(coords, axis=0)
            distances = np.sum((filled_centroids - c_k)**2, axis=1) 
            nearest_idx = np.argmin(distances)
            nearest_color = segment_colors[filled_segments[nearest_idx]]
            
            reconstructed[seg_mask] = nearest_color
            
    return np.clip(reconstructed, 0, 1)

def evaluate_reconstruction(original, reconstructed):
    mse_val = mean_squared_error(original, reconstructed)
    psnr_val = peak_signal_noise_ratio(original, reconstructed, data_range=1.0)
    ssim_val = structural_similarity(original, reconstructed, data_range=1.0, channel_axis=2)
    return mse_val, psnr_val, ssim_val

def run_pipeline(image_path, ratio=0.6):
    img = load_image(image_path)
    h, w = img.shape[:2]
    
    n_segments = 30000
    segments = segment_slic(img, n_segments=n_segments)
    
    mask = sample_pixels(img, segments, ratio)
    actual_ratio = np.sum(mask[:,:,0] if mask.ndim==3 else mask) / (h * w)
    
    sampled_img = np.zeros_like(img)
    mask_3d = mask if mask.ndim == 3 else np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)
    sampled_img[mask_3d] = img[mask_3d]
    
    reconstructed = reconstruct_cluster_aware(img, mask, segments)
    mse_val, psnr_val, ssim_val = evaluate_reconstruction(img, reconstructed)
    
    print("=== CENTROID SAMPLING (CLUSTER-AWARE RECON) ===")
    print(f"Target Ratio: {ratio*100:.1f}%")
    print(f"Actual Ratio: {actual_ratio*100:.2f}%")
    print(f"MSE:  {mse_val:.6f}")
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print("=============================================")
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(color.label2rgb(segments, img, kind='avg'))
    axes[1].set_title(f'Segmented ({np.max(segments)} regions)')
    axes[1].axis('off')
    
    axes[2].imshow(sampled_img)
    axes[2].set_title(f'Sampled ({actual_ratio*100:.1f}%)')
    axes[2].axis('off')
    
    axes[3].imshow(reconstructed)
    axes[3].set_title(f'Reconstructed\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.3f}')
    axes[3].axis('off')
    
    plt.suptitle("Centroid Sampling + Cluster Reconstruction")
    plt.tight_layout()
    plt.savefig('Pictures\\Results\\Method 2\\results_centroid.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    import skimage.data
    import os
    sample_path = "C:\\Users\\Win 11\\Downloads\\Image_Reconstruction\\Pictures\\Images\\kodim04.png"
    if not os.path.exists(sample_path):
        io.imsave(sample_path, skimage.data.astronaut())
    run_pipeline(sample_path, ratio=0.6)
