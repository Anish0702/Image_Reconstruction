import numpy as np
from skimage.restoration import denoise_nl_means

# =========================
# DENOISER (NON-LOCAL MEANS)
# =========================
def denoiser(img):
    return denoise_nl_means(
        img,
        patch_size=5,
        patch_distance=6,
        h=0.08,
        channel_axis=2
    )

# =========================
# ADMM REFINEMENT
# =========================
def admm_refine(initial, original, mask, iterations=40):
    x = initial.copy()
    v = x.copy()
    u = np.zeros_like(x)

    mask3 = np.repeat(mask[:, :, None], 3, axis=2)

    for _ in range(iterations):

        # Step 1: Data consistency (CRUCIAL)
        x = v - u
        x[mask3] = original[mask3]

        # Step 2: Denoising prior
        v = denoiser(x + u)

        # Step 3: Dual update
        u = u + x - v

    return np.clip(x, 0, 1)