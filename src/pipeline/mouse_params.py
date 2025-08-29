import numpy as np

# Allen Neuropixels Visual Coding geometry for mice:
# - Monocular (right eye), monitor 15 cm from eye
# - Spans 120° (H) × 95° (V) of visual space, spherical warping applied
MONITOR_DISTANCE_CM = 15.0
FOV_DEG = (120.0, 95.0)  # (horizontal, vertical)

# Default motion ranges (roll rotation; yaw/pitch -> x/y translation as fraction of image dim)
# Literature suggests small head/eye movements in head-fixed prep; keep conservative but non-zero defaults.

"""
Saccades occurred in all directions yet were biased along the horizontal as compared with the vertical axis 
(21,981 horizontal and 12,550 vertical saccades from 10 animals; binomial test, P < 0.0001; Fig 1d). They had a 
mean amplitude of 19.0 +- 1.6 degrees.
 https://www.nature.com/articles/s41586-022-05196-w?utm_source=chatgpt.com


 an image is 64x64 pixels, so 64/120 = 0.533 pixels/degrees.
 19.0 degrees * 0.533 pixels/degrees = 10.127 pixels.
 64/95 = 0.6737 pixels/degrees so 19.0 degrees * 0.6737 pixels/degrees = 12.8 pixels.
 
 so the translations as a fraction of the image size are:
 (10.127/64, 12.8/64) = (0.158, 0.200)
"""


DEFAULT_ROLL_DEG = 0.0            # ± degrees
DEFAULT_TRANSLATE = (0, 0)#(0.158, 0.200)  # (x frac, y frac) 

# Keys are spatial frequency in cycles/deg; values are contrast thresholds (Michelson, 0..1).
# SF are from Prusky & Douglas 2006, thresholds match various papers.
MOUSE_CSF_TARGET = {
    
    0.031: 0.23,
   
    0.066: 0.04,  
    0.098: 0.05,
    
    0.11: 0.07,
    0.22:0.105,
    0.31:0.23
}

# Search grids for blur/noise fitting (sigma in pixels at pre-warp resolution; noise is Gaussian std in [0,1] tensor space)
BLUR_SIGMA_GRID = np.array([0, 1, 2, 4, 8,])     # px - uniformly sampled discrete values
NOISE_STD_GRID  = np.linspace(0.01, 0.5, 8)   # unitless (tensor scale)

# Grating detection fitting defaults
N_SAMPLES_PER_CLASS = 5000     # per SF per contrast (keep reasonable to avoid long runs; increase if you want)
CONTRAST_SWEEP = np.linspace(0, 1, 8)    # Michelson contrasts for psychometric curves
PATCH_SIZE = 28                # patch side in pixels for patchwise std feature
THRESH_CRITERION = 0.75        # 50% correct criterion for threshold
SEED = 1234