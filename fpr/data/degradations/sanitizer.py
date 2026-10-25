"""Sanitizer / dry skin: ridges thinned, broken into fragments and lighter.

Alcohol sanitizer dries the skin, so ridges lose contact with the platen and
print as faint, broken dashes; at high severity over the whole fingertip.
Registered in degradations/__init__.py.
"""
import numpy as np
from scipy import ndimage

import config
from fpr.data.degradations import shapes


def degrade(image: np.ndarray, fg_mask: np.ndarray, severity: float,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Thin, fragment and fade ridges inside one large noisy blob.

    Input: uint8 (H, W) clean print, bool (H, W) foreground, severity in [0, 1].
    Output: (degraded uint8 (H, W), damage_mask bool (H, W)).
    Severity scales blob size (whole fingertip at 1), thinning, the share of
    pixels broken (25% -> 65%, varied by a low-frequency field) and fading (-> 0.5).
    Gaps take a light grey (config.SANITIZER_GAP_GREY), never pure white.
    """
    radius = shapes.lerp(config.SANITIZER_RADIUS_PX, severity) * rng.uniform(0.9, 1.0)
    alpha = shapes.soft_blob(fg_mask, radius, rng)

    thin = int(round(shapes.lerp(config.SANITIZER_THIN_PX, severity)))
    dry = ndimage.grey_dilation(image, footprint=shapes.disk(thin)).astype(np.float32)
    # Fade first, so the gaps added next keep their capped grey (never pure white).
    dry = 255 - (255 - dry) * shapes.lerp(config.SANITIZER_FADE, severity)

    # Dryness is patchy: a low-frequency field raises or lowers the local gap share.
    patchy = shapes.smooth_noise(image.shape, config.SANITIZER_BREAK_LF_SIGMA_PX, rng)
    fraction = shapes.lerp(config.SANITIZER_BREAK_FRACTION, severity)
    probability = np.clip(fraction * (1 + config.SANITIZER_BREAK_MODULATION * np.clip(patchy, -1, 1)),
                          0, config.SANITIZER_BREAK_MAX)
    dry = shapes.ridge_dropout(dry, probability, rng, max_grey=config.SANITIZER_GAP_GREY)
    return shapes.finish(image, dry, alpha, fg_mask)
