"""Pictures for the demo panels: mask outlines, the mask itself, and a heat map of |repaired - clean|."""
import cv2
import numpy as np
from scipy import ndimage

TRUE_COLOR, PREDICTED_COLOR = (230, 40, 40), (0, 190, 230)     # red = true damage, cyan = segmenter


def edge(mask: np.ndarray, width: int = 2) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, iterations=width)


def outlined(image: np.ndarray, true_mask: np.ndarray, predicted: np.ndarray | None = None) -> np.ndarray:
    """RGB uint8: the grey print with the true mask outlined in red and the predicted one in cyan."""
    rgb = np.repeat(image[..., None], 3, axis=2)
    rgb[edge(true_mask)] = TRUE_COLOR
    if predicted is not None:
        rgb[edge(predicted)] = PREDICTED_COLOR
    return rgb


def mask_picture(used: np.ndarray, true_mask: np.ndarray) -> np.ndarray:
    """RGB uint8 of the mask the model was given (white), with the true mask's outline in red."""
    rgb = np.repeat((used.astype(np.uint8) * 255)[..., None], 3, axis=2)
    rgb[edge(true_mask)] = TRUE_COLOR
    return rgb


def difference(repaired_n: np.ndarray, clean_n: np.ndarray, true_mask: np.ndarray) -> np.ndarray:
    """RGB heat map of |repaired - clean| in the evaluator's (normalised) view; true mask outlined in white."""
    diff = np.abs(repaired_n.astype(np.int16) - clean_n.astype(np.int16))
    heat = cv2.applyColorMap(np.clip(diff * 2, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat[edge(true_mask)] = (255, 255, 255)
    return heat
