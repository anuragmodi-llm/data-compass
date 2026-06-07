"""
Perceptual hash module.
Generates a 63-bit visual fingerprint (pHash) for each page image using DCT,
and measures visual distance between pages using Hamming distance.
"""

import numpy as np
from PIL import Image


def compute_phash(page_image: Image.Image) -> int:
    """
    Compute a DCT-based perceptual hash for a page image.

    Steps:
        1. Resize to 32x32 greyscale (low-res captures gross visual structure)
        2. Convert to float numpy array
        3. Apply 2D DCT via scipy to convert pixel space → frequency space
        4. Take top-left 8x8 low-frequency block (64 values)
        5. Exclude index [0,0] (DC component = brightness) → 63 values
        6. Binarise: bit=1 if value > mean, bit=0 otherwise
        7. Pack bits into integer

    The hash is brightness-invariant (DC component excluded) and robust
    to minor scan quality variation.

    Args:
        page_image: PIL Image of the page (any size, any mode).

    Returns:
        63-bit integer hash.
    """
    from scipy.fft import dctn

    # Step 1: resize and greyscale
    img = page_image.resize((32, 32), Image.LANCZOS).convert("L")

    # Step 2: float array
    pixels = np.array(img, dtype=np.float32)

    # Step 3: 2D DCT
    dct = dctn(pixels, norm="ortho")

    # Step 4: top-left 8x8
    dct_low = dct[:8, :8].flatten()

    # Step 5: exclude DC component (index 0)
    dct_low = dct_low[1:]  # 63 values

    # Step 6: binarise
    mean_val = np.mean(dct_low)
    hash_bits = (dct_low > mean_val).astype(int)

    # Step 7: pack into integer
    hash_int = int("".join(map(str, hash_bits)), 2)
    return hash_int


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    Compute the Hamming distance between two pHash integers.
    Counts the number of bit positions where the two hashes differ.

    Returns:
        Integer 0–63. 0 = identical, 63 = completely different.
        Interpretation:
            0–10:  visually near-identical (same document)
            11–20: visually similar (ambiguous — check metadata score)
            21+:   visually different (likely boundary)
    """
    xor = hash1 ^ hash2
    return bin(xor).count("1")


def compute_phash_all_pages(page_images: list[Image.Image]) -> list[int]:
    """
    Compute pHash for every page image in a list.

    Args:
        page_images: List of PIL Images, 0-indexed.

    Returns:
        List of 63-bit hash integers, one per page.
    """
    return [compute_phash(img) for img in page_images]
