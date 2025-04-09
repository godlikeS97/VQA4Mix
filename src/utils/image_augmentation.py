"""
Image augmentation utilities for VQA4Mix project.
"""

import cv2
import numpy as np
import skimage.io as io
from PIL import Image, ImageEnhance, ImageFilter

def load_image(image_path):
    """
    Load an image from a file path.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        numpy.ndarray: The loaded image.
    """
    return io.imread(image_path)

def apply_brightness_contrast(image, brightness=1.0, contrast=1.0):
    """
    Apply brightness and contrast adjustments to an image.
    
    Args:
        image (numpy.ndarray): The input image.
        brightness (float): Brightness factor (1.0 means no change).
        contrast (float): Contrast factor (1.0 means no change).
        
    Returns:
        numpy.ndarray: The adjusted image.
    """
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Apply brightness adjustment
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness)
    
    # Apply contrast adjustment
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast)
    
    # Convert back to numpy array
    return np.array(pil_image)

def apply_blur(image, radius=2.0):
    """
    Apply Gaussian blur to an image.
    
    Args:
        image (numpy.ndarray): The input image.
        radius (float): Blur radius.
        
    Returns:
        numpy.ndarray: The blurred image.
    """
    pil_image = Image.fromarray(image)
    pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(pil_image)

def apply_noise(image, var=0.01):
    """
    Apply Gaussian noise to an image.
    
    Args:
        image (numpy.ndarray): The input image.
        var (float): Variance of the Gaussian noise.
        
    Returns:
        numpy.ndarray: The noisy image.
    """
    row, col, ch = image.shape
    mean = 0
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_rotation(image, angle=10):
    """
    Apply rotation to an image.
    
    Args:
        image (numpy.ndarray): The input image.
        angle (float): Rotation angle in degrees.
        
    Returns:
        numpy.ndarray: The rotated image.
    """
    pil_image = Image.fromarray(image)
    pil_image = pil_image.rotate(angle, expand=True)
    return np.array(pil_image)

def apply_augmentations(image, augmentations=None):
    """
    Apply a series of augmentations to an image.
    
    Args:
        image (numpy.ndarray): The input image.
        augmentations (list): List of augmentation functions to apply.
        
    Returns:
        numpy.ndarray: The augmented image.
    """
    if augmentations is None:
        # Default augmentations
        augmented = apply_brightness_contrast(image, brightness=1.2, contrast=1.1)
        augmented = apply_blur(augmented, radius=1.0)
    else:
        augmented = image.copy()
        for augmentation_func, params in augmentations:
            augmented = augmentation_func(augmented, **params)
    
    return augmented
