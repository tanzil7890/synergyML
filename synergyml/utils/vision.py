"""Utility functions for vision tasks."""

import base64
from io import BytesIO
from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageChops


def load_image(image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """Load image from various input types.
    
    Parameters
    ----------
    image : Union[str, Image.Image, np.ndarray]
        Input image as file path, PIL Image, or numpy array
        
    Returns
    -------
    Image.Image
        PIL Image object
    """
    if isinstance(image, str):
        return Image.open(image)
    elif isinstance(image, np.ndarray):
        return Image.fromarray(image)
    elif isinstance(image, Image.Image):
        return image
    else:
        raise ValueError(
            f"Unsupported image type: {type(image)}. "
            "Must be file path, PIL Image, or numpy array."
        )


def encode_image(image: Union[str, Image.Image, np.ndarray], format: str = "PNG") -> str:
    """Encode image to base64 string.
    
    Parameters
    ----------
    image : Union[str, Image.Image, np.ndarray]
        Input image
    format : str, optional
        Image format for encoding, by default "PNG"
        
    Returns
    -------
    str
        Base64 encoded image
    """
    img = load_image(image)
    buffered = BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()


def resize_image(
    image: Union[str, Image.Image, np.ndarray],
    size: Tuple[int, int],
    keep_aspect: bool = True
) -> Image.Image:
    """Resize image while optionally maintaining aspect ratio.
    
    Parameters
    ----------
    image : Union[str, Image.Image, np.ndarray]
        Input image
    size : Tuple[int, int]
        Target size (width, height)
    keep_aspect : bool, optional
        Whether to maintain aspect ratio, by default True
        
    Returns
    -------
    Image.Image
        Resized image
    """
    img = load_image(image)
    if keep_aspect:
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return img
    return img.resize(size, Image.Resampling.LANCZOS)


def normalize_image(
    image: Union[str, Image.Image, np.ndarray],
    mean: Optional[Union[float, Tuple[float, ...]]] = None,
    std: Optional[Union[float, Tuple[float, ...]]] = None
) -> np.ndarray:
    """Normalize image pixel values.
    
    Parameters
    ----------
    image : Union[str, Image.Image, np.ndarray]
        Input image
    mean : Optional[Union[float, Tuple[float, ...]]], optional
        Mean for normalization, by default None
    std : Optional[Union[float, Tuple[float, ...]]], optional
        Standard deviation for normalization, by default None
        
    Returns
    -------
    np.ndarray
        Normalized image array
    """
    img = load_image(image)
    img_array = np.array(img).astype(np.float32)
    
    if mean is None:
        mean = img_array.mean()
    if std is None:
        std = img_array.std()
        
    return (img_array - mean) / std


def validate_image(
    image: Union[str, Image.Image, np.ndarray],
    min_size: Optional[Tuple[int, int]] = None,
    max_size: Optional[Tuple[int, int]] = None,
    allowed_formats: Optional[Tuple[str, ...]] = None
) -> bool:
    """Validate image properties.
    
    Parameters
    ----------
    image : Union[str, Image.Image, np.ndarray]
        Input image
    min_size : Optional[Tuple[int, int]], optional
        Minimum allowed size (width, height), by default None
    max_size : Optional[Tuple[int, int]], optional
        Maximum allowed size (width, height), by default None
    allowed_formats : Optional[Tuple[str, ...]], optional
        Allowed image formats, by default None
        
    Returns
    -------
    bool
        Whether the image is valid
    """
    try:
        img = load_image(image)
        
        if min_size is not None:
            if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                return False
                
        if max_size is not None:
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                return False
                
        if allowed_formats is not None:
            if img.format not in allowed_formats:
                return False
                
        return True
    except Exception:
        return False 


def enhance_image(
    image: Union[str, Image.Image, np.ndarray],
    brightness: float = 1.0,
    contrast: float = 1.0,
    sharpness: float = 1.0,
    color: float = 1.0
) -> Image.Image:
    """Enhance image properties.
    
    Parameters
    ----------
    image : Union[str, Image.Image, np.ndarray]
        Input image
    brightness : float, optional
        Brightness factor, by default 1.0
    contrast : float, optional
        Contrast factor, by default 1.0
    sharpness : float, optional
        Sharpness factor, by default 1.0
    color : float, optional
        Color factor, by default 1.0
        
    Returns
    -------
    Image.Image
        Enhanced image
    """
    img = load_image(image)
    
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    if color != 1.0:
        img = ImageEnhance.Color(img).enhance(color)
        
    return img


def apply_filters(
    image: Union[str, Image.Image, np.ndarray],
    blur: bool = False,
    edge_enhance: bool = False,
    sharpen: bool = False,
    smooth: bool = False
) -> Image.Image:
    """Apply various filters to image.
    
    Parameters
    ----------
    image : Union[str, Image.Image, np.ndarray]
        Input image
    blur : bool, optional
        Apply Gaussian blur, by default False
    edge_enhance : bool, optional
        Enhance edges, by default False
    sharpen : bool, optional
        Sharpen image, by default False
    smooth : bool, optional
        Smooth image, by default False
        
    Returns
    -------
    Image.Image
        Filtered image
    """
    img = load_image(image)
    
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(2))
    if edge_enhance:
        img = img.filter(ImageFilter.EDGE_ENHANCE)
    if sharpen:
        img = img.filter(ImageFilter.SHARPEN)
    if smooth:
        img = img.filter(ImageFilter.SMOOTH)
        
    return img


def extract_dominant_colors(
    image: Union[str, Image.Image, np.ndarray],
    n_colors: int = 5,
    resize_to: Optional[Tuple[int, int]] = (150, 150)
) -> List[Tuple[Tuple[int, int, int], float]]:
    """Extract dominant colors from image.
    
    Parameters
    ----------
    image : Union[str, Image.Image, np.ndarray]
        Input image
    n_colors : int, optional
        Number of colors to extract, by default 5
    resize_to : Optional[Tuple[int, int]], optional
        Size to resize image to before processing, by default (150, 150)
        
    Returns
    -------
    List[Tuple[Tuple[int, int, int], float]]
        List of (RGB color, percentage) tuples
    """
    img = load_image(image)
    
    if resize_to:
        img = resize_image(img, resize_to, keep_aspect=False)
        
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    pixels = np.float32(img).reshape(-1, 3)
    
    # Use k-means to find dominant colors
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get colors and their percentages
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Count occurrences of each label
    unique_labels, counts = np.unique(labels, return_counts=True)
    percentages = counts / len(labels)
    
    # Convert to RGB tuples and percentages
    result = [
        ((int(color[0]), int(color[1]), int(color[2])), float(percentage))
        for color, percentage in zip(colors, percentages)
    ]
    
    # Sort by percentage
    return sorted(result, key=lambda x: x[1], reverse=True)


def get_image_stats(
    image: Union[str, Image.Image, np.ndarray]
) -> Dict[str, Union[Tuple[int, int], str, float]]:
    """Get basic image statistics.
    
    Parameters
    ----------
    image : Union[str, Image.Image, np.ndarray]
        Input image
        
    Returns
    -------
    Dict[str, Union[Tuple[int, int], str, float]]
        Dictionary containing image statistics
    """
    img = load_image(image)
    
    # Convert to RGB for consistent stats
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    
    stats = {
        'size': img.size,
        'mode': img.mode,
        'format': img.format,
        'mean_rgb': tuple(img_array.mean(axis=(0, 1)).astype(int)),
        'std_rgb': tuple(img_array.std(axis=(0, 1)).astype(int)),
        'min_rgb': tuple(img_array.min(axis=(0, 1))),
        'max_rgb': tuple(img_array.max(axis=(0, 1))),
        'aspect_ratio': img.size[0] / img.size[1],
    }
    
    return stats


def create_image_grid(
    images: List[Union[str, Image.Image, np.ndarray]],
    grid_size: Optional[Tuple[int, int]] = None,
    padding: int = 10,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """Create a grid of images.
    
    Parameters
    ----------
    images : List[Union[str, Image.Image, np.ndarray]]
        List of images
    grid_size : Optional[Tuple[int, int]], optional
        Grid dimensions (rows, cols), by default None
    padding : int, optional
        Padding between images, by default 10
    background_color : Tuple[int, int, int], optional
        Background color, by default (255, 255, 255)
        
    Returns
    -------
    Image.Image
        Grid of images
    """
    if not images:
        raise ValueError("No images provided")
        
    # Load all images
    imgs = [load_image(img) for img in images]
    
    # Convert all images to RGB
    imgs = [img.convert('RGB') if img.mode != 'RGB' else img for img in imgs]
    
    # Determine grid size if not provided
    if grid_size is None:
        n_images = len(imgs)
        n_cols = int(np.ceil(np.sqrt(n_images)))
        n_rows = int(np.ceil(n_images / n_cols))
        grid_size = (n_rows, n_cols)
    
    # Find maximum dimensions
    max_width = max(img.size[0] for img in imgs)
    max_height = max(img.size[1] for img in imgs)
    
    # Create grid
    grid_width = grid_size[1] * max_width + (grid_size[1] + 1) * padding
    grid_height = grid_size[0] * max_height + (grid_size[0] + 1) * padding
    
    grid = Image.new('RGB', (grid_width, grid_height), background_color)
    
    # Place images in grid
    for idx, img in enumerate(imgs):
        if idx >= grid_size[0] * grid_size[1]:
            break
            
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        
        x = col * (max_width + padding) + padding
        y = row * (max_height + padding) + padding
        
        # Center image in its cell
        x_offset = (max_width - img.size[0]) // 2
        y_offset = (max_height - img.size[1]) // 2
        
        grid.paste(img, (x + x_offset, y + y_offset))
    
    return grid 


def augment_image(
    image: Union[str, Image.Image, np.ndarray],
    rotate: Optional[float] = None,
    flip: Optional[str] = None,  # 'horizontal', 'vertical', or 'both'
    crop: Optional[Tuple[int, int, int, int]] = None,  # (left, top, right, bottom)
    scale: Optional[float] = None,
    translate: Optional[Tuple[int, int]] = None,  # (x, y)
) -> Image.Image:
    """Apply various augmentations to an image.
    
    Parameters
    ----------
    image : Union[str, Image.Image, np.ndarray]
        Input image
    rotate : Optional[float], optional
        Rotation angle in degrees, by default None
    flip : Optional[str], optional
        Flip direction, by default None
    crop : Optional[Tuple[int, int, int, int]], optional
        Crop coordinates, by default None
    scale : Optional[float], optional
        Scale factor, by default None
    translate : Optional[Tuple[int, int]], optional
        Translation offset, by default None
        
    Returns
    -------
    Image.Image
        Augmented image
    """
    img = load_image(image)
    
    if rotate is not None:
        img = img.rotate(rotate, expand=True)
    
    if flip:
        if flip == 'horizontal':
            img = ImageOps.mirror(img)
        elif flip == 'vertical':
            img = ImageOps.flip(img)
        elif flip == 'both':
            img = ImageOps.mirror(ImageOps.flip(img))
    
    if crop:
        img = img.crop(crop)
    
    if scale:
        new_size = tuple(int(dim * scale) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    if translate:
        img = ImageOps.expand(img, border=(translate[0], translate[1], 0, 0), fill=0)
        img = img.crop((0, 0, img.size[0] - translate[0], img.size[1] - translate[1]))
    
    return img


def blend_images(
    image1: Union[str, Image.Image, np.ndarray],
    image2: Union[str, Image.Image, np.ndarray],
    alpha: float = 0.5,
    blend_mode: str = "normal"  # "normal", "multiply", "screen", "overlay"
) -> Image.Image:
    """Blend two images together.
    
    Parameters
    ----------
    image1 : Union[str, Image.Image, np.ndarray]
        First image
    image2 : Union[str, Image.Image, np.ndarray]
        Second image
    alpha : float, optional
        Blend factor, by default 0.5
    blend_mode : str, optional
        Blending mode, by default "normal"
        
    Returns
    -------
    Image.Image
        Blended image
    """
    img1 = load_image(image1)
    img2 = load_image(image2)
    
    # Ensure both images are the same size
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    
    # Convert to RGBA
    img1 = img1.convert('RGBA')
    img2 = img2.convert('RGBA')
    
    if blend_mode == "normal":
        return Image.blend(img1, img2, alpha)
    elif blend_mode == "multiply":
        return ImageChops.multiply(img1, img2)
    elif blend_mode == "screen":
        return ImageChops.screen(img1, img2)
    elif blend_mode == "overlay":
        return ImageChops.overlay(img1, img2)
    else:
        raise ValueError(f"Unsupported blend mode: {blend_mode}")


def apply_artistic_filters(
    image: Union[str, Image.Image, np.ndarray],
    style: str = "sketch",  # "sketch", "watercolor", "oil_painting"
    intensity: float = 1.0
) -> Image.Image:
    """Apply artistic filters to image.
    
    Parameters
    ----------
    image : Union[str, Image.Image, np.ndarray]
        Input image
    style : str, optional
        Artistic style to apply, by default "sketch"
    intensity : float, optional
        Effect intensity, by default 1.0
        
    Returns
    -------
    Image.Image
        Stylized image
    """
    img = load_image(image)
    
    if style == "sketch":
        # Convert to grayscale and invert
        gray = ImageOps.grayscale(img)
        inverted = ImageOps.invert(gray)
        # Apply Gaussian blur
        blurred = inverted.filter(ImageFilter.GaussianBlur(radius=intensity * 3))
        # Blend with original
        return ImageChops.dodge(gray, blurred)
    
    elif style == "watercolor":
        # Reduce color palette
        img = img.quantize(colors=32).convert('RGB')
        # Apply median filter for smooth transitions
        img = img.filter(ImageFilter.MedianFilter(size=int(intensity * 3)))
        # Enhance color and contrast
        img = ImageEnhance.Color(img).enhance(1.5)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        return img
    
    elif style == "oil_painting":
        # Apply edge enhancement
        img = img.filter(ImageFilter.EDGE_ENHANCE)
        # Reduce detail with median filter
        img = img.filter(ImageFilter.MedianFilter(size=int(intensity * 5)))
        # Enhance color and contrast
        img = ImageEnhance.Color(img).enhance(1.3)
        img = ImageEnhance.Contrast(img).enhance(1.3)
        return img
    
    else:
        raise ValueError(f"Unsupported style: {style}")


def create_image_collage(
    images: List[Union[str, Image.Image, np.ndarray]],
    layout: str = "grid",  # "grid", "horizontal", "vertical", "random"
    spacing: int = 10,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    random_rotation: bool = False
) -> Image.Image:
    """Create a collage from multiple images.
    
    Parameters
    ----------
    images : List[Union[str, Image.Image, np.ndarray]]
        List of images
    layout : str, optional
        Collage layout type, by default "grid"
    spacing : int, optional
        Space between images, by default 10
    background_color : Tuple[int, int, int], optional
        Background color, by default (255, 255, 255)
    random_rotation : bool, optional
        Whether to randomly rotate images, by default False
        
    Returns
    -------
    Image.Image
        Collage image
    """
    if not images:
        raise ValueError("No images provided")
    
    # Load and convert all images
    imgs = [load_image(img).convert('RGBA') for img in images]
    
    if layout == "grid":
        return create_image_grid(imgs, padding=spacing, background_color=background_color)
    
    elif layout in ["horizontal", "vertical"]:
        # Calculate total size
        if layout == "horizontal":
            total_width = sum(img.size[0] for img in imgs) + spacing * (len(imgs) - 1)
            max_height = max(img.size[1] for img in imgs)
            size = (total_width, max_height)
        else:  # vertical
            max_width = max(img.size[0] for img in imgs)
            total_height = sum(img.size[1] for img in imgs) + spacing * (len(imgs) - 1)
            size = (max_width, total_height)
        
        # Create background
        collage = Image.new('RGBA', size, background_color)
        
        # Place images
        x, y = 0, 0
        for img in imgs:
            if random_rotation:
                img = img.rotate(np.random.uniform(-30, 30), expand=True)
            
            if layout == "horizontal":
                y = (size[1] - img.size[1]) // 2
                collage.paste(img, (x, y), img)
                x += img.size[0] + spacing
            else:  # vertical
                x = (size[0] - img.size[0]) // 2
                collage.paste(img, (x, y), img)
                y += img.size[1] + spacing
        
        return collage
    
    elif layout == "random":
        # Calculate size to fit all images
        max_width = sum(img.size[0] for img in imgs) // 2
        max_height = sum(img.size[1] for img in imgs) // 2
        size = (max_width, max_height)
        
        collage = Image.new('RGBA', size, background_color)
        
        for img in imgs:
            if random_rotation:
                img = img.rotate(np.random.uniform(-30, 30), expand=True)
            
            # Random position
            x = np.random.randint(0, max(1, size[0] - img.size[0]))
            y = np.random.randint(0, max(1, size[1] - img.size[1]))
            
            collage.paste(img, (x, y), img)
        
        return collage
    
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def analyze_image_composition(
    image: Union[str, Image.Image, np.ndarray]
) -> Dict[str, Any]:
    """Analyze image composition and visual elements.
    
    Parameters
    ----------
    image : Union[str, Image.Image, np.ndarray]
        Input image
        
    Returns
    -------
    Dict[str, Any]
        Analysis results including:
        - Rule of thirds points
        - Symmetry score
        - Color harmony
        - Edge density
        - Focal points
    """
    img = load_image(image)
    img_array = np.array(img)
    
    # Convert to grayscale for edge detection
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # Calculate edge density using Sobel
    from scipy import ndimage
    sobel_h = ndimage.sobel(gray, axis=0)
    sobel_v = ndimage.sobel(gray, axis=1)
    edge_density = np.sqrt(sobel_h**2 + sobel_v**2)
    
    # Calculate rule of thirds points
    h, w = gray.shape
    thirds_h = [h // 3, 2 * h // 3]
    thirds_w = [w // 3, 2 * w // 3]
    
    # Calculate symmetry score
    symmetry_v = np.mean(np.abs(gray - np.fliplr(gray)))
    symmetry_h = np.mean(np.abs(gray - np.flipud(gray)))
    
    # Get color harmony
    colors = extract_dominant_colors(img, n_colors=5)
    
    # Find focal points (regions with high edge density)
    focal_points = []
    for i in range(3):
        for j in range(3):
            region = edge_density[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
            if np.mean(region) > np.mean(edge_density):
                focal_points.append((i, j))
    
    return {
        'rule_of_thirds_points': {
            'horizontal': thirds_h,
            'vertical': thirds_w
        },
        'symmetry_score': {
            'vertical': float(symmetry_v),
            'horizontal': float(symmetry_h)
        },
        'color_harmony': colors,
        'edge_density': {
            'mean': float(np.mean(edge_density)),
            'std': float(np.std(edge_density))
        },
        'focal_points': focal_points
    } 