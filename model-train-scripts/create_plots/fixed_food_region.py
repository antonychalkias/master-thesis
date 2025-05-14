def find_food_region(image):
    """
    Attempt to find the food region in the image using more advanced image processing.
    Returns bounding box coordinates (x1, y1, x2, y2) for the food.
    """
    # Convert PIL Image to numpy array for processing
    img_array = np.array(image)
    width, height = image.size
    
    # Default fallback values - will be used if detection fails
    default_x_min = (width - int(width * 0.8)) // 2
    default_y_min = (height - int(height * 0.8)) // 2
    default_x_max = default_x_min + int(width * 0.8)
    default_y_max = default_y_min + int(height * 0.8)
    
    # Check if required libraries are available
    if not HAS_SKIMAGE:
        print("Advanced food detection unavailable: scikit-image not installed")
        return default_x_min, default_y_min, default_x_max, default_y_max
    
    try:
        # Convert to grayscale if image is color
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # Convert to LAB color space which is better for color segmentation
            lab_img = color.rgb2lab(img_array[:,:,:3] / 255.0)
            
            # Apply SLIC superpixel segmentation for better object detection
            # This groups similar color regions together
            segments = segmentation.slic(lab_img, n_segments=100, compactness=10, start_label=1)
            
            # Find the segment closest to the center (likely to be the food)
            center_y, center_x = int(height / 2), int(width / 2)
            center_segment = segments[center_y, center_x]
            
            # Create a mask of the center segment and surrounding similar segments
            mask = np.zeros_like(segments, dtype=bool)
            mask[segments == center_segment] = True
            
            # Analyze the surrounding segments and include them if they're similar
            props = measure.regionprops(segments)
            
            # Expand to include similar nearby segments
            for prop in props:
                prop_center = prop.centroid
                distance = np.sqrt((prop_center[0] - center_y)**2 + (prop_center[1] - center_x)**2)
                if distance < min(height, width) * 0.4:  # Include segments close to the center
                    mask[segments == prop.label] = True
            
            # Use morphological operations to clean up the mask
            mask = ndimage.binary_dilation(mask, iterations=3)
            mask = ndimage.binary_erosion(mask, iterations=2)
            mask = ndimage.binary_dilation(mask, iterations=2)
            
            # Find contours of the mask
            contours = measure.find_contours(mask.astype(float), 0.5)
            
            if contours and len(contours) > 0:
                # Find the largest contour by area
                largest_contour = max(contours, key=lambda x: len(x))
                
                # Get bounding box from contour
                y_indices, x_indices = largest_contour[:, 0], largest_contour[:, 1]
                
                # Check if we have enough points to form a meaningful bounding box
                if len(x_indices) < 10 or len(y_indices) < 10:
                    raise ValueError("Contour too small, falling back to edge detection")
            else:
                # Fallback to edge detection if no contours found
                raise ValueError("No contours found, falling back to edge detection")
        
        # Fallback to edge detection
        else:
            raise ValueError("Image format requires edge detection fallback")
            
    except Exception as e:
        print(f"Color-based segmentation failed: {e}. Trying edge detection...")
        try:
            # Use edge detection to find the food region
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray_img = np.dot(img_array[:, :, :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            else:
                gray_img = img_array
            
            # Apply Gaussian blur to reduce noise
            blurred = filters.gaussian(gray_img, sigma=1.0)
            
            # Use Canny edge detection
            edges = feature.canny(blurred, sigma=2)
            
            # Find the coordinates of the edges
            y_indices, x_indices = np.where(edges)
            
            # Check if we found any edges
            if len(x_indices) < 10 or len(y_indices) < 10:
                raise ValueError("Insufficient edges detected")
                
        except Exception as e2:
            print(f"Edge detection also failed: {e2}. Using default bounding box.")
            return default_x_min, default_y_min, default_x_max, default_y_max
    
    try:
        # Get bounding box coordinates
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Check if the bounding box is too small (less than 20% of the image)
        min_bbox_size = min(width, height) * 0.2
        if (x_max - x_min < min_bbox_size) or (y_max - y_min < min_bbox_size):
            print("Detected region too small, using default bounding box")
            return default_x_min, default_y_min, default_x_max, default_y_max
        
        # Add padding (20% of image dimensions)
        padding_x = int(width * 0.2)
        padding_y = int(height * 0.2)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(width, x_max + padding_x)
        y_max = min(height, y_max + padding_y)
        
        return x_min, y_min, x_max, y_max
        
    except Exception as e:
        print(f"Error processing detection results: {e}. Using default bounding box.")
        return default_x_min, default_y_min, default_x_max, default_y_max
