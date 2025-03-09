import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_pipes_keep_symbols(image_path, output_path=None, 
                             # Adjustable parameters with default values
                             # Thresholding parameters
                             adaptive_block_size=11,
                             adaptive_C=2,
                             preprocessing_blur=0,  # 0 for no blur, or odd numbers like 3, 5, etc.
                             
                             # Symbol detection parameters
                             symbol_min_area=50,
                             symbol_max_area=8000,
                             symbol_min_aspect=0.4,
                             symbol_max_aspect=2.5,
                             symbol_min_circularity=0.3,
                             symbol_solidity_threshold=0.6,
                             
                             # Text detection parameters
                             text_min_area=10,
                             text_max_area=500,
                             text_min_aspect=0.2,
                             text_max_circularity=0.3,
                             
                             # Morphological parameters
                             symbol_dilation_size=5,
                             text_dilation_size=3,
                             cleanup_kernel_size=3,
                             
                             # Pipe detection control
                             use_inverse_method=False,    # Set to False to use direct pipe detection
                             line_detection_threshold=40, # For direct detection
                             min_line_length=30,          # For direct detection
                             max_line_gap=15             # For direct detection
                             ):
    """
    Process a P&ID diagram to remove pipes while preserving symbols.
    
    Args:
        image_path (str): Path to the input P&ID diagram
        output_path (str, optional): Path to save the output image
        
    Returns:
        numpy.ndarray: Processed image with pipes removed
    """
    # Step 1: Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Keep a copy of the original for final composition
    original = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Optional blur for noise reduction
    if preprocessing_blur > 0:
        gray = cv2.GaussianBlur(gray, (preprocessing_blur, preprocessing_blur), 0)
    
    # Apply adaptive thresholding 
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_C)
    
    # Step 2: Identify symbols using connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create masks for symbols and text
    symbol_mask = np.zeros_like(gray)
    text_mask = np.zeros_like(gray)
    
    # Process each connected component
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        component = (labels == i).astype(np.uint8) * 255
        
        # Calculate additional properties
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = contours[0]
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull) if len(hull) > 2 else 0
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Circularity: 4 * pi * area / perimeter^2
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Aspect ratio
            aspect_ratio = float(width) / float(height) if height > 0 else 0
            
            # Symbol criteria with adjustable parameters
            if ((symbol_min_area < area < symbol_max_area and 
                 (symbol_min_aspect < aspect_ratio < symbol_max_aspect or 
                  circularity > symbol_min_circularity)) or 
                (area > 500 and solidity > symbol_solidity_threshold) or
                (50 < area < 800 and circularity > 0.5)):
                symbol_mask = cv2.bitwise_or(symbol_mask, component)
            
            # Text criteria with adjustable parameters
            elif (text_min_area < area < text_max_area and width > 0 and height > 0 and 
                  (aspect_ratio > text_min_aspect or circularity < text_max_circularity)):
                text_mask = cv2.bitwise_or(text_mask, component)
    
    # Dilate masks to ensure complete coverage
    symbol_kernel = np.ones((symbol_dilation_size, symbol_dilation_size), np.uint8)
    text_kernel = np.ones((text_dilation_size, text_dilation_size), np.uint8)
    
    symbol_mask_dilated = cv2.dilate(symbol_mask, symbol_kernel, iterations=1)
    text_mask_dilated = cv2.dilate(text_mask, text_kernel, iterations=1)
    
    # Step 3: Create the pipe mask
    if use_inverse_method:
        # Inverse method: assume everything not a symbol or text is a pipe
        protection_mask = cv2.bitwise_or(symbol_mask_dilated, text_mask_dilated)
        pipe_mask = binary.copy()
        pipe_mask[protection_mask > 0] = 0
        
        # Clean up the pipe mask
        cleanup_kernel = np.ones((cleanup_kernel_size, cleanup_kernel_size), np.uint8)
        pipe_mask = cv2.morphologyEx(pipe_mask, cv2.MORPH_OPEN, cleanup_kernel)
    else:
        # Direct pipe detection method
        binary_pipes_only = binary.copy()
        
        # Remove the identified symbols from consideration
        protection_mask = cv2.bitwise_or(symbol_mask_dilated, text_mask_dilated)
        binary_pipes_only[protection_mask > 0] = 0
        
        # Line detection using Hough Transform
        lines = cv2.HoughLinesP(
            binary_pipes_only, 
            rho=1, 
            theta=np.pi/180, 
            threshold=line_detection_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        # Create a mask for detected lines
        pipe_mask = np.zeros_like(gray)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(pipe_mask, (x1, y1), (x2, y2), 255, 2)
    
    # Step 4: Create the final result
    result = original.copy()
    
    # Set pipe areas to white
    result[pipe_mask > 0] = [255, 255, 255]
    
    # Save the result if an output path is provided
    if output_path:
        cv2.imwrite(output_path, result)
    
    # Create a visualization to show the process
    visualization = create_visualization(original, binary, pipe_mask, 
                                         symbol_mask_dilated, text_mask_dilated, result)
    
    return result, visualization, {
        'binary': binary,
        'pipe_mask': pipe_mask,
        'symbol_mask': symbol_mask_dilated,
        'text_mask': text_mask_dilated
    }


def create_visualization(original, binary, pipe_mask, symbol_mask, text_mask, result):
    """
    Create a visualization of the different stages of processing.
    """
    rows, cols = 2, 3
    plt.figure(figsize=(15, 10))
    
    plt.subplot(rows, cols, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(rows, cols, 2)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')
    
    plt.subplot(rows, cols, 3)
    plt.imshow(pipe_mask, cmap='gray')
    plt.title('Detected Pipes')
    plt.axis('off')
    
    plt.subplot(rows, cols, 4)
    plt.imshow(symbol_mask, cmap='gray')
    plt.title('Detected Symbols')
    plt.axis('off')
    
    plt.subplot(rows, cols, 5)
    plt.imshow(text_mask, cmap='gray')
    plt.title('Detected Text')
    plt.axis('off')
    
    plt.subplot(rows, cols, 6)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Final Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('pipe_removal_visualization.png')
    plt.close()
    
    return plt.gcf()


if __name__ == "__main__":
    input_path = "input.jpg"
    output_path = "pid_diagram_no_pipes.jpg"
    result, visualization, masks = remove_pipes_keep_symbols(
        input_path,
        output_path,

        adaptive_block_size=15, 
        adaptive_C=3,            
        preprocessing_blur=5,
        
        symbol_min_area=30,      
        symbol_max_area=10000,   
        symbol_min_aspect=0.3,   
        symbol_max_aspect=3.0,   
        
        text_min_area=5,         
        text_max_area=800,       
        
        symbol_dilation_size=10,
        text_dilation_size=5,
        
        # For cleaner results:
        cleanup_kernel_size=2
    )
    
    print(f"Processed image saved to {output_path}")
    print("Visualization saved to pipe_removal_visualization.png")
