import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from scipy import ndimage

class PIDSymbolDetector:
    def __init__(self, model_path, conf_threshold=0.25, device=None):
        """
        Initialize the P&ID symbol detector
        
        Args:
            model_path: Path to the trained YOLOv11 model
            conf_threshold: Confidence threshold for detection
            device: Device to run inference on ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        print(f"Model loaded from: {model_path}")
    
    def detect_symbols(self, image_path):
        """
        Detect symbols in the P&ID diagram
        
        Args:
            image_path: Path to the input image
            
        Returns:
            detections: List of detections
            image: Original image
        """
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            device=0 if self.device.type == 'cuda' else 'cpu',
            verbose=False
        )
        
        # Load original image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        return results, image
    
    def visualize_detections(self, image, results, output_path=None):
        """
        Visualize the detected symbols
        
        Args:
            image: Original image
            results: Detection results
            output_path: Path to save the visualization
            
        Returns:
            vis_image: Visualization image
        """
        # Create a copy of the image
        vis_image = image.copy()
        
        # Draw bounding boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"Class: {cls}, {conf:.2f}"
                cv2.putText(vis_image, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization if output path is provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"Visualization saved to: {output_path}")
        
        return vis_image

class PIDPipeRemover:
    def __init__(self, line_thickness=2):
        """
        Initialize the P&ID pipe remover
        
        Args:
            line_thickness: Thickness of lines to remove
        """
        self.line_thickness = line_thickness
    
    def create_symbol_mask(self, image, results):
        """
        Create a binary mask for the detected symbols
        
        Args:
            image: Input image
            results: Detection results
            
        Returns:
            symbol_mask: Binary mask of detected symbols
        """
        h, w = image.shape[:2]
        symbol_mask = np.zeros((h, w), dtype=np.uint8)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Create a mask for the symbol
                cv2.rectangle(symbol_mask, (x1, y1), (x2, y2), 255, -1)
        
        return symbol_mask
    
    def detect_pipes(self, image, symbol_mask):
        """
        Detect pipes in the image
        
        Args:
            image: Input image
            symbol_mask: Binary mask of detected symbols
            
        Returns:
            pipes_mask: Binary mask of detected pipes
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Apply binary thresholding to highlight all lines
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Exclude symbol areas from edge detection
        edges = cv2.bitwise_and(edges, cv2.bitwise_not(symbol_mask))
        
        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30, 
            minLineLength=20, 
            maxLineGap=10
        )
        
        # Create pipes mask
        pipes_mask = np.zeros((h, w), dtype=np.uint8)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(pipes_mask, (x1, y1), (x2, y2), 255, self.line_thickness)
        
        # Connect broken lines with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        pipes_mask = cv2.dilate(pipes_mask, kernel, iterations=1)
        
        # Remove small objects (noise)
        pipes_mask = self.remove_small_objects(pipes_mask, min_size=50)
        
        # Ensure pipes are not in symbol areas
        pipes_mask = cv2.bitwise_and(pipes_mask, cv2.bitwise_not(symbol_mask))
        
        return pipes_mask
    
    def detect_curved_pipes(self, image, symbol_mask, straight_pipes_mask):
        """
        Detect curved pipes that might be missed by the Hough transform
        
        Args:
            image: Input image
            symbol_mask: Binary mask of detected symbols
            straight_pipes_mask: Binary mask of detected straight pipes
            
        Returns:
            curved_pipes_mask: Binary mask of detected curved pipes
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Remove symbol areas and straight pipes
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(symbol_mask))
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(straight_pipes_mask))
        
        # Apply morphological operations to connect curved segments
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove small objects (noise)
        curved_pipes_mask = self.remove_small_objects(binary, min_size=30)
        
        return curved_pipes_mask
    
    def remove_small_objects(self, binary_image, min_size=50):
        """
        Remove small connected components from binary image
        
        Args:
            binary_image: Binary image
            min_size: Minimum size of objects to keep
            
        Returns:
            cleaned_image: Cleaned binary image
        """
        # Label connected components
        labeled, num_features = ndimage.label(binary_image)
        
        # Measure the size of each component
        sizes = np.bincount(labeled.ravel())
        
        # Set background (label 0) size to 0
        if len(sizes) > 0:
            sizes[0] = 0
        
        # Create a mask of components to remove
        mask_sizes = sizes < min_size
        
        # Remove small components
        remove_pixels = mask_sizes[labeled]
        cleaned_image = binary_image.copy()
        cleaned_image[remove_pixels] = 0
        
        return cleaned_image
    
    def remove_pipes(self, image, symbol_mask):
        """
        Remove pipes from the image
        
        Args:
            image: Input image
            symbol_mask: Binary mask of detected symbols
            
        Returns:
            result_image: Image with pipes removed
            debug_info: Dictionary of debug information
        """
        # Detect straight pipes
        straight_pipes_mask = self.detect_pipes(image, symbol_mask)
        
        # Detect curved pipes
        curved_pipes_mask = self.detect_curved_pipes(image, symbol_mask, straight_pipes_mask)
        
        # Combine pipe masks
        pipes_mask = cv2.bitwise_or(straight_pipes_mask, curved_pipes_mask)
        
        # Create final result
        result_image = image.copy()
        
        # Determine background color (mode of border pixels)
        border_pixels = np.concatenate([
            image[0, :],    
            image[-1, :],   
            image[:, 0],    
            image[:, -1]  
        ])
        
        if len(image.shape) == 3: 
            bg_color = np.median(border_pixels, axis=0).astype(np.uint8)
        else: 
            bg_color = np.median(border_pixels).astype(np.uint8)
        
        # Replace pipe pixels with background color
        if len(image.shape) == 3:  # Color image
            result_image[pipes_mask > 0] = bg_color
        else:  # Grayscale image
            result_image[pipes_mask > 0] = bg_color
        
        # Prepare debug information
        debug_info = {
            'symbol_mask': symbol_mask,
            'straight_pipes_mask': straight_pipes_mask,
            'curved_pipes_mask': curved_pipes_mask,
            'combined_pipes_mask': pipes_mask
        }
        
        return result_image, debug_info
    
    def visualize_pipeline(self, image, debug_info, output_path=None):
        """
        Visualize the pipe removal pipeline
        
        Args:
            image: Original image
            debug_info: Debug information from remove_pipes
            output_path: Path to save visualization
            
        Returns:
            vis_image: Visualization image
        """
        # Create a visualization with 2x2 grid
        h, w = image.shape[:2]
        vis_image = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # Original image
        if len(image.shape) == 3:
            vis_image[:h, :w] = image
        else:
            vis_image[:h, :w] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Symbol mask
        symbol_mask_vis = cv2.cvtColor(debug_info['symbol_mask'], cv2.COLOR_GRAY2BGR)
        symbol_mask_vis[:, :, 0] = 0  # Set red channel to 0
        symbol_mask_vis[:, :, 2] = 0  # Set blue channel to 0
        vis_image[:h, w:] = symbol_mask_vis
        
        # Pipes mask
        pipes_mask_vis = cv2.cvtColor(debug_info['combined_pipes_mask'], cv2.COLOR_GRAY2BGR)
        pipes_mask_vis[:, :, 1] = 0  # Set green channel to 0
        pipes_mask_vis[:, :, 2] = 0  # Set blue channel to 0
        vis_image[h:, :w] = pipes_mask_vis
        
        # Result image (with pipes removed)
        if 'result_image' in debug_info:
            result_image = debug_info['result_image']
            if len(result_image.shape) == 3:
                vis_image[h:, w:] = result_image
            else:
                vis_image[h:, w:] = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        cv2.putText(vis_image, "Original Image", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, "Symbol Mask", (w+10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, "Pipes Mask", (10, h+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, "Result Image", (w+10, h+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization if output path is provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"Pipeline visualization saved to: {output_path}")
        
        return vis_image

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='P&ID Symbol Detection and Pipe Removal')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained YOLOv11 model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input image or directory')
    parser.add_argument('--output', type=str, default='results',
                        help='Path to save results')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detection')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on ("cuda" or "cpu")')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the detection and pipe removal process')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector and pipe remover
    detector = PIDSymbolDetector(args.model, args.conf, args.device)
    pipe_remover = PIDPipeRemover()
    
    # Process input (file or directory)
    input_path = args.input
    if os.path.isfile(input_path):
        # Process single file
        process_image(input_path, args.output, detector, pipe_remover, args.visualize)
    elif os.path.isdir(input_path):
        # Process directory
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_path = os.path.join(input_path, filename)
                process_image(image_path, args.output, detector, pipe_remover, args.visualize)
    else:
        print(f"Error: Input path {input_path} is not valid")
        sys.exit(1)
    
    print("Processing completed!")

def process_image(image_path, output_dir, detector, pipe_remover, visualize):
    """Process a single image"""
    print(f"Processing: {image_path}")
    
    # Get base filename
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Detect symbols
    results, image = detector.detect_symbols(image_path)
    
    # Visualize detections
    if visualize:
        vis_output_path = os.path.join(output_dir, f"{base_filename}_detections.jpg")
        detector.visualize_detections(image, results, vis_output_path)
    
    # Create symbol mask
    symbol_mask = pipe_remover.create_symbol_mask(image, results)
    
    # Remove pipes
    result_image, debug_info = pipe_remover.remove_pipes(image, symbol_mask)
    
    # Add result image to debug info (for visualization)
    debug_info['result_image'] = result_image
    
    # Visualize pipe removal pipeline
    if visualize:
        vis_output_path = os.path.join(output_dir, f"{base_filename}_pipeline.jpg")
        pipe_remover.visualize_pipeline(image, debug_info, vis_output_path)
    
    # Save result image
    result_output_path = os.path.join(output_dir, f"{base_filename}_result.jpg")
    cv2.imwrite(result_output_path, result_image)
    print(f"Result saved to: {result_output_path}")

if __name__ == "__main__":
    main()