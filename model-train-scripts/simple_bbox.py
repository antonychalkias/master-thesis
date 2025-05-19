#!/usr/bin/env python3
"""
Simple script to create bounding boxes on food images.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def add_bounding_box(image_path, output_path):
    """Add a bounding box to an image."""
    # Load image
    try:
        # Open the image
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        
        # Create figure and axis
        fig, ax = plt.subplots(1)
        
        # Display the image
        ax.imshow(img_array)
        
        # Create a simple centered bounding box covering 70% of the image
        h, w = img_array.shape[:2]
        box_w, box_h = int(w * 0.7), int(h * 0.7)
        x1, y1 = (w - box_w) // 2, (h - box_h) // 2
        x2, y2 = x1 + box_w, y1 + box_h
        
        # Add rectangle patch to the image
        rect = patches.Rectangle(
            (x1, y1), box_w, box_h, 
            linewidth=3, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add a simple label
        label_text = "Sample Food"
        ax.text(
            x1, y1-10, label_text,
            color='white', fontsize=12, weight='bold',
            bbox=dict(facecolor='green', alpha=0.7)
        )
        
        # Turn off axis
        ax.axis('off')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        
        print(f"Image with bounding box saved to {output_path}")
        
        # Also create a version using PIL for better quality
        img_with_box = image.copy()
        draw = ImageDraw.Draw(img_with_box)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("Arial", 24)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw text background
        draw.rectangle([x1, y1-30, x1+200, y1], fill=(0, 128, 0))
        
        # Draw text
        draw.text((x1+5, y1-28), "Sample Food", fill=(255, 255, 255), font=font)
        
        # Save the enhanced image
        pil_output = output_path.replace('.jpg', '_pil.jpg')
        img_with_box.save(pil_output)
        print(f"PIL version saved to {pil_output}")
        
        return True
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: simple_bbox.py <input_image> <output_image>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)
    
    success = add_bounding_box(input_path, output_path)
    
    if success:
        print("Bounding box added successfully!")
    else:
        print("Failed to add bounding box.")
        sys.exit(1)

if __name__ == "__main__":
    main()
