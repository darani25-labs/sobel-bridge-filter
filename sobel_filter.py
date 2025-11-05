import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sobel_filter(image_path, output_path='bridge_sobel_output.jpg'):
    """
    Apply Sobel filter to highlight edges in a bridge image.
    
    Parameters:
    - image_path: Path to the input bridge image
    - output_path: Path to save the filtered output image
    """
    
    # Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel filter in X direction
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    
    # Apply Sobel filter in Y direction
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the magnitude of gradients
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to 0-255 range
    sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
    
    # Convert absolute values for visualization
    sobel_x_abs = np.uint8(np.absolute(sobel_x))
    sobel_y_abs = np.uint8(np.absolute(sobel_y))
    
    # Save the filtered image
    cv2.imwrite(output_path, sobel_combined)
    print(f"Filtered image saved to: {output_path}")
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Bridge Image')
    plt.axis('off')
    
    # Grayscale image
    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    
    # Sobel X
    plt.subplot(2, 3, 3)
    plt.imshow(sobel_x_abs, cmap='gray')
    plt.title('Sobel X (Vertical Edges)')
    plt.axis('off')
    
    # Sobel Y
    plt.subplot(2, 3, 4)
    plt.imshow(sobel_y_abs, cmap='gray')
    plt.title('Sobel Y (Horizontal Edges)')
    plt.axis('off')
    
    # Combined Sobel
    plt.subplot(2, 3, 5)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Combined Sobel Filter')
    plt.axis('off')
    
    # Edge overlay on original
    plt.subplot(2, 3, 6)
    overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    edges_colored = cv2.applyColorMap(sobel_combined, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(overlay, 0.7, cv2.cvtColor(edges_colored, cv2.COLOR_BGR2RGB), 0.3, 0)
    plt.imshow(overlay)
    plt.title('Edge Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return sobel_combined


# Example usage
if __name__ == "__main__":
    # Replace with your bridge image path
    image_path = "images/bridge.jpg"  # Change this to your image file
    
    # Apply Sobel filter
    filtered_image = apply_sobel_filter(image_path, output_path='output/bridge_edges.jpg')
    
    print("\nSobel filter applied successfully!")
    print("The filtered image highlights structural edges and features of the bridge.")