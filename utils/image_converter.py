from PIL import Image
import numpy as np
import os

def resize_to_28x28_pil(img_path, output_path):
    """Resize image to 28x28 using PIL/Pillow."""
    # Open image
    img = Image.open(img_path)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Resize to 26x26 (or maintain aspect ratio)
    img = img.resize((26, 26), Image.Resampling.LANCZOS)
    
    # Create 28x28 canvas
    new_img = Image.new('L', (28, 28), color=0)  # Black background
    # Calculate position to center the image
    paste_x = (28 - 26) // 2
    paste_y = (28 - 26) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    # Save
    new_img.save(output_path, format='BMP')
    return np.array(new_img)

def process_images_pil():
    """Process images from 'images' folder to 'images_processed' folder."""
    current_directory = os.getcwd();
    input_folder =  current_directory + "/images"
    output_folder =  current_directory + "/images_processed"
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: '{input_folder}' folder not found in current directory!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Please make sure you have an '{input_folder}' folder with images.")
        return
    
    supported_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    
    processed = 0
    number = 1
    rank = 0
    for filename in os.listdir(input_folder):
        new_filename = str(number) + "-" + str(rank)  + ".bmp"
        if any(filename.lower().endswith(ext) for ext in supported_ext):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, new_filename)
            
            try:
                resize_to_28x28_pil(input_path, output_path)
                print(f"✓ {new_filename}")
                processed += 1
                rank += 1
                if rank == 20:
                    rank = 0 
                    number += 1

                if number == 10 : 
                    number = 0
            except Exception as e:
                print(f"✗ {new_filename}: {e}")
    
    if processed > 0:
        print(f"\nSuccessfully processed {processed} images!")
        print(f"Check the '{output_folder}' folder.")
    else:
        print(f"\nNo images processed. Make sure you have images in '{input_folder}' folder.")

if __name__ == "__main__":
    process_images_pil()