import os
from PIL import Image
import numpy as np


import glob
image_files = glob.glob('data/isotropic_true/*.png')
# Iterate over each image file
for file_name in image_files:
    # Construct the full file path
    file_path = os.path.join(file_name)

    # Open the image
    image = Image.open(file_path)

    # Invert the image
    inverted_image = Image.eval(image, lambda x: 255 - x)

    # Save the inverted image
    inverted_image.save(file_path)

    # Close the image
    image.close()

print("Image inversion complete!")