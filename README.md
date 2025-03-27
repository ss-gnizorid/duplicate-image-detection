# Image Duplicate Detection Toolkit

This project is a lightweight toolkit for detecting duplicate or near-duplicate images using perceptual image hashing and various image transformations. The toolkit is designed to help you process images by generating multiple variations (or "permutations") of each image, computing hashes for each variation, and comparing these hashes against a database to identify duplicates.

## Overview

The toolkit is built around two main classes:

- **ImagePermutation**  
Generates image transformations, computes perceptual hashes, and converts images to byte representations. It supports two modes:
- **Basic**: Generates a few standard permutations (original, horizontal flip, vertical flip).
- **Advanced**: In addition to the basic transformations, it performs rotations, zooming, simulated JPEG compression, and brightness/contrast adjustments to generate more variations.

- **ImageScanner**  
Compares a database of image hashes (stored in a Pandas DataFrame) against test images, using Hamming distance to determine exact or fuzzy matches. It also provides a utility to convert stored image bytes back into a PIL image.

## Features

- **Image Transformation and Permutation:**  
Create multiple variations of each image (flips, rotations, zoom, brightness/contrast adjustments, and compression) to robustly capture image similarities.

- **Perceptual Hashing:**  
Uses the `imagehash` library to generate average hashes for each transformed image, facilitating effective comparisons.

- **Byte Conversion:**  
Converts images into a byte format for storage and later retrieval.

- **Batch Processing:**  
Easily process entire directories of images and store the results in a Pandas DataFrame.

- **Fuzzy Matching:**  
Compare images based on a configurable Hamming distance threshold to allow for near-duplicate detection.

## Installation

To use this toolkit, install the following dependencies:

```bash
pip install pillow pandas numpy imagehash
```

## Usage

### Processing Images with ImagePermutation

The `ImagePermutation` class handles image loading, transformation, hash generation, and byte conversion.

**Example:**

```python
from ImagePermutation import ImagePermutation

# Initialize with desired resize dimensions and mode ('basic' or 'advanced')
ip = ImagePermutation(resize=(256, 256), mode='advanced')

# Process a single image to generate hashes and bytes
image_data = ip.generate_hashes_and_bytes("path/to/your/image.jpg")

# Process all images in a directory
image_db = ip.process_image_directory("path/to/image/directory")
```

### Comparing Images with ImageScanner

The `ImageScanner` class takes a DataFrame of image data (from ImagePermutation) and compares test images against the stored database.

**Example:**

```python
import pandas as pd
from ImagePermutation import ImagePermutation, ImageScanner

# Build the image database from a directory of images
ip = ImagePermutation(mode='basic')
image_db = ip.process_image_directory("path/to/image/database")
image_db_df = pd.DataFrame(image_db)

# Initialize the ImageScanner with the image database
isc = ImageScanner(_database=image_db_df)

# Process test images to generate hash data
test_hashes = ip.process_image_directory("path/to/test/images")
test_df = isc.batch_test_hashes_to_df(test_hashes)

# Compare hashes between test images and the database (using a threshold, e.g., 5)
matches_df = isc.compare_hashes_fuzzy_crosswise_df(test_df, threshold=5)

# Print the results of matching images
isc.print_matches_df()
```

### Viewing Stored Images

Since the toolkit stores images as byte data, you can convert these bytes back into an image using the static method `bytes_to_image()`. In a cloud-based or headless environment (like SageMaker or JupyterLab), you might need to save the image to a file to view it.

**Example:**

```python
# Convert image bytes back to an image
# Adjust 'original_bytes' to the actual column name in your DataFrame
image_bytes = image_db_df.iloc[0]['original_bytes']
restored_image = ImageScanner.bytes_to_image(image_bytes)

# Save the image to a file for viewing
restored_image.save("/home/sagemaker-user/restored_image_test.jpg")
```

## Code Structure

### ImagePermutation Class

- **`image_to_bytes(image)`**  
Converts a PIL image to bytes (PNG format).

- **`generate_permutations(image_path)`**  
Generates a dictionary of image permutations (flips, rotations, zoom, brightness/contrast, simulated compression).

- **`generate_hashes_and_bytes(image_path)`**  
Processes an image to compute hashes and convert each permutation to byte format.

- **`process_image_directory(folder_path)`**  
Processes all image files in a directory and returns a list of dictionaries containing the image data.

### ImageScanner Class

- **`bytes_to_image(image_bytes)`**  
*Static Method:* Converts stored image bytes back into a PIL image.

- **`batch_test_hashes_to_df(batch_test_hashes)`**  
Converts a list of hash dictionaries into a Pandas DataFrame.

- **`compare_hashes_fuzzy_crosswise_df(test_df, threshold)`**  
Compares test image hashes against a database of hashes using Hamming distance, returning matches that are exact or within a fuzzy threshold.

- **`print_matches_df()`**  
Prints the matching results in a readable format.

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request with your improvements.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

- [Pillow](https://python-pillow.org/) for image processing.
- [pandas](https://pandas.pydata.org/) for data management.
- [imagehash](https://github.com/JohannesBuchner/imagehash) for generating perceptual image hashes.
