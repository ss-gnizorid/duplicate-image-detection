# duplicate-image-detection
Sitetracker duplicate image detection

## ImagePermutation Module

The `ImagePermutation` module provides functionality to extract hash-maps and bytes from a base image and various transformed versions of that image. This helps in detecting if an image already exists in an image database. The module can perform these operations on individual or groups of images.

### Classes

#### ImagePermutation

The `ImagePermutation` class is responsible for generating image permutations and their corresponding hashes and bytes.

**Attributes:**
- `resize` (List[int]): The desired width and height for resizing.
- `mode` (Literal['basic', 'advanced']): The processing mode. Defaults to 'basic'.

**Methods:**
- `__init__(self, resize: Tuple[int] = (256,256), mode: Literal['basic', 'advanced'] = 'basic')`: Initializes an `ImagePermutation` instance.
- `mode`: Property getter and setter for the processing mode.
- `image_to_bytes(self, image) -> str`: Converts an image to bytes in PNG format.
- `generate_permutations(self, image_path) -> dict`: Generates a dictionary of transformed images.
- `generate_hashes_and_bytes(self, image_path) -> dict`: Generates the hash-keys and bytes for a given image.
- `process_image_directory(self, folder_path) -> list`: Processes all images in a folder, generating a dictionary for each.

#### ImageScanner

The `ImageScanner` class compares a dataframe of hashes and bytes to an existing database of hashes and bytes, returning matching values via various functions.

**Attributes:**
- `database`: The existing database of image hashes and bytes.
- `matching_images`: List of matching images.
- `matching_images_dict`: Dictionary of matching images.

**Methods:**
- `__init__(self, _database)`: Initializes an `ImageScanner` instance.
- `compare_hashes_fuzzy_crosswise(self, test_hashes, df, threshold = 5)`: Compares each hash from the test_hashes dictionary to the stored image database.
- `print_matches(self, match_list)`: Prints the matches found.

### Usage

To use the `ImagePermutation` and `ImageScanner` classes, you can follow the example provided in the `main.py` file:

```python
from ImagePermutation.ImagePermutation import ImagePermutation, ImageScanner
import pandas as pd

def main():
    ip = ImagePermutation(mode='basic')
    image_db = ip.process_image_directory('test-images/')
    image_db_df = pd.DataFrame(image_db)
    isc = ImageScanner(_database=image_db_df)
    test_hashes = ip.generate_hashes_and_bytes('test-perms/cat-flip.jpg')
    isc.compare_hashes_fuzzy_crosswise(test_hashes, image_db)
    isc.print_matches(isc.matching_images)
    print(pd.DataFrame.from_dict(isc.matching_images_dict))

if __name__ == "__main__":
    main()
