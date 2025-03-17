import os
import io
import imagehash
import numpy as np
import pandas as pd
from typing import List, Tuple, Literal
from PIL import Image, ImageEnhance, ImageFilter

class ImagePermutation:
    """
    ImagePermutation extracts the hash-maps and bytes from a base image, and various transformed versions of that image, to detect if it already exists in an image database.
    This class can perform these operations on individual, or groups of images.

    Attributes:
        resize (List[int]): The desired width and height for resizing.
        mode (Literal['basic', 'advanced']): The processing mode. Defaults to 'basic'.
    """

    def __init__(self, resize: Tuple[int] = (256,256), mode: Literal['basic', 'advanced'] = 'basic'):
        """
        Initializes an ImagePermutation instance.

        Args:
            resize (List[int]): The desired [width, height] for resizing.
            mode (Literal['basic', 'advanced'], optional): Processing mode. 
                Defaults to 'basic'. Advanced mode generates more permutations that are less common.
        """
        self.resize = resize
        self._mode = None  # Initialize internal mode storage
        self.mode = mode  # Use the setter for validation.

    @property
    def mode(self) -> str:
        """Getter for mode."""
        return self._mode

    @mode.setter
    def mode(self, value: str):
        """Setter for mode validation."""
        if value not in ('basic', 'advanced'):
            raise ValueError(f'Invalid mode [{value}]. Expected basic or advanced')
        self._mode = value

    def image_to_bytes(self, image) -> str:
        with io.BytesIO() as output:
            image.save(output, format='PNG')
            return output.getvalue()

    def generate_permutations(self, image_path) -> dict:
        """
        Generates dictionary of the transformed images
        
        Args:
            image_path: The input image path

        Returns:
            permutations: dictionary of image permuations
        """
        def rotated_image(base_img):
            rotated_image_dict = {}
            for angle in [-5, 5]:
                rotated_image_dict[f"rotated_{angle}_degrees"] = base_img.rotate(
                angle, resample=Image.BICUBIC, fillcolor=128
            )
            return rotated_image_dict

        def zoomed_image(base_img, zoom_amount: float):
            zoomed_image_dict = {}
            w, h = base_img.size
            crop_w, crop_h = int(w * (1-zoom_amount)), int(h * (1 - zoom_amount))
            left = (w - crop_w) // 2
            upper = (h - crop_h) // 2
            zoomed = base_img.crop((left, upper, left + crop_w, upper + crop_h))
            zoomed_image_dict[f'zoomed_{zoom_amount * 100}_percent'] = zoomed.resize(self.resize, Image.LANCZOS)
            return zoomed_image_dict
        
        def brightness_adj_image(base_img):
            returndict = {}
            # Brightness adjustments
            for factor in [0.9, 1.1]:
                enhancer = ImageEnhance.Brightness(base_img)
                returndict[f"brightness_{factor}x"] = enhancer.enhance(factor)
            return returndict

        def contrast_adj_image(base_img):
            returndict = {}
            # Contrast adjustments
            for factor in [0.9, 1.1]:
                enhancer = ImageEnhance.Contrast(base_img)
                returndict[f"contrast_{factor}x"] = enhancer.enhance(factor)
            return returndict

        def simulate_compression(base_img):
            return_dict = {}
            with io.BytesIO() as temp_buffer:
                base_img.save(temp_buffer, format="JPEG", quality=85)
                temp_buffer.seek(0)
                jpeg_img = Image.open(temp_buffer)
                # Ensure consistent format & size
                return_dict["jpeg_compression"] = jpeg_img.convert("L").resize(self.resize)


        mode = self.mode
        permutations = {}
        base_img = Image.open(image_path).resize(self.resize).convert("L")
        
        # Base perms
        h_flip = base_img.transpose(Image.FLIP_LEFT_RIGHT)
        v_flip = base_img.transpose(Image.FLIP_TOP_BOTTOM)
        permutations['original'] = base_img
        permutations['H_Flip'] = h_flip
        permutations['v_Flip'] = v_flip

        # Advanced perms
        if mode == 'advanced':
            rotations = rotated_image(base_img)
            permutations.update(rotations)
            zoomed = zoomed_image(base_img, zoom_amount=0.05)
            permutations.update(zoomed)
            compressed = simulate_compression(base_img)
            permutations.update(compressed)

        return permutations

    def generate_hashes_and_bytes(self, image_path) -> dict:
        """
        Generates the hash-keys and bytes for a given image, requires calling of generate_permutations to generate image variations

        Args:
            image_path: path to the image, to be passed into generate_permutations to generate variations.

        Returns:
            result: dictionary of image, image_path, hashes and bytes.
        """
        # Define 'result' dictionary to store the image and image path alongside its hashes & bytes
        result = {}
        filename = os.path.basename(image_path)
        result["filename"] = filename
        result["filepath"] = image_path

        permutations = self.generate_permutations(image_path)
        for key, img in permutations.items():
            # Compute average hash (convert the result to string)
            img_hash = str(imagehash.average_hash(img))
            result[f"{key}_hash"] = img_hash
            # Convert image to bytes (PNG format)
            result[f"{key}_bytes"] = self.image_to_bytes(img)

        return result

    def process_image_directory(self, folder_path) -> list:
        """
        Process all images in a folder, generating a dictionary for each.
        Only files with extensions in image_extensions are processed.

        Returns:
            list of dictionaries of hashes and bytes of each image in directory.
        """
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        results = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                image_path = os.path.join(folder_path, file)
                data = self.generate_hashes_and_bytes(image_path)
                results.append(data)
        return results  
        

class ImageScanner():
    """
    ImageScanner compares dataframe of hashes & bytes to existing database of hashes and bytes, returning
    matching values via various functions.
    """

    def __init__(self, _database):
        self.database = _database
        self.matching_images = []
        self.matching_images_dict = {}



    def compare_hashes_fuzzy_crosswise(self, test_hashes, df, threshold = 5):
        """
        Compares each hash from the test_hashes dictionary to the stored image database df.

        Args:
            test_hashes -> dict.
            df -> pd.DataFrame.
            threshold -> int, distance threshold between hashes to flag a fuzzy match.

        Returns:
            matching_images -> list of matches, stores in self.matching_images
            matching_images_dict -> dict of matches to turn into df, stores in self.matching_images_dict
        """
        df = self.database
        matching_images = []
        matching_images_dict = {'file': [], 'match_type': [], 'match_level': []}
        test_filename = test_hashes.get("filename", "Test Image")

        # For each row (stored image) in the DataFrame
        for idx, row in df.iterrows():
            existing_filename = row["filename"]
            row_matches = []

            # Compare every hash in test_hashes to every hash in row
            for test_key, test_val in test_hashes.items():
                if test_key.endswith("_hash"):
                    test_hash = imagehash.hex_to_hash(test_val)

                    # Loop over every stored hash in the current row
                    for store_key, store_val in row.items():
                        if store_key.endswith("_hash"):
                            stored_hash = imagehash.hex_to_hash(store_val)
                            distance = test_hash - stored_hash  # Hamming distance
                            if distance == 0:
                                row_matches.append(f"{test_key} vs {store_key} (Exact)")
                                matching_images_dict['file'].append(existing_filename)
                                matching_images_dict['match_type'].append(f"{test_key} vs {store_key}")
                                matching_images_dict['match_level'].append('Exact')
                            elif distance <= threshold:
                                row_matches.append(
                                    f"{test_key} vs {store_key} (Fuzzy, Distance={distance})"
                                )
                                matching_images_dict['file'].append(existing_filename)
                                matching_images_dict['match_type'].append(f"{test_key} vs {store_key}")
                                matching_images_dict['match_level'].append('Fuzzy, Distance={distance}')

            if row_matches:
                matching_images.append((existing_filename, row_matches))

        self.matching_images = matching_images
        self.matching_images_dict = matching_images_dict

    def print_matches(self, match_list):
        match_list = self.matching_images
        if len(match_list) > 0:
            for match in match_list:
                filename, transformations = match
                print(f"Match found with '{filename}': {', '.join(transformations)}")
                print("--------------------")
        else:
            print("Brand new image")

