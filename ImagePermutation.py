import os
import io
import imagehash
import numpy as np
import pandas as pd
from typing import List, Tuple, Literal

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
        self.mode = mode

    def image_to_bytes(self, image):
        with io.BytesIO() as output:
            image.save(output, format='PNG')
            return output.getvalue()

    def generate_permutations(self, image_path, mode):
        # TODO: implement mode check that only executes the base permutations if basic, else all if Advanced
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
        rotations = rotated_image(base_img)
        permutations.update(rotations)
        zoomed = zoomed_image(base_img, zoom_amount=0.05)
        permutations.update(zoomed)

      
        


ip = ImagePermutation()