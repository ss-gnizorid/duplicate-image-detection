import os
import io
import imagehash
import numpy as np
import pandas as pd
from typing import List, Literal

class ImagePermutation:
    def __init__(self, resize: List[int], mode: Literal['basic', 'advanced']):
        self.resize = resize
        self.mode = mode

    def image_to_bytes(self, image):
        with io.BytesIO() as output:
            image.save(output, format='PNG')
            return output.getvalue()