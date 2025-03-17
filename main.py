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