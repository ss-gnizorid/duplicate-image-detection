from ImagePermutation.ImagePermutation import ImagePermutation, ImageScanner
import pandas as pd

def main():
    
    ip = ImagePermutation(mode='basic')
    image_db = ip.process_image_directory(
        '/home/sagemaker-user/duplicate-images/duplicate-image-detection/test-images'
    )
    image_db_df = pd.DataFrame(image_db)

    isc = ImageScanner(_database=image_db_df)
    test_hashes = ip.process_image_directory(
        '/home/sagemaker-user/duplicate-images/duplicate-image-detection/test-perms'
    )
    test_df = isc.batch_test_hashes_to_df(test_hashes)
    matches_df = isc.compare_hashes_fuzzy_crosswise_df(test_df, threshold=5)
    
    isc.print_matches_df()

if __name__ == "__main__":
    main()
