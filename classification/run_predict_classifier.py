"""
Wrapper script to run classification predictions from root directory
"""

if __name__ == "__main__":
    import sys, os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _ROOT)
    from classification.predict_classifier import predict_from_folder

    MODEL_PATH  = os.path.join(_ROOT, 'checkpoint', 'breast_cancer_classifier.pth')
    TEST_FOLDER = os.path.join(_ROOT, 'final_dataset', 'images', '1')  # Change to your test folder
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using: python run_train_classifier.py")
    elif not os.path.exists(TEST_FOLDER):
        print(f"Error: Test folder not found: {TEST_FOLDER}")
        print("Please specify a valid folder path in this script")
    else:
        print(f"Running predictions on images in: {TEST_FOLDER}\n")
        
        results = predict_from_folder(
            model_path=MODEL_PATH,
            folder_path=TEST_FOLDER,
            model_type='complex',
            input_size=128,
            max_images=10
        )
