"""
Wrapper script to run classification evaluation from root directory
Uses .venv_cuda environment for GPU support
"""

if __name__ == "__main__":
    import sys, os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _ROOT)
    import torch

    MODEL_PATH = os.path.join(_ROOT, 'checkpoint', 'breast_cancer_classifier.pth')
    DATA_ROOT  = os.path.join(_ROOT, 'final_dataset', 'images')
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using: python run_train_classifier.py")
    else:
        from classification.evaluate_classifier import evaluate_model
        
        results = evaluate_model(
            model_path=MODEL_PATH,
            data_root=DATA_ROOT,
            model_type='complex',
            input_size=128,
            batch_size=8
        )
