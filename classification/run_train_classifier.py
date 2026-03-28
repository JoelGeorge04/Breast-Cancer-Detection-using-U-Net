"""
Wrapper script to run classification training from root directory
Uses .venv_cuda environment for GPU support
"""

if __name__ == "__main__":
    import argparse
    import sys, os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _ROOT)
    import torch

    parser = argparse.ArgumentParser(description="Train a breast cancer classifier model")
    parser.add_argument(
        "--model_type",
        type=str,
        default="complex",
        choices=["complex", "simple", "tiny", "mobile", "wide"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Output checkpoint filename (saved under checkpoint/)"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Training learning rate")
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    from classification.train_classifier import train

    default_name_map = {
        "complex": "breast_cancer_classifier.pth",
        "simple": "simple_ensemble.pth",
        "tiny": "tiny_ensemble.pth",
        "mobile": "mobile_ensemble.pth",
        "wide": "wide_ensemble.pth",
    }

    overrides = {
        "model_type": args.model_type,
        "model_name": args.model_name or default_name_map[args.model_type],
    }
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    print(f"Training config override: {overrides}\n")
    train(config_overrides=overrides)
