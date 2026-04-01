import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from .classification_model import BreastCancerClassifier, SimpleCNN

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def preprocess_image(image_path, target_size=128):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image tensor
    """
    # Load image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize
    img = cv2.resize(img, (target_size, target_size))
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype("float32") / 255.0
    
    # Convert to tensor format (C, H, W)
    img = np.transpose(img, (2, 0, 1))
    
    # Add batch dimension
    img = torch.from_numpy(img).unsqueeze(0)
    
    return img


def predict_single_image(model, image_path, device, target_size=128):
    """
    Predict on a single image
    
    Args:
        model: Trained model
        image_path: Path to image
        device: Device to run on
        target_size: Image size
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    img_tensor = preprocess_image(image_path, target_size).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probability = torch.sigmoid(output).item()
        prediction = 1 if probability > 0.5 else 0
    
    return {
        'prediction': prediction,
        'probability': probability,
        'label': 'Cancerous' if prediction == 1 else 'Non-Cancerous',
        'confidence': probability if prediction == 1 else (1 - probability)
    }


def predict_batch(model_path, image_paths, model_type='complex', input_size=128, visualize=True):
    """
    Predict on multiple images
    
    Args:
        model_path: Path to saved model
        image_paths: List of image paths
        model_type: 'simple' or 'complex'
        input_size: Input image size
        visualize: Whether to visualize results
    """
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading model from {model_path}...")
    if model_type == 'simple':
        model = SimpleCNN(input_size=input_size)
    else:
        model = BreastCancerClassifier(input_size=input_size)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully\n")
    
    # Predict on each image
    results = []
    for img_path in image_paths:
        print(f"Processing: {img_path}")
        try:
            result = predict_single_image(model, img_path, device, input_size)
            result['image_path'] = img_path
            results.append(result)
            
            print(f"  Prediction: {result['label']}")
            print(f"  Probability: {result['probability']:.4f}")
            print(f"  Confidence: {result['confidence']:.2%}\n")
        except Exception as e:
            print(f"  Error: {e}\n")
    
    # Visualize if requested
    if visualize and len(results) > 0:
        visualize_predictions(results, input_size)
    
    return results


def visualize_predictions(results, target_size=128):
    """
    Visualize prediction results
    """
    n_images = len(results)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx, result in enumerate(results):
        # Load and display image
        img = cv2.imread(result['image_path'])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img)
        
        # Set title with prediction
        color = 'red' if result['prediction'] == 1 else 'green'
        title = f"{result['label']}\n"
        title += f"Prob: {result['probability']:.3f} ({result['confidence']:.1%})"
        axes[idx].set_title(title, color=color, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'predictions_visualization.png'")
    plt.show()


def predict_from_folder(model_path, folder_path, model_type='complex', input_size=128, max_images=10):
    """
    Predict on all images in a folder
    
    Args:
        model_path: Path to saved model
        folder_path: Path to folder containing images
        model_type: 'simple' or 'complex'
        input_size: Input image size
        max_images: Maximum number of images to process
    """
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = []
    
    for file in os.listdir(folder_path):
        if file.lower().endswith(image_extensions):
            image_paths.append(os.path.join(folder_path, file))
            if len(image_paths) >= max_images:
                break
    
    if len(image_paths) == 0:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Found {len(image_paths)} images to process\n")
    
    # Predict
    results = predict_batch(model_path, image_paths, model_type, input_size)
    
    # Summary statistics
    if len(results) > 0:
        n_cancerous = sum(1 for r in results if r['prediction'] == 1)
        n_non_cancerous = len(results) - n_cancerous
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total images: {len(results)}")
        print(f"Cancerous: {n_cancerous} ({100*n_cancerous/len(results):.1f}%)")
        print(f"Non-Cancerous: {n_non_cancerous} ({100*n_non_cancerous/len(results):.1f}%)")
        print("="*60)
    
    return results


if __name__ == "__main__":
    # Example 1: Predict on a single image
    print("="*60)
    print("Example 1: Single Image Prediction")
    print("="*60)

    MODEL_PATH = os.path.join(_PROJECT_ROOT, 'checkpoint', 'breast_cancer_classifier.pth')

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Please train the model first using train_classifier.py")
    else:
        # Find a sample image
        sample_folders = [
            os.path.join(_PROJECT_ROOT, 'final_dataset', 'images', '0'),
            os.path.join(_PROJECT_ROOT, 'final_dataset', 'images', '1'),
        ]

        sample_image = None
        for folder in sample_folders:
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    sample_image = os.path.join(folder, files[0])
                    break

        if sample_image:
            results = predict_batch(
                model_path=MODEL_PATH,
                image_paths=[sample_image],
                model_type='complex',
                input_size=128,
                visualize=True
            )
        else:
            print("No sample images found in final_dataset/images/")

        # Example 2: Predict on folder
        print("\n" + "="*60)
        print("Example 2: Folder Prediction")
        print("="*60)

        test_folder = os.path.join(_PROJECT_ROOT, 'final_dataset', 'images', '1')
        if os.path.exists(test_folder):
            results = predict_from_folder(
                model_path=MODEL_PATH,
                folder_path=test_folder,
                model_type='complex',
                input_size=128,
                max_images=5
            )
