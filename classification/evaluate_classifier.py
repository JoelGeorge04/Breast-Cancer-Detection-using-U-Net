import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from .classification_model import BreastCancerClassifier, SimpleCNN
from .classification_loader import load_classification_data

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

def evaluate_model(model_path, data_root, model_type='complex', input_size=128, batch_size=8):
    """
    Evaluate a trained classification model
    
    Args:
        model_path: Path to the saved model
        data_root: Root directory containing test data
        model_type: 'simple' or 'complex'
        input_size: Input image size
        batch_size: Batch size for evaluation
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
    
    # Load data
    print("Loading test data...")
    test_loader, test_dataset = load_classification_data(
        data_root,
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )
    
    if len(test_dataset) == 0:
        print("No test data found!")
        return
    
    print(f"Test dataset size: {len(test_dataset)}\n")
    
    # Evaluate
    all_preds = []
    all_probs = []
    all_labels = []
    correct = 0
    total = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.75).float()
            
            # Store predictions
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Calculate accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = 100 * correct / total
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    print("="*60)
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Non-Cancerous (0)', 'Cancerous (1)'],
                                digits=4))
    
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    cm_path   = os.path.join(_RESULTS_DIR, 'confusion_matrix.png')
    dist_path = os.path.join(_RESULTS_DIR, 'prediction_distribution.png')

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Cancerous', 'Cancerous'],
                yticklabels=['Non-Cancerous', 'Cancerous'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\nConfusion matrix saved as '{cm_path}'")
    
    # Plot probability distribution
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_probs[all_labels == 0], bins=30, alpha=0.7, label='Non-Cancerous', color='blue')
    plt.hist(all_probs[all_labels == 1], bins=30, alpha=0.7, label='Cancerous', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(all_probs)), all_probs, c=all_labels, cmap='coolwarm', alpha=0.6, s=10)
    plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability')
    plt.title('Predictions vs True Labels')
    plt.colorbar(label='True Label')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    print(f"Prediction distribution saved as '{dist_path}'")
    
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


if __name__ == "__main__":
    # Example usage
    MODEL_PATH = os.path.join(_PROJECT_ROOT, 'checkpoint', 'breast_cancer_classifier.pth')
    DATA_ROOT  = os.path.join(_PROJECT_ROOT, 'final_dataset', 'images')
    
    results = evaluate_model(
        model_path=MODEL_PATH,
        data_root=DATA_ROOT,
        model_type='complex',  # or 'simple'
        input_size=128,
        batch_size=8
    )
