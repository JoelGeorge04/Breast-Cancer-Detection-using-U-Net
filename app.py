import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'classification'))

from classification_model import build_classifier

# Constants
IMAGE_SIZE = 128
DECISION_THRESHOLD = 0.5

# Ensemble model definitions.
ENSEMBLE_MODELS = [
    {
        "name": "Complex CNN",
        "model_type": "complex",
        "path": "checkpoint/breast_cancer_classifier.pth",
    },
    {
        "name": "TinyCNN (ensemble)",
        "model_type": "tiny",
        "path": "checkpoint/tiny_ensemble.pth",
    },
    {
        "name": "MobileStyleCNN (ensemble)",
        "model_type": "mobile",
        "path": "checkpoint/mobile_ensemble.pth",
    },
    {
        "name": "WideShallowCNN (ensemble)",
        "model_type": "wide",
        "path": "checkpoint/wide_ensemble.pth",
    },
]

# Set page config
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load available ensemble models from checkpoint paths."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaded_models = []
    missing_models = []

    for model_cfg in ENSEMBLE_MODELS:
        model_path = model_cfg["path"]
        if not os.path.exists(model_path):
            missing_models.append(model_cfg)
            continue

        model = build_classifier(model_cfg["model_type"], input_size=IMAGE_SIZE)
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        model.load_state_dict(state)
        model.eval()
        model.to(device)

        loaded_models.append(
            {
                "name": model_cfg["name"],
                "model_type": model_cfg["model_type"],
                "path": model_path,
                "model": model,
            }
        )

    return loaded_models, missing_models, device

def generate_mask(image, K=3):
    """Generate mask using K-means clustering"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Ensure RGB format
    if len(img_array.shape) == 2:  # Grayscale
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_array
    
    # Reshape for K-means
    pixel_vals = img_rgb.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    
    # K-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10).fit(pixel_vals)
    labels = kmeans.labels_.reshape(img_rgb.shape[:2])
    
    # Get the most common cluster 
    unique, counts = np.unique(labels, return_counts=True)
    tumor_cluster = unique[np.argmax(counts)]
    
    # Create binary mask
    mask = np.uint8(labels == tumor_cluster) * 255
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 7)
    
    return mask

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Resize to training size
    img_resized = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to RGB if needed
    if len(img_resized.shape) == 2:  # Grayscale
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif img_resized.shape[2] == 4:  # RGBA
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
    
    # Normalize
    img_normalized = img_resized.astype("float32") / 255.0
    
    # Transpose to (C, H, W) format
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    
    # Convert to tensor
    img_tensor = torch.tensor(img_transposed).unsqueeze(0)
    
    return img_tensor, img_resized

def predict_classification_ensemble(loaded_models, img_tensor, device):
    """Generate ensemble prediction by averaging per-model probabilities."""
    img_tensor = img_tensor.to(device)
    model_outputs = []

    with torch.no_grad():
        for model_entry in loaded_models:
            output = model_entry["model"](img_tensor)
            probability = torch.sigmoid(output).item()
            model_outputs.append(
                {
                    "name": model_entry["name"],
                    "model_type": model_entry["model_type"],
                    "probability": probability,
                }
            )

    probability = float(np.mean([m["probability"] for m in model_outputs]))

    # Class 0: Non-cancerous, Class 1: Cancerous
    predicted_class = 1 if probability > DECISION_THRESHOLD else 0
    confidence = probability if predicted_class == 1 else (1 - probability)

    return predicted_class, probability, confidence, model_outputs

# Main UI
st.title(" Breast Cancer Classification")
st.markdown("Upload a medical image to classify as **cancerous** or **non-cancerous**.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        """
        This application uses a **Deep Learning CNN model** 
        to classify breast cancer images.
        
        **How to use:**
        1. Upload a medical image (PNG, JPG, JPEG)
        2. View the classification results
        3. See confidence scores
        
        **Classes:**
        - **Class 0**: Non-cancerous 
        - **Class 1**: Cancerous 
        """
    )
    
    st.header("Model Info")
    st.success(f"**Accuracy**: 83.24%")
    st.info(f"**Image Size**: {IMAGE_SIZE}x{IMAGE_SIZE}")
    st.info(f"**Decision Threshold**: {DECISION_THRESHOLD}")
    device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.info(f"**Device**: {device_info}")

# Load ensemble models
loaded_models, missing_models, device = load_models()

if len(loaded_models) > 0:
    with st.sidebar:
        st.success(f"**Loaded Models**: {len(loaded_models)}/{len(ENSEMBLE_MODELS)}")
        loaded_names = [m["name"] for m in loaded_models]
        st.caption("Active ensemble members:")
        for name in loaded_names:
            st.write(f"- {name}")

        if len(missing_models) > 0:
            st.warning("Some ensemble checkpoints are missing. Running with available models only.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a medical image...",
        type=["png", "jpg", "jpeg"],
        help="Upload a breast cancer medical image for classification"
    )
    
    if uploaded_file is not None:
        # Read and display original image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Process image
        with st.spinner(" Processing image..."):
            img_tensor, img_resized = preprocess_image(image)
            predicted_class, probability, confidence, model_outputs = predict_classification_ensemble(
                loaded_models, img_tensor, device
            )
            # Generate mask
            mask = generate_mask(image)
        
        # Display results in 3 columns
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader(" Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader(" Generated Mask")
            st.image(mask, use_container_width=True, clamp=True)
        
        with col3:
            st.subheader(" Classification Results")
            
            # Show prediction with color coding
            if predicted_class == 0:
                st.success("###  Non-Cancerous")
                st.markdown("The image is classified as **non-cancerous**.")
            else:
                st.error("###  Cancerous")
                st.markdown("The image is classified as **cancerous**.")
            
            # Confidence metrics
            st.markdown("---")
            st.metric(
                "Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.metric(
                "Ensemble Members",
                f"{len(loaded_models)}"
            )
            
            class_label = "Class 1 (Cancerous)" if predicted_class == 1 else "Class 0 (Non-cancerous)"
            st.metric(
                "Predicted Class",
                class_label
            )
            
            # Progress bars for probability
            st.markdown("---")
            st.markdown("**Probability Distribution:**")
            
            non_cancerous_prob = (1 - probability) * 100
            cancerous_prob = probability * 100
            
            st.markdown(f"**Non-Cancerous:** {non_cancerous_prob:.2f}%")
            st.progress(non_cancerous_prob / 100)
            
            st.markdown(f"**Cancerous:** {cancerous_prob:.2f}%")
            st.progress(cancerous_prob / 100)
        
        # Additional info
        with st.expander(" Detailed Information"):
            st.markdown(f"""
            - **Raw Probability Score**: {probability:.4f}
            - **Decision Threshold**: {DECISION_THRESHOLD}
            - **Image Size**: {IMAGE_SIZE}x{IMAGE_SIZE} pixels
            - **Model Accuracy**: 83.24%
            - **Processing Device**: {device}
            """)

        with st.expander(" Ensemble Breakdown"):
            for output in model_outputs:
                st.write(
                    f"- {output['name']} ({output['model_type']}): "
                    f"{output['probability'] * 100:.2f}% cancer probability"
                )
    
    else:
        st.info(" Please upload an image to begin classification")
else:
    model_paths = "\n".join([f"- {m['path']}" for m in ENSEMBLE_MODELS])
    st.error("Failed to load any ensemble model checkpoint. Please train/save at least one of these:\n" + model_paths)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit • CNN Classification • PyTorch</p>
    </div>
    """,
    unsafe_allow_html=True
)


#  .\.venv_py310\Scripts\streamlit run app.py