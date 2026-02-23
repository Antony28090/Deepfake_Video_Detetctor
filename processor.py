import torch
import torchvision.transforms as transforms
import cv2
import warnings
import numpy as np
from model import DeepFakeModel
import os

warnings.filterwarnings("ignore")

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize Model ---
# Load the model structure (EfficientNet-B0 based)
model = DeepFakeModel(pretrained=False).to(device) # No need to download ImageNet weights again for inference
try:
    # Model weights are expected to be in the same directory
    # Try loading the best model first, then the final, then the legacy one
    if os.path.exists('deepfake_model_best.pth'):
        weights_path = 'deepfake_model_best.pth'
    elif os.path.exists('deepfake_model_final.pth'):
        weights_path = 'deepfake_model_final.pth'
    else:
        weights_path = 'deepfake_model.pth' # Fallback
        
    print(f"Loading weights from: {weights_path}")
    
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Trained model loaded successfully.")
    model.eval()
except FileNotFoundError:
    print("--- WARNING: No model weights found. Predictions will be random/wrong! ---")
except Exception as e:
    print(f"Error loading model: {e}")
    # Add robust fallback logic here if needed for transitions
    

def extract_and_predict_faces(video_path):
    # NOTE: Training matched 'train.py' logic: Full Frame -> RGB -> Resize(224,224)
    # No MTCNN Face Detection used in training, so we must NOT use it here.

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video file."

    # Store probabilities for soft voting
    fake_probs = []
    
    # EfficientNet Expects 224x224
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    frame_count = 0
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Process every 10th frame (Sampling rate)
        if frame_count % 10 != 0:
            continue

        # Convert to RGB (OpenCV is BGR)
        # This matches the 'Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))' in train.py
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Apply the transform directly to the FULL FRAME
            # unsqueeze(0) to add batch dimension [1, 3, 224, 224]
            input_tensor = transform(rgb_frame).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                
                # Class 1 is Fake, Class 0 is Real
                fake_prob = probs[0][1].item()
                fake_probs.append(fake_prob)
                processed_frames += 1
                
        except Exception as e:
            continue

    cap.release()

    if not fake_probs:
        return "No frames processed."

    # --- Robust Aggregation Strategy ---
    fake_probs_np = np.array(fake_probs)
    
    # Calculate core statistics
    avg_prob = np.mean(fake_probs_np)
    p90_prob = np.percentile(fake_probs_np, 90)
    max_prob = np.max(fake_probs_np)
    
    log_message = (
        f"File: {os.path.basename(video_path)}\n"
        f"Processed Frames: {processed_frames}\n"
        f"Avg: {avg_prob:.4f} | P90: {p90_prob:.4f} | Max: {max_prob:.4f}\n"
        f"--------------------------------------------------\n"
    )
    print(log_message)
    
    # Log to file for debugging
    try:
        with open("debug_last_run.txt", "w") as f:
            f.write(log_message)
    except:
        pass
    
    # Decision Logic v6 (Adaptive Threshold)
    # Problem: 
    # - Real "Noisy" Video: Avg 0.48 (Uncertain), Max 0.88. Flagged as Fake.
    # - Fake "Glitch" Video: Avg 0.12 (Looks Real), Max 0.77. Flagged as Real (initially).
    #
    # Solution: Adaptive Thresholds based on "Base Uncertainty" (Avg Prob).
    
    if avg_prob > 0.4:
        # Case A: Video is confusing/noisy (high base probability).
        # We need STRONGER evidence to call it fake to avoid False Positives.
        # Require substantial sustained evidence (P90) or extreme peak.
        if p90_prob > 0.85 or max_prob > 0.95:
             final_score = max_prob
        else:
             final_score = 0.1 # Default to Real if evidence isn't strong enough
             print("Result: Tuned to Real (High Noise, Insufficient Peak)")

    else:
        # Case B: Video mostly looks real (clean).
        # We look for "Glitches" (spikes in probability).
        if max_prob > 0.7:
            final_score = max_prob
        elif p90_prob > 0.4:
            final_score = p90_prob
        else:
            final_score = avg_prob

    if final_score > 0.5:
        confidence = final_score * 100
        return f"{confidence:.2f}% Fake"
    else:
        confidence = (1 - final_score) * 100
        return f"{confidence:.2f}% Real"
