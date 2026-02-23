import torch
from model import DeepFakeModel
import os

def verify():
    print("Verifying model loading...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFakeModel(pretrained=False).to(device)
    
    path = 'deepfake_model_best.pth'
    if not os.path.exists(path):
        print(f"FAILED: {path} not found.")
        return

    try:
        # Load state dict
        # strict=False allows loading if there are minor mismatches (e.g. classifier head name changes)
        # But for 'near perfect' we want exact match if possible.
        # User defined EfficientNet in model.py, hopefully trained model matches EXACTLY.
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict, strict=True) 
        print(f"SUCCESS: Loaded {path} with strict=True.")
        
        # Test inference shape
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(dummy_input)
        print(f"Inference Test: Output shape {output.shape}")
        
    except Exception as e:
        print(f"FAILED: Error loading weights: {e}")
        print("Tip: If Architecture mismatch, ensure model.py matches train.py definition exactly.")

if __name__ == "__main__":
    verify()
