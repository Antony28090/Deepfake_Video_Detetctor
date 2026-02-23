import torch
from model import DeepFakeModel
from processor import extract_and_predict_faces
import os

def test_model():
    print("Testing Model instantiation...")
    try:
        model = DeepFakeModel(pretrained=False) # Don't download weights for test
        print("Model created successfully.")
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        print(f"Output shape: {out.shape}")
        assert out.shape == (1, 2)
        print("Forward pass successful.")
    except Exception as e:
        print(f"Model test failed: {e}")

def test_imports():
    print("\nTesting Imports for train.py...")
    try:
        import train
        print("train.py imported successfully.")
    except Exception as e:
        print(f"train.py import failed: {e}")

if __name__ == "__main__":
    test_model()
    test_imports()
