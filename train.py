import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from model import DeepFakeModel
import os
import cv2
from PIL import Image
import copy
import glob
from tqdm import tqdm
import random

# --- Configuration ---
# Updated to the path where kagglehub saves ff-c23
DATA_DIR = "/kaggle/input/ff-c23/FaceForensics++_C23" 
BATCH_SIZE = 64  # Increased for H100 power
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Video Dataset ---
class VideoDeepFakeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, frames_per_video=1):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.video_paths = []
        self.labels = []
        
        # FF++ Structure: Looks for .mp4 files in original and manipulated sequences
        search_path = os.path.join(root_dir, '**', '*.mp4')
        all_videos = glob.glob(search_path, recursive=True)
        
        print(f"Indexing {len(all_videos)} video files...")
        
        for v_path in all_videos:
            path_lower = v_path.lower()
            # FF++ specific folder names: 'original' vs 'manipulated'
            if 'original' in path_lower or 'youtube' in path_lower:
                self.video_paths.append(v_path)
                self.labels.append(0) # Real
            elif 'manipulated' in path_lower or 'deepfakes' in path_lower:
                self.video_paths.append(v_path)
                self.labels.append(1) # Fake
                
        print(f"Found {len(self.video_paths)} videos. Real: {self.labels.count(0)}, Fake: {self.labels.count(1)}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        v_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Use OpenCV to capture a frame from the video
        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Select a random frame index to avoid just seeing the first frame of every video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = random.randint(0, max(0, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return self.__getitem__(random.randint(0, len(self) - 1))
            
        # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        if self.transform:
            image = self.transform(image)

        return image, label

# --- Training Function with H100 Optimizations ---
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scaler = GradScaler() # For Mixed Precision training on H100

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Use autocast for faster training on modern GPUs
                    with autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save to Google Drive if mounted, otherwise local
                save_path = '/content/drive/MyDrive/deepfake_best.pth' if os.path.exists('/content/drive') else 'best_model.pth'
                torch.save(model.state_dict(), save_path)

    model.load_state_dict(best_model_wts)
    return model

def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Use the High-Speed H100 local path
    dataset_path = DATA_DIR
    if not os.path.exists(dataset_path):
        print(f"ERROR: {dataset_path} not found. Ensure download_data.py finished successfully.")
        return

    full_dataset = VideoDeepFakeDataset(dataset_path, transform=data_transforms['train'])
    
    # 80/20 Train/Val Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4), # Workers > 0 for H100
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }

    model = DeepFakeModel(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS)

if __name__ == "__main__":
    main()
