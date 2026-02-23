from torch import nn
from torchvision import models

class DeepFakeModel(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepFakeModel, self).__init__()
        # Use EfficientNet B0 - great balance of speed and accuracy
        # weights='DEFAULT' loads the best available pretrained weights (ImageNet)
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None
            
        self.model = models.efficientnet_b0(weights=weights)
        
        # Modify the classifier head for binary classification (Real vs Fake)
        # EfficientNet's classifier is a Sequential block, the final layer is usually '1'
        num_ftrs = self.model.classifier[1].in_features
        
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), 
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.model(x)
