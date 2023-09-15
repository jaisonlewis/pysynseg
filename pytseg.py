import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import cv2
import os
import matplotlib.pyplot as plt
import albumentations as A
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
import wandb
from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger
from torchvision.transforms import ToTensor
import numpy as np

train_losses = []
val_losses = []

# Initialize wandb
wandb.init(project="syn-seg")
WANDB_START_METHOD="thread"


# Create directories for images and masks
os.makedirs('new_syn_images', exist_ok=True)
os.makedirs('new_syn_masks', exist_ok=True)

# Check directories
if not os.path.exists('new_syn_images') or not os.path.exists('new_syn_masks'):
    raise FileNotFoundError('Image or mask directory not found')

# DataLoader handles data loading and preprocessing
class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.use_augmentations = False
        
        # Define image transform
        self.image_transform = A.Compose([
            ToPILImage(),
            ToTensor()
        ])
        
        # Define mask transform
        self.mask_transform = A.Compose([
            ToPILImage(),
            ToTensor()
        ])
        
        # Define augmentations
        self.augmentations = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ElasticTransform(),
            A.OpticalDistortion(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = image / 255.0 # Normalize
        mask = cv2.imread(mask_path, 0)
        
        # Apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
      
        # Convert to PIL Image
        image = Image.fromarray((image * 255).astype('uint8'))
        mask = Image.fromarray(mask.astype('uint8'))
        
        # Apply transforms
        image = ToTensor()(image)
        mask = ToTensor()(mask)
        
        return image, mask

# Create dataset
image_dataset = Dataset(image_dir='new_syn_images', mask_dir='new_syn_masks')

# Split dataset
n_val = int(len(image_dataset) * 0.2)
n_train = len(image_dataset) - n_val
train_set, val_set = torch.utils.data.random_split(image_dataset, [n_train, n_val])

# Dataloaders
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

# Initialize model, optimizer
model = smp.Unet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Scheduler, SummaryWriter
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
writer = SummaryWriter()

criterion = torch.nn.BCEWithLogitsLoss()


# Training loop
def train():
    num_epochs = 10  # Define the number of epochs
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
        wandb.log({
            "Epoch": epoch,
            "Train Loss": train_loss / len(train_loader),
            "Validation Loss": val_loss / len(val_loader),
            "Learning Rate": scheduler.get_last_lr()[0]
        })
        scheduler.step()
        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)
        writer.add_image('Images/Validation', x[0], epoch)
        print(f'Epoch {epoch + 1}/{num_epochs} Train Loss: {train_loss / len(train_loader)} Val Loss: {val_loss / len(val_loader)}')
        train_losses.append(train_loss / len(train_loader))  # Update the train_losses list
        val_losses.append(val_loss / len(val_loader))  # Update the val_losses list
sweep_config = {
  "method": "bayes",
  "metric": {"name": "val_loss", "goal": "minimize"},    
  "parameters": {
    "lr": {"distribution": "uniform", "min": 0.0001, "max": 0.01},
    "batch_size": {"values": [8, 16, 32]},
  }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="segmentation")

# Run the sweep
wandb.agent(sweep_id, function=train)

# Visualization
# After training, plot the losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualization: Show sample input images, ground truth masks, and model predictions
n_samples = 5
sample_indices = np.random.randint(0, len(x_val), n_samples)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(sample_indices):
    plt.subplot(3, n_samples, i + 1)
    plt.imshow(x_val[idx])
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(3, n_samples, n_samples + i + 1)
    plt.imshow(np.squeeze(y_val[idx]), cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(3, n_samples, 2 * n_samples + i + 1)
    predicted_mask = best_model.predict(np.expand_dims(x_val[idx], axis=0))
    plt.imshow(np.squeeze(predicted_mask), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
plt.tight_layout()
plt.show()
