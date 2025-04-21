import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import time
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to empty CUDA cache
def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Define the dataset class
class GlacierChangeDataset(Dataset):
    def __init__(self, data_dir, is_train=True, transform=None):
        """
        Args:
            data_dir (str): Directory containing the dataset
            is_train (bool): If True, return training set, else test set
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all image and mask file pairs
        image_files = sorted([f for f in os.listdir(os.path.join(data_dir, 'images')) if f.endswith('.npy')])
        mask_files = sorted([f for f in os.listdir(os.path.join(data_dir, 'masks')) if f.endswith('.npy')])
        
        # Ensure matching pairs
        valid_pairs = []
        for img_file in image_files:
            base_name = img_file.replace('.npy', '')
            mask_file = f"{base_name}.npy"
            if mask_file in mask_files:
                valid_pairs.append((img_file, mask_file))
        
        # Split into train/test sets (80/20 split)
        train_pairs, test_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)
        
        # Use either train or test set based on is_train flag
        self.file_pairs = train_pairs if is_train else test_pairs
        
        print(f"{'Training' if is_train else 'Testing'} dataset: {len(self.file_pairs)} samples")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        # Load image and mask
        img_file, mask_file = self.file_pairs[idx]
        
        # Load data
        img_path = os.path.join(self.data_dir, 'images', img_file)
        mask_path = os.path.join(self.data_dir, 'masks', mask_file)
        
        image = np.load(img_path)
        mask = np.load(mask_path)
        
        # Check if image is in [H, W, C] format and transpose to [C, H, W]
        if len(image.shape) == 3 and image.shape[-1] == 16:  # If last dimension is channels
            image = np.transpose(image, (2, 0, 1))  # Change to [C, H, W]
        
        # Normalize data - ensure values are in appropriate ranges
        image = np.clip(image, 0, 1)  # Assume images should be in [0, 1]
        mask = np.clip(mask, 0, 1)    # Ensure masks are binary [0, 1]
        
        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        # Final safety check
        image = torch.clamp(image, 0, 1)
        mask = torch.clamp(mask, 0, 1)
        
        return image, mask

# Function to check dataset samples
def check_dataset_values(dataset, num_samples=5):
    """Check a few samples from the dataset to verify value ranges"""
    for i in range(min(num_samples, len(dataset))):
        image, mask = dataset[i]
        
        print(f"Sample {i}:")
        print(f"  Image shape: {image.shape}, min: {image.min().item():.4f}, max: {image.max().item():.4f}")
        print(f"  Mask shape: {mask.shape}, min: {mask.min().item():.4f}, max: {mask.max().item():.4f}")
        print(f"  Mask values in [0,1]: {torch.all((mask >= 0) & (mask <= 1)).item()}")
        
        # Check for NaN or inf values
        print(f"  Image has NaN: {torch.isnan(image).any().item()}, Inf: {torch.isinf(image).any().item()}")
        print(f"  Mask has NaN: {torch.isnan(mask).any().item()}, Inf: {torch.isinf(mask).any().item()}")
        print()

# Attention Gate
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

# Define the efficient U-Net with Attention model
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=16, out_channels=1):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, 32)  # Reduced from 64
        self.enc2 = self._block(32, 64)           # Reduced from 128
        self.enc3 = self._block(64, 128)          # Reduced from 256
        self.enc4 = self._block(128, 256)         # Reduced from 512
        
        # Bottleneck
        self.bottleneck = self._block(256, 512)   # Reduced from 1024
        
        # Decoder with attention
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec4 = self._block(512, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec3 = self._block(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec2 = self._block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=32, F_l=32, F_int=16)
        self.dec1 = self._block(64, 32)
        
        # Final output
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got {x.shape}")
        if x.size(1) != 16:
            raise ValueError(f"Expected 16 channels, got {x.size(1)}")
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with attention
        dec4 = self.upconv4(bottleneck)
        dec4 = self.att4(g=dec4, x=enc4)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))
        
        dec3 = self.upconv3(dec4)
        dec3 = self.att3(g=dec3, x=enc3)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))
        
        dec2 = self.upconv2(dec3)
        dec2 = self.att2(g=dec2, x=enc2)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        
        dec1 = self.upconv1(dec2)
        dec1 = self.att1(g=dec1, x=enc1)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        
        # Final output
        out = self.final(dec1)
        
        return out

# Define an inefficient Complex CNN model for comparison
class ComplexCNN(nn.Module):
    def __init__(self, in_channels=16, out_channels=1):
        super(ComplexCNN, self).__init__()
        
        # Overly complex encoder with redundant processing - but with more stable initialization
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Added BatchNorm for stability
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1),  # 1x1 conv adds computation with minimal benefit
            nn.BatchNorm2d(64),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)  # Added BatchNorm
        )
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),  # Another 1x1 conv
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Overly deep network with redundant blocks but more stable
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),  # More 1x1 convs
            nn.BatchNorm2d(256)
        )
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512)
        )
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Excessive processing in the bottleneck but with BatchNorm
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024)
        )
        
        # Inefficient decoder with repeated upsampling + conv instead of transposed convs
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512)
        )
        
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256)
        )
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Multiple final convolutions when one would suffice
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )
        
        # Initialize weights (using a slightly different init for stability)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use Xavier/Glorot initialization for stability
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Check dimensions
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got {x.shape}")
        if x.size(1) != 16:
            raise ValueError(f"Expected 16 channels, got {x.size(1)}")
        
        # Encoder path with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder path with skip connections
        dec4 = self.upsample4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upsample3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upsample2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upsample1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Final output
        output = self.final(dec1)
        
        return output

# Define loss function
class DiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCEWithLogitsLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight, reduction='mean')
    
    def forward(self, inputs, targets, smooth=1):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # For Dice Loss, we need sigmoid here since we're using logits
        inputs_sigmoid = torch.sigmoid(inputs)
        
        # Dice Loss calculation
        inputs_sigmoid = inputs_sigmoid.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs_sigmoid * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_sigmoid.sum() + targets.sum() + smooth)
        
        # Combined loss
        return bce_loss + dice_loss

# Comparative training and evaluation function
def compare_models(train_loader, test_loader, criterion, epochs=10):
    """
    Train and compare the efficient AttentionUNet vs inefficient ComplexCNN
    """
    # Initialize models
    efficient_model = AttentionUNet(in_channels=16, out_channels=1).to(device)
    inefficient_model = ComplexCNN(in_channels=16, out_channels=1).to(device)
    
    # Initialize optimizers with lower learning rate and weight decay for stability
    efficient_optimizer = torch.optim.Adam(efficient_model.parameters(), lr=0.0001, weight_decay=1e-5)
    inefficient_optimizer = torch.optim.Adam(inefficient_model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # For tracking metrics
    efficient_train_losses = []
    efficient_val_losses = []
    inefficient_train_losses = []
    inefficient_val_losses = []
    
    # For tracking time
    efficient_times = []
    inefficient_times = []
    
    # For mixed precision training
    scaler = GradScaler()
    
    print("Starting comparative training...")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # ======= EFFICIENT MODEL TRAINING =======
        efficient_start_time = time.time()
        efficient_model.train()
        efficient_train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc="Efficient Model Training"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            efficient_optimizer.zero_grad()
            
            # Use autocast for mixed precision
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = efficient_model(inputs)
                loss = criterion(outputs, targets)
            
            # Scale the loss and do backward pass
            scaler.scale(loss).backward()
            
            # Apply gradient clipping for stability
            scaler.unscale_(efficient_optimizer)
            torch.nn.utils.clip_grad_norm_(efficient_model.parameters(), max_norm=1.0)
            
            scaler.step(efficient_optimizer)
            scaler.update()
            
            efficient_train_loss += loss.item() * inputs.size(0)
        
        efficient_train_loss = efficient_train_loss / len(train_loader.dataset)
        efficient_train_losses.append(efficient_train_loss)
        
        # Validation phase
        efficient_model.eval()
        efficient_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Efficient Model Validation"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = efficient_model(inputs)
                    loss = criterion(outputs, targets)
                
                efficient_val_loss += loss.item() * inputs.size(0)
        
        efficient_val_loss = efficient_val_loss / len(test_loader.dataset)
        efficient_val_losses.append(efficient_val_loss)
        
        efficient_epoch_time = time.time() - efficient_start_time
        efficient_times.append(efficient_epoch_time)
        
        print(f"Efficient Model - Train Loss: {efficient_train_loss:.4f}, Val Loss: {efficient_val_loss:.4f}, Time: {efficient_epoch_time:.2f}s")
        
        # Free up memory
        empty_cuda_cache()
        
        # ======= INEFFICIENT MODEL TRAINING =======
        inefficient_start_time = time.time()
        inefficient_model.train()
        inefficient_train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc="Inefficient Model Training"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            inefficient_optimizer.zero_grad()
            
            # Use autocast for mixed precision
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = inefficient_model(inputs)
                loss = criterion(outputs, targets)
            
            # Scale the loss and do backward pass
            scaler.scale(loss).backward()
            
            # Apply gradient clipping for stability
            scaler.unscale_(inefficient_optimizer)
            torch.nn.utils.clip_grad_norm_(inefficient_model.parameters(), max_norm=1.0)
            
            scaler.step(inefficient_optimizer)
            scaler.update()
            
            inefficient_train_loss += loss.item() * inputs.size(0)
        
        inefficient_train_loss = inefficient_train_loss / len(train_loader.dataset)
        inefficient_train_losses.append(inefficient_train_loss)
        
        # Validation phase
        inefficient_model.eval()
        inefficient_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Inefficient Model Validation"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = inefficient_model(inputs)
                    loss = criterion(outputs, targets)
                
                inefficient_val_loss += loss.item() * inputs.size(0)
        
        inefficient_val_loss = inefficient_val_loss / len(test_loader.dataset)
        inefficient_val_losses.append(inefficient_val_loss)
        
        inefficient_epoch_time = time.time() - inefficient_start_time
        inefficient_times.append(inefficient_epoch_time)
        
        print(f"Inefficient Model - Train Loss: {inefficient_train_loss:.4f}, Val Loss: {inefficient_val_loss:.4f}, Time: {inefficient_epoch_time:.2f}s")
        
        # Free up memory
        empty_cuda_cache()
    
    # Save models
    torch.save({
        'model_state_dict': efficient_model.state_dict(),
        'optimizer_state_dict': efficient_optimizer.state_dict(),
    }, 'efficient_model.pth')
    
    torch.save({
        'model_state_dict': inefficient_model.state_dict(),
        'optimizer_state_dict': inefficient_optimizer.state_dict(),
    }, 'inefficient_model.pth')
    
    # Return models and metrics for further analysis
    return {
        'efficient_model': efficient_model,
        'inefficient_model': inefficient_model,
        'efficient_train_losses': efficient_train_losses,
        'efficient_val_losses': efficient_val_losses,
        'inefficient_train_losses': inefficient_train_losses,
        'inefficient_val_losses': inefficient_val_losses,
        'efficient_times': efficient_times,
        'inefficient_times': inefficient_times
    }

def plot_model_comparison(metrics):
    """
    Generate comparison plots for the efficient vs inefficient models
    """
    epochs = list(range(1, len(metrics['efficient_train_losses']) + 1))
    
    # 1. Training Loss Comparison
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, metrics['efficient_train_losses'], 'b-', marker='o', label='Efficient Model (Training)')
    plt.plot(epochs, metrics['efficient_val_losses'], 'b--', marker='s', label='Efficient Model (Validation)')
    plt.plot(epochs, metrics['inefficient_train_losses'], 'r-', marker='o', label='Inefficient Model (Training)')
    plt.plot(epochs, metrics['inefficient_val_losses'], 'r--', marker='s', label='Inefficient Model (Validation)')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_loss_comparison.png', dpi=300)
    plt.close()
    
    # 2. Training Time Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(np.array(epochs) - 0.2, metrics['efficient_times'], width=0.4, label='Efficient Model', color='blue', alpha=0.7)
    plt.bar(np.array(epochs) + 0.2, metrics['inefficient_times'], width=0.4, label='Inefficient Model', color='red', alpha=0.7)
    
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('model_time_comparison.png', dpi=300)
    plt.close()
    
    # 3. Cumulative Training Time
    efficient_cumulative = np.cumsum(metrics['efficient_times'])
    inefficient_cumulative = np.cumsum(metrics['inefficient_times'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, efficient_cumulative, 'b-', marker='o', label='Efficient Model')
    plt.plot(epochs, inefficient_cumulative, 'r-', marker='o', label='Inefficient Model')
    
    plt.xlabel('Epochs')
    plt.ylabel('Cumulative Time (seconds)')
    plt.title('Cumulative Training Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cumulative_time_comparison.png', dpi=300)
    plt.close()
    
    # 5. Model Efficiency Summary
    # Calculate average metrics
    avg_efficient_time = np.mean(metrics['efficient_times'])
    avg_inefficient_time = np.mean(metrics['inefficient_times'])
    time_increase = (avg_inefficient_time / avg_efficient_time - 1) * 100
    
    best_efficient_val = min(metrics['efficient_val_losses'])
    best_inefficient_val = min(metrics['inefficient_val_losses'])
    
    efficient_params = sum(p.numel() for p in metrics['efficient_model'].parameters())
    inefficient_params = sum(p.numel() for p in metrics['inefficient_model'].parameters())
    param_increase = (inefficient_params / efficient_params - 1) * 100
    
    # Create table for the paper
    model_summary = {
        'Model': ['Efficient (AttentionUNet)', 'Inefficient (ComplexCNN)', 'Difference (%)'],
        'Parameters': [f"{efficient_params:,}", f"{inefficient_params:,}", f"+{param_increase:.1f}%"],
        'Training Time/Epoch': [f"{avg_efficient_time:.2f}s", f"{avg_inefficient_time:.2f}s", f"+{time_increase:.1f}%"],
        'Best Validation Loss': [f"{best_efficient_val:.4f}", f"{best_inefficient_val:.4f}", ""]
    }
    
    # Convert to visualization
    fig, ax = plt.figure(figsize=(8, 3)), plt.gca()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=[model_summary[k] for k in list(model_summary.keys())],
                   colLabels=None,
                   rowLabels=list(model_summary.keys()),
                   cellLoc='center',
                   loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title("Model Efficiency Comparison")
    plt.tight_layout()
    plt.savefig('model_summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model_comparison(efficient_model, inefficient_model, test_loader, criterion):
    """Compare the performance metrics of both models on the test set"""
    
    # Dictionary to store all metrics
    results = {}
    
    # Evaluate both models
    for model_name, model in [('Efficient', efficient_model), ('Inefficient', inefficient_model)]:
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Evaluating {model_name} Model"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Measure inference time
                start_time = time.time()
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                end_time = time.time()
                inference_times.append(end_time - start_time)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                test_loss += loss.item() * inputs.size(0)
                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
        
        # Calculate metrics
        test_loss = test_loss / len(test_loader.dataset)
        avg_inference_time = np.mean(inference_times)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Precision, recall, F1
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Dice coefficient and IoU
        all_targets_array = np.array(all_targets)
        all_preds_array = np.array(all_preds)
        
        intersection = np.sum(all_targets_array * all_preds_array)
        dice = (2. * intersection) / (np.sum(all_targets_array) + np.sum(all_preds_array)) if (np.sum(all_targets_array) + np.sum(all_preds_array)) > 0 else 0
        union = np.sum(all_targets_array) + np.sum(all_preds_array) - intersection
        iou = intersection / union if union > 0 else 0
        
        # Store metrics
        results[model_name] = {
            'test_loss': test_loss,
            'avg_inference_time': avg_inference_time,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'dice': dice,
            'iou': iou,
            'confusion_matrix': cm,
            'all_probs': all_probs,
            'all_preds': all_preds,
            'all_targets': all_targets
        }
    
    # Create comparative visualizations
    
    # 1. Metrics comparison bar chart
    metrics_to_plot = ['precision', 'recall', 'f1', 'dice', 'iou']
    efficient_values = [results['Efficient'][m] for m in metrics_to_plot]
    inefficient_values = [results['Inefficient'][m] for m in metrics_to_plot]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    plt.bar(x - width/2, efficient_values, width, label='Efficient Model', color='blue', alpha=0.7)
    plt.bar(x + width/2, inefficient_values, width, label='Inefficient Model', color='red', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x, metrics_to_plot)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('performance_metrics_comparison.png', dpi=300)
    plt.close()
    
    # 2. Inference time comparison
    plt.figure(figsize=(8, 5))
    plt.bar(['Efficient Model', 'Inefficient Model'], 
           [results['Efficient']['avg_inference_time'], results['Inefficient']['avg_inference_time']],
           color=['blue', 'red'], alpha=0.7)
    
    plt.xlabel('Model')
    plt.ylabel('Average Inference Time (seconds)')
    plt.title('Inference Time Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('inference_time_comparison.png', dpi=300)
    plt.close()
    
    # 3. Side-by-side confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(results['Efficient']['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Change', 'Change'],
               yticklabels=['No Change', 'Change'], ax=ax1)
    ax1.set_title('Efficient Model Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    sns.heatmap(results['Inefficient']['confusion_matrix'], annot=True, fmt='d', cmap='Reds',
               xticklabels=['No Change', 'Change'],
               yticklabels=['No Change', 'Change'], ax=ax2)
    ax2.set_title('Inefficient Model Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_comparison.png', dpi=300)
    plt.close()
    
    # 4. ROC curves comparison
    plt.figure(figsize=(8, 6))
    
    # Calculate ROC for efficient model
    fpr_efficient, tpr_efficient, _ = roc_curve(results['Efficient']['all_targets'], results['Efficient']['all_probs'])
    roc_auc_efficient = auc(fpr_efficient, tpr_efficient)
    
    # Calculate ROC for inefficient model
    fpr_inefficient, tpr_inefficient, _ = roc_curve(results['Inefficient']['all_targets'], results['Inefficient']['all_probs'])
    roc_auc_inefficient = auc(fpr_inefficient, tpr_inefficient)
    
    # Plot ROC curves
    plt.plot(fpr_efficient, tpr_efficient, 'b-', 
             label=f'Efficient Model (AUC = {roc_auc_efficient:.3f})')
    plt.plot(fpr_inefficient, tpr_inefficient, 'r-', 
             label=f'Inefficient Model (AUC = {roc_auc_inefficient:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve_comparison.png', dpi=300)
    plt.close()
    
    # 5. Create a summary table for the paper
    metrics_summary = {
        'Metric': ['Test Loss', 'Precision', 'Recall', 'F1 Score', 'Dice Coefficient', 'IoU', 'Inference Time (s)'],
        'Efficient Model': [
            f"{results['Efficient']['test_loss']:.4f}",
            f"{results['Efficient']['precision']:.4f}",
            f"{results['Efficient']['recall']:.4f}",
            f"{results['Efficient']['f1']:.4f}",
            f"{results['Efficient']['dice']:.4f}",
            f"{results['Efficient']['iou']:.4f}",
            f"{results['Efficient']['avg_inference_time']*1000:.2f} ms"
        ],
        'Inefficient Model': [
            f"{results['Inefficient']['test_loss']:.4f}",
            f"{results['Inefficient']['precision']:.4f}",
            f"{results['Inefficient']['recall']:.4f}",
            f"{results['Inefficient']['f1']:.4f}",
            f"{results['Inefficient']['dice']:.4f}",
            f"{results['Inefficient']['iou']:.4f}",
            f"{results['Inefficient']['avg_inference_time']*1000:.2f} ms"
        ]
    }
    
    # Convert to visualization
    fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=[metrics_summary[k] for k in list(metrics_summary.keys())],
                   colLabels=None,
                   rowLabels=None,
                   cellLoc='center',
                   loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title("Performance Metrics Comparison Summary")
    plt.tight_layout()
    plt.savefig('performance_summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

# Visualize sample predictions from both models
def visualize_predictions(efficient_model, inefficient_model, test_loader, num_samples=4):
    """Visualize and compare predictions from both models on sample images"""
    efficient_model.eval()
    inefficient_model.eval()
    
    # Get a few samples
    sample_images = []
    sample_masks = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            sample_images.append(inputs)
            sample_masks.append(targets)
            if len(sample_images) >= num_samples:
                break
    
    # Create figure for visualization
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i in range(num_samples):
        inputs = sample_images[i].to(device)
        mask = sample_masks[i].to(device)
        
        # Get predictions from both models
        with torch.no_grad():
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                efficient_output = efficient_model(inputs)
                inefficient_output = inefficient_model(inputs)
        
        efficient_pred = (torch.sigmoid(efficient_output) > 0.5).float()
        inefficient_pred = (torch.sigmoid(inefficient_output) > 0.5).float()
        
        # Convert to numpy for visualization
        # Get RGB channels from time 1 (first 3 bands)
        rgb_t1 = inputs[0, :3].cpu().numpy()
        rgb_t1 = np.transpose(rgb_t1, (1, 2, 0))
        rgb_t1 = np.clip(rgb_t1, 0, 1)
        
        # Get mask and predictions
        true_mask = mask[0, 0].cpu().numpy()
        efficient_mask = efficient_pred[0, 0].cpu().numpy()
        inefficient_mask = inefficient_pred[0, 0].cpu().numpy()
        
        # Plot original image
        axes[i, 0].imshow(rgb_t1)
        axes[i, 0].set_title('Sentinel-2 Image')
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot efficient model prediction
        axes[i, 2].imshow(efficient_mask, cmap='gray')
        axes[i, 2].set_title('Efficient Model')
        axes[i, 2].axis('off')
        
        # Plot inefficient model prediction
        axes[i, 3].imshow(inefficient_mask, cmap='gray')
        axes[i, 3].set_title('Inefficient Model')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('model_predictions_comparison.png', dpi=300)
    plt.close()

# Function to create model architecture visualizations
def visualize_model_architectures():
    """Create visual representations of both model architectures"""
    
    # Create figure for AttentionUNet
    plt.figure(figsize=(12, 10))
    plt.title("Efficient Model: AttentionUNet Architecture", fontsize=16)
    
    # Simplified architecture visualization for AttentionUNet
    blocks = ['Input', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Bottleneck',
              'Dec4+Att4', 'Dec3+Att3', 'Dec2+Att2', 'Dec1+Att1', 'Output']
    
    # Define block positions (x, y, width, height)
    positions = {
        'Input': (1, 0, 1, 1),
        'Enc1': (0, 1, 1, 1),
        'Enc2': (0, 2, 1, 1),
        'Enc3': (1, 2, 1, 1),
        'Enc4': (2, 2, 1, 1),
        'Bottleneck': (3, 2, 1, 1),
        'Dec4+Att4': (3, 1, 1, 1),
        'Dec3+Att3': (2, 1, 1, 1),
        'Dec2+Att2': (2, 0, 1, 1),
        'Dec1+Att1': (3, 0, 1, 1),
        'Output': (2.5, -0.5, 1, 0.5)
    }
    
    # Define colors for different block types
    colors = {
        'Input': 'lightblue',
        'Enc1': 'lightgreen',
        'Enc2': 'lightgreen',
        'Enc3': 'lightgreen',
        'Enc4': 'lightgreen',
        'Bottleneck': 'gold',
        'Dec4+Att4': 'lightcoral',
        'Dec3+Att3': 'lightcoral',
        'Dec2+Att2': 'lightcoral',
        'Dec1+Att1': 'lightcoral',
        'Output': 'violet'
    }
    
    # Draw blocks
    for block in blocks:
        x, y, w, h = positions[block]
        rect = mpatches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black',
                               facecolor=colors[block], alpha=0.7)
        plt.gca().add_patch(rect)
        plt.text(x + w/2, y + h/2, block, ha='center', va='center', fontsize=10)
    
    # Draw connections
    connections = [
        ('Input', 'Enc1'), ('Enc1', 'Enc2'), ('Enc2', 'Enc3'), ('Enc3', 'Enc4'),
        ('Enc4', 'Bottleneck'), ('Bottleneck', 'Dec4+Att4'), ('Dec4+Att4', 'Dec3+Att3'),
        ('Dec3+Att3', 'Dec2+Att2'), ('Dec2+Att2', 'Dec1+Att1'), ('Dec1+Att1', 'Output')
    ]
    
    # Skip connections
    skip_connections = [
        ('Enc4', 'Dec4+Att4'), ('Enc3', 'Dec3+Att3'),
        ('Enc2', 'Dec2+Att2'), ('Enc1', 'Dec1+Att1')
    ]
    
    # Draw regular connections
    for start, end in connections:
        start_x, start_y, start_w, start_h = positions[start]
        end_x, end_y, end_w, end_h = positions[end]
        plt.arrow(start_x + start_w/2, start_y + start_h/2,
                 (end_x + end_w/2) - (start_x + start_w/2),
                 (end_y + end_h/2) - (start_y + start_h/2),
                 head_width=0.05, head_length=0.05, fc='black', ec='black', length_includes_head=True)
    
    # Draw skip connections with dashed lines
    for start, end in skip_connections:
        start_x, start_y, start_w, start_h = positions[start]
        end_x, end_y, end_w, end_h = positions[end]
        plt.plot([start_x + start_w, end_x],
                [start_y + start_h/2, end_y + end_h/2],
                'r--', linewidth=1)
    
    # Add annotation for attention mechanism
    plt.text(2.5, 1.5, "Attention\nGates", ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", fc='lightyellow', ec="black", lw=1))
    
    # Set plot limits
    plt.xlim(-0.5, 4.5)
    plt.ylim(-1, 3.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('efficient_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create figure for ComplexCNN
    plt.figure(figsize=(12, 10))
    plt.title("Inefficient Model: ComplexCNN Architecture", fontsize=16)
    
    # Simplified architecture visualization for ComplexCNN - use same layout but add more complexity
    blocks = ['Input', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Bottleneck',
              'Dec4', 'Dec3', 'Dec2', 'Dec1', 'Output']
    
    # Use same positions but different colors to show inefficiency
    colors = {
        'Input': 'lightblue',
        'Enc1': 'lightgreen',
        'Enc2': 'lightgreen',
        'Enc3': 'lightgreen',
        'Enc4': 'lightgreen',
        'Bottleneck': 'gold',
        'Dec4': 'salmon',
        'Dec3': 'salmon',
        'Dec2': 'salmon',
        'Dec1': 'salmon',
        'Output': 'violet'
    }
    
    # Modified positions to show more complexity
    positions = {
        'Input': (1, 0, 1, 1),
        'Enc1': (0, 1, 1, 1),
        'Enc2': (0, 2, 1, 1),
        'Enc3': (1, 2, 1, 1),
        'Enc4': (2, 2, 1, 1),
        'Bottleneck': (3, 2, 1, 1),
        'Dec4': (3, 1, 1, 1),
        'Dec3': (2, 1, 1, 1),
        'Dec2': (2, 0, 1, 1),
        'Dec1': (3, 0, 1, 1),
        'Output': (2.5, -0.5, 1, 0.5)
    }
    
    # Draw blocks
    for block in blocks:
        x, y, w, h = positions[block]
        rect = mpatches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black',
                               facecolor=colors[block], alpha=0.7)
        plt.gca().add_patch(rect)
        plt.text(x + w/2, y + h/2, block, ha='center', va='center', fontsize=10)
    
    # Draw connections
    connections = [
        ('Input', 'Enc1'), ('Enc1', 'Enc2'), ('Enc2', 'Enc3'), ('Enc3', 'Enc4'),
        ('Enc4', 'Bottleneck'), ('Bottleneck', 'Dec4'), ('Dec4', 'Dec3'),
        ('Dec3', 'Dec2'), ('Dec2', 'Dec1'), ('Dec1', 'Output')
    ]
    
    # Skip connections (same as in AttentionUNet but without attention)
    skip_connections = [
        ('Enc4', 'Dec4'), ('Enc3', 'Dec3'),
        ('Enc2', 'Dec2'), ('Enc1', 'Dec1')
    ]
    
    # Draw regular connections
    for start, end in connections:
        start_x, start_y, start_w, start_h = positions[start]
        end_x, end_y, end_w, end_h = positions[end]
        plt.arrow(start_x + start_w/2, start_y + start_h/2,
                 (end_x + end_w/2) - (start_x + start_w/2),
                 (end_y + end_h/2) - (start_y + start_h/2),
                 head_width=0.05, head_length=0.05, fc='black', ec='black', length_includes_head=True)
    
    # Draw skip connections with dashed lines
    for start, end in skip_connections:
        start_x, start_y, start_w, start_h = positions[start]
        end_x, end_y, end_w, end_h = positions[end]
        plt.plot([start_x + start_w, end_x],
                [start_y + start_h/2, end_y + end_h/2],
                'r--', linewidth=1)
    
    # Add annotations for inefficiency
    plt.text(0.5, 3, "Extra Conv\nLayers", ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", fc='mistyrose', ec="red", lw=1))
    plt.text(3.5, 3, "Redundant\nProcessing", ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", fc='mistyrose', ec="red", lw=1))
    plt.text(4, 1, "Inefficient\nUpsampling", ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", fc='mistyrose', ec="red", lw=1))
    
    # Set plot limits
    plt.xlim(-0.5, 4.5)
    plt.ylim(-1, 3.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('inefficient_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Dataset parameters
    DATA_DIR = "khumbu_change_dataset"  # Update this to your dataset directory
    BATCH_SIZE = 2
    NUM_WORKERS = 2
    
    # Training parameters
    NUM_EPOCHS = 10  # Reduced to 10 epochs as requested
    
    # Check if dataset files exist
    if not os.path.exists(DATA_DIR):
        print(f"Dataset directory '{DATA_DIR}' not found. Make sure it exists and contains the right data.")
        exit(1)
    
    # Create datasets and dataloaders
    train_dataset = GlacierChangeDataset(DATA_DIR, is_train=True)
    test_dataset = GlacierChangeDataset(DATA_DIR, is_train=False)
    
    # Check dataset samples
    print("Checking dataset samples...")
    check_dataset_values(train_dataset)
    
    # Create optimized dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Define loss function
    criterion = DiceBCEWithLogitsLoss()
    
    # Train and compare models
    print("Starting model comparison...")
    comparison_metrics = compare_models(train_loader, test_loader, criterion, epochs=NUM_EPOCHS)
    
    # Plot comparison results
    plot_model_comparison(comparison_metrics)
    
    # Evaluate models
    print("Evaluating model performance...")
    evaluation_results = evaluate_model_comparison(
        comparison_metrics['efficient_model'],
        comparison_metrics['inefficient_model'],
        test_loader,
        criterion
    )
    
    # Visualize sample predictions
    print("Visualizing sample predictions...")
    visualize_predictions(
        comparison_metrics['efficient_model'],
        comparison_metrics['inefficient_model'],
        test_loader
    )
    
    # Visualize model architectures
    print("Creating model architecture visualizations...")
    visualize_model_architectures()
    
    print("\n==== COMPARISON SUMMARY ====")
    # Calculate efficiency differences
    efficient_params = sum(p.numel() for p in comparison_metrics['efficient_model'].parameters())
    inefficient_params = sum(p.numel() for p in comparison_metrics['inefficient_model'].parameters())
    param_increase = (inefficient_params / efficient_params - 1) * 100
    
    avg_efficient_time = np.mean(comparison_metrics['efficient_times'])
    avg_inefficient_time = np.mean(comparison_metrics['inefficient_times'])
    time_increase = (avg_inefficient_time / avg_efficient_time - 1) * 100
    
    efficient_inf_time = evaluation_results['Efficient']['avg_inference_time'] * 1000  # ms
    inefficient_inf_time = evaluation_results['Inefficient']['avg_inference_time'] * 1000  # ms
    inf_time_increase = (inefficient_inf_time / efficient_inf_time - 1) * 100
    
    print(f"Parameter Count: Efficient={efficient_params:,} vs Inefficient={inefficient_params:,} (+{param_increase:.1f}%)")
    print(f"Training Time/Epoch: Efficient={avg_efficient_time:.2f}s vs Inefficient={avg_inefficient_time:.2f}s (+{time_increase:.1f}%)")
    print(f"Inference Time: Efficient={efficient_inf_time:.2f}ms vs Inefficient={inefficient_inf_time:.2f}ms (+{inf_time_increase:.1f}%)")
    print(f"Dice Coefficient: Efficient={evaluation_results['Efficient']['dice']:.4f} vs Inefficient={evaluation_results['Inefficient']['dice']:.4f}")
    print(f"IoU Score: Efficient={evaluation_results['Efficient']['iou']:.4f} vs Inefficient={evaluation_results['Inefficient']['iou']:.4f}")
    
    print("\nAll visualizations and model comparisons have been saved.")
    print("Files generated:")
    print("- model_loss_comparison.png: Training and validation loss curves")
    print("- model_time_comparison.png: Training time per epoch")
    print("- cumulative_time_comparison.png: Cumulative training time")
    print("- model_summary_comparison.png: Model efficiency metrics")
    print("- performance_metrics_comparison.png: Accuracy metrics")
    print("- inference_time_comparison.png: Inference time comparison")
    print("- confusion_matrix_comparison.png: Confusion matrices")
    print("- roc_curve_comparison.png: ROC curves")
    print("- performance_summary_comparison.png: Performance metrics table")
    print("- model_predictions_comparison.png: Sample predictions")
    print("- efficient_model_architecture.png: AttentionUNet architecture")
    print("- inefficient_model_architecture.png: ComplexCNN architecture")
    print("- efficient_model.pth: Trained efficient model checkpoint")
    print("- inefficient_model.pth: Trained inefficient model checkpoint")
    
    print("\nCompleted!")