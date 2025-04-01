import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

# Define a slightly smaller U-Net with Attention model to save memory
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=16, out_channels=1):
        super(AttentionUNet, self).__init__()
        
        # Reduced filter sizes to save memory
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

# Function to train the model with all optimizations
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=50, patience=10, accumulation_steps=4):
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    
    # For mixed precision training
    scaler = GradScaler()
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Use autocast for mixed precision
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets) / accumulation_steps
            
            # Scale the loss and do backward pass
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * inputs.size(0) * accumulation_steps
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Free up memory
        empty_cuda_cache()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / accumulation_steps
                
                val_loss += loss.item() * inputs.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
        
        # Print epoch stats
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            patience_counter = 0
            
            # Save the best model
            torch.save({
                'model_state_dict': best_model_weights,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'best_glacier_change_model.pth')
            
            print(f"New best model saved with val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Free up memory again
        empty_cuda_cache()
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.close()
    
    return model, train_losses, val_losses

# Function to evaluate the model and visualize results
def evaluate_model(model, test_loader, criterion, num_samples=5):
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    sample_inputs = []
    sample_outputs = []
    sample_targets = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            test_loss += loss.item() * inputs.size(0)
            
            # Convert predictions to binary (0 or 1)
            outputs_sigmoid = torch.sigmoid(outputs)
            preds = (outputs_sigmoid > 0.5).float()
            
            # Store predictions and targets for metrics calculation
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Store some samples for visualization
            if i < num_samples:
                sample_inputs.append(inputs.cpu().numpy())
                sample_outputs.append(outputs.cpu().numpy())
                sample_targets.append(targets.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Calculate metrics: dice coefficient and IoU
    dice_sum = 0
    iou_sum = 0
    n_samples = len(all_predictions)
    
    for pred, target in zip(all_predictions, all_targets):
        pred = pred.flatten()
        target = target.flatten()
        
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        
        # Avoid division by zero
        if (np.sum(pred) + np.sum(target)) > 0:
            dice = (2. * intersection) / (np.sum(pred) + np.sum(target))
        else:
            dice = 1.0  # Perfect dice when both pred and target are empty
            
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0  # Perfect IoU when both pred and target are empty
        
        dice_sum += dice
        iou_sum += iou
    
    avg_dice = dice_sum / n_samples
    avg_iou = iou_sum / n_samples
    
    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    
    # Visualize some results
    plt.figure(figsize=(15, 10))
    
    for i in range(min(num_samples, len(sample_inputs))):
        # Get RGB from time 1 (assuming first 3 bands are RGB)
        rgb_t1 = sample_inputs[i][0, :3]
        rgb_t1 = np.transpose(rgb_t1, (1, 2, 0))
        
        # Get RGB from time 2 (assuming next 3 bands are RGB)
        rgb_t2 = sample_inputs[i][0, 4:7]
        rgb_t2 = np.transpose(rgb_t2, (1, 2, 0))
        
        # Get prediction and target
        pred = sample_outputs[i][0, 0]
        target = sample_targets[i][0, 0]
        
        # Create change overlay on RGB time 2
        rgb_overlay = rgb_t2.copy()
        rgb_overlay[:, :, 0] = np.where(pred > 0.5, 1.0, rgb_overlay[:, :, 0])
        
        # Plot
        plt.subplot(num_samples, 4, i*4 + 1)
        plt.imshow(np.clip(rgb_t1, 0, 1))
        plt.title(f"RGB (t1) - Sample {i+1}")
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i*4 + 2)
        plt.imshow(np.clip(rgb_t2, 0, 1))
        plt.title(f"RGB (t2) - Sample {i+1}")
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i*4 + 3)
        plt.imshow(pred > 0.5, cmap='gray')
        plt.title(f"Prediction - Sample {i+1}")
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i*4 + 4)
        plt.imshow(target, cmap='gray')
        plt.title(f"Ground Truth - Sample {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()
    
    return test_loss, avg_dice, avg_iou

# Main execution
if __name__ == "__main__":
    # Dataset parameters
    DATA_DIR = "khumbu_change_dataset"
    BATCH_SIZE = 2
    NUM_WORKERS = 2
    
    # Model parameters
    IN_CHANNELS = 16  # 8 bands from optical (RGB+NIR from both dates) + 4 SAR + 2 NDSI + 2 NDWI
    OUT_CHANNELS = 1  # Binary segmentation (change/no-change)
    
    # Training parameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    PATIENCE = 15  # For early stopping
    ACCUMULATION_STEPS = 4  # Gradient accumulation steps
    
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
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Create the model (using smaller architecture)
    model = AttentionUNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device)
    
    # Define loss function and optimizer
    criterion = DiceBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    trained_model, train_losses, val_losses = train_model(
        model, 
        train_loader, 
        test_loader,  # Using test set as validation since our dataset is small
        criterion, 
        optimizer,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        accumulation_steps=ACCUMULATION_STEPS
    )
    
    # Evaluate the model
    test_loss, dice_coef, iou = evaluate_model(trained_model, test_loader, criterion)
    
    print("Training and evaluation complete!")
    print(f"Test Loss: {test_loss:.4f}, Dice Coefficient: {dice_coef:.4f}, IoU: {iou:.4f}")