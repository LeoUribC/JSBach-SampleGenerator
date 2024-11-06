import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from autoencoder import VAE

# Training parameters
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150

# Path to spectrograms (update this as needed)
SPECTROGRAMS_PATH = "/content/spectrograms"


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fsdd(spectrograms_path):
    x_train = []
    
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)

            if spectrogram.shape[-1] == 1:
                spectrogram = spectrogram[..., 0]
                #spectrogram = np.squeeze(spectrogram, axis=-1)

            x_train.append(spectrogram)
    
    x_train = np.array(x_train)
    #x_train = x_train[..., np.newaxis]  # Add channel dimension
    
    # Convert to PyTorch tensor
    x_train = torch.tensor(x_train, dtype=torch.float32)
    
    return x_train


def train(model, data_loader, learning_rate, epochs):
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to the specified device (GPU or CPU)
    model = model.to(device)
    
    model.train()  # Set model to training mode
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            x = batch[0].to(device)
            
            # Forward pass
            recon_x, mu, log_var = model(x)
            
            # Compute loss
            loss = model.loss_function(recon_x, x, mu, log_var)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print the average loss per epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")
    
    return model


if __name__ == "__main__":
    # Load data
    x_train = load_fsdd(SPECTROGRAMS_PATH)
    
    # Prepare data loader
    dataset = TensorDataset(x_train)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize VAE model
    autoencoder = VAE(
        input_shape=(1, 256, 848),  # Adjusted shape for PyTorch (channels first)
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )

    # Train model
    autoencoder = train(autoencoder, data_loader, LEARNING_RATE, EPOCHS)
    
    # Save model
    autoencoder.save_model("model")


