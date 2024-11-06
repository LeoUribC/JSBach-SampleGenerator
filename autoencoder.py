import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle
import numpy as np


class VAE(nn.Module):

    """
    VAE represents a Deep Convolutional VAE architecture with
    mirrored encoder and decoder components.
    """

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        super(VAE, self).__init__()
        
        # Model parameters
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = 1000000

        # Encoder
        self.encoder = self._build_encoder()

        # Determine shape before bottleneck by passing a dummy input
        self._shape_before_bottleneck = self._get_shape_before_bottleneck()

        # Latent space sampling layers
        self.mu = nn.Linear(self._shape_before_bottleneck, latent_space_dim)
        self.log_var = nn.Linear(self._shape_before_bottleneck, latent_space_dim)

        # Decoder
        self.decoder = self._build_decoder()


    def _build_encoder(self):
        layers = []
        in_channels = self.input_shape[0]
        for filters, kernel, stride in zip(self.conv_filters, self.conv_kernels, self.conv_strides):
            layers.append(nn.Conv2d(in_channels, filters, kernel_size=kernel, stride=stride, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(filters))
            in_channels = filters
        return nn.Sequential(*layers)


    # def _get_shape_before_bottleneck(self):
    #     # Pass a dummy input through the encoder to determine the shape
    #     with torch.no_grad():
    #         dummy_input = torch.zeros(1, *self.input_shape)  # Shape: (1, channels, height, width)
    #         output = self.encoder(dummy_input)
    #         return int(np.prod(output.size()[1:]))  # Flatten the output shape


    def _get_shape_before_bottleneck(self):
        # Pass a dummy input through the encoder to determine the 3D shape before flattening
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)  # Shape: (1, channels, height, width)
            output = self.encoder(dummy_input)
            return output.shape[1:]  # Return (channels, height, width)


    def _build_decoder(self):
        layers = []
        in_channels = self.conv_filters[-1]
        for filters, kernel, stride in zip(reversed(self.conv_filters), reversed(self.conv_kernels), reversed(self.conv_strides)):
            layers.append(nn.ConvTranspose2d(in_channels, filters, kernel_size=kernel, stride=stride, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(filters))
            in_channels = filters
        layers.append(nn.ConvTranspose2d(in_channels, self.input_shape[0], kernel_size=self.conv_kernels[0], stride=self.conv_strides[0], padding=1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)


    # def forward(self, x):
    #     # Encoding
    #     x = self.encoder(x)
    #     x = torch.flatten(x, start_dim=1)
    #     mu = self.mu(x)
    #     log_var = self.log_var(x)
    #     z = self.reparameterize(mu, log_var)
        
    #     # Decoding
    #     x_recon = self.decoder(z.view(-1, self.conv_filters[-1], 1, 1))
    #     return x_recon, mu, log_var


    def forward(self, x):
        # Encoding
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self.reparameterize(mu, log_var)
        
        # Decoding
        # Reshape z to match the input size expected by the decoder
        z = z.view(-1, self.conv_filters[-1], self._shape_before_bottleneck[1], self._shape_before_bottleneck[2])
        x_recon = self.decoder(z)
        
        return x_recon, mu, log_var


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


    def loss_function(self, recon_x, x, mu, log_var):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return self.reconstruction_loss_weight * recon_loss + kl_loss


    # # Ensure loss function expects consistent shapes
    # def loss_function(self, recon_x, x, mu, log_var):
    #     # Reconstruction loss
    #     recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    #     # KL divergence loss
    #     kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    #     return self.reconstruction_loss_weight * recon_loss + kl_loss


    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, "vae_weights.pth"))


    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "vae_weights.pth")))



# # Test VAE initialization with sample parameters
# vae = VAE( input_shape=(1, 28, 28),
#             conv_filters=(32, 64, 64, 64),
#             conv_kernels=[3, 3, 3, 3],
#             conv_strides=[1, 2, 2, 1],
#             latent_space_dim=2 )

# # Print summary of model structure to verify
# print(vae)
