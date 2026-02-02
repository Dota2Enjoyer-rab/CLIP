from transformers import CLIPModel, CLIPTokenizer
import torch.nn as nn

# Define architecture for CLIP model
class SAE(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def get_sae(input_dim, latent_dim):
    return SAE(latent_dim=latent_dim, input_dim=input_dim)