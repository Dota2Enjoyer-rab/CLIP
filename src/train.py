from src.clip_architecture import get_sae
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import torch
import transformers
from torchvision import transforms, datasets
from torch.nn.utils import DataLoader
from src.utils import save_logs, save_model

# Define training function for SAE with CLIP features
def train_sae(epochs):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Dataset

    transform = transformers.Compose([transformers.Resize(224), transforms.ToTensor()])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # SAE model
    input_dim = clip_model.config.projection_dim
    latent_dim = 128
    sae = get_sae(input_dim=input_dim, latent_dim=latent_dim).to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for imgs, _ in loader:
            imgs = imgs.to(device)
            with torch.no_grad():
                clip_features = clip_model.get_image_features(imgs)
            x_hat, z = sae(clip_features)
            loss = criterion(x_hat, clip_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            save_logs(epoch, loss.item())