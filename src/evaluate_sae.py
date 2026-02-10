import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import explained_variance_score
import numpy as np
from src.clip_architecture import get_sae
from transformers import CLIPModel

# Защита для Windows + multiprocessing
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Загружаем CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # Трансформации (те же, что при обучении)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])

    # Test датасет
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,           # уменьшил до 128, чтобы точно влезло в 4 ГБ
        shuffle=False,
        num_workers=0,            # ← 0 на Windows — самый безопасный и стабильный вариант
        pin_memory=True
    )

    # Загружаем обученный SAE
    input_dim = clip_model.config.projection_dim  # 512
    latent_dim = 4096                             # подставь свой из чекпоинта
    k = 48                                        # подставь свой
    sae = get_sae(input_dim=input_dim, latent_dim=latent_dim, k=k).to(device)

    checkpoint_path = r"C:\Users\VN\Desktop\diploma\CLIP\checkpoints\sae_k48_lr3.0e-04_wd1.0e-05_aux0.050_e30_loss0.03400.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()

    print(f"Модель загружена с эпохи {checkpoint['epoch']}, loss = {checkpoint['loss']:.6f}")

    mse_total = 0
    cos_total = 0
    all_z = []
    all_orig = []
    all_recon = []

    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)

            # Получаем CLIP-фичи
            vision_outputs = clip_model.vision_model(imgs, output_hidden_states=False)
            pooled = (
                vision_outputs.pooler_output
                if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None
                else vision_outputs.last_hidden_state.mean(dim=1)
            )
            clip_features = clip_model.visual_projection(pooled)

            # Реконструкция SAE
            x_hat, z, _ = sae(clip_features)  # игнорируем aux

            mse_total += ((clip_features - x_hat)**2).mean().item()
            cos_total += torch.nn.functional.cosine_similarity(clip_features, x_hat, dim=-1).mean().item()

            all_z.append(z.cpu())
            all_orig.append(clip_features.cpu())
            all_recon.append(x_hat.cpu())

    mse = mse_total / len(test_loader)
    cos = cos_total / len(test_loader)

    all_z = torch.cat(all_z, dim=0)
    all_orig = torch.cat(all_orig, dim=0)
    all_recon = torch.cat(all_recon, dim=0)

    evr = explained_variance_score(all_orig.numpy(), all_recon.numpy())
    fvu = 1 - evr

    l0 = (all_z > 1e-6).float().sum(dim=1).mean().item()
    dead = (all_z.mean(dim=0) < 1e-8).float().mean().item() * 100

    print(f"MSE: {mse:.6f}")
    print(f"Cosine Similarity: {cos:.4f}")
    print(f"Explained Variance Ratio (EVR): {evr:.4f}")
    print(f"Fraction of Variance Unexplained (FVU): {fvu:.4f}")
    print(f"L0 (среднее число активных фич): {l0:.2f}")
    print(f"Dead features (%): {dead:.2f}%")