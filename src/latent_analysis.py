import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import pandas as pd
from PIL import Image
from src.clip_architecture import get_sae
from transformers import CLIPModel
import requests
import json
import time

# ──────────────────────────────────────────────────────────────
# Настройки
CHECKPOINT_PATH = r"C:\Users\VN\Desktop\diploma\CLIP\checkpoints\sae_k24_lr2.9e-04_wd3.8e-06_aux0.008_e25_loss0.01656.pth"
LATENT_DIM = 4096
K = 24
TOP_K = 8
MIN_MAX_ACT = 0.5          # ← порог: если max_act < 0.5 — латент мёртвый, пропускаем
OUTPUT_CSV = "live_latent_interpretations.csv"
SAVE_IMAGES_DIR = "top_activations_live"
os.makedirs(SAVE_IMAGES_DIR, exist_ok=True)

API_KEY = "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"  # твой ключ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Загрузка моделей
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
for p in clip_model.parameters():
    p.requires_grad_(False)

sae = get_sae(input_dim=clip_model.config.projection_dim, latent_dim=LATENT_DIM, k=K).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
sae.load_state_dict(checkpoint['model_state_dict'])
sae.eval()
print(f"Модель загружена с эпохи {checkpoint['epoch']}, loss = {checkpoint['loss']:.6f}")

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

# Сбор активаций и сохранение изображений
all_activations = []
all_images_paths = []

global_idx = 0
with torch.no_grad():
    for batch_idx, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device)
        vision_outputs = clip_model.vision_model(imgs, output_hidden_states=False)
        pooled = (
            vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else
            vision_outputs.last_hidden_state.mean(dim=1)
        )
        clip_features = clip_model.visual_projection(pooled)

        _, z, _ = sae(clip_features)
        all_activations.append(z.cpu())

        # Сохранение изображений (отмена нормализации)
        imgs_np = imgs.cpu().permute(0, 2, 3, 1).numpy()
        imgs_np = (imgs_np * np.array([0.26862954, 0.26130258, 0.27577711]) + np.array([0.48145466, 0.4578275, 0.40821073])).clip(0, 1) * 255
        imgs_np = imgs_np.astype(np.uint8)

        for i in range(imgs_np.shape[0]):
            path = f"{SAVE_IMAGES_DIR}/img_{global_idx}.png"
            Image.fromarray(imgs_np[i]).save(path)
            all_images_paths.append(path)
            global_idx += 1

all_activations = torch.cat(all_activations, dim=0)

# Сбор ТОЛЬКО живых латентов
live_latents = []
for latent_idx in range(LATENT_DIM):
    acts = all_activations[:, latent_idx]
    max_act = acts.max().item()

    if max_act < MIN_MAX_ACT:
        continue  # пропускаем мёртвые

    top_indices = torch.topk(acts, k=TOP_K).indices.numpy()
    top_paths = [all_images_paths[i] for i in top_indices]
    top_values = acts[top_indices].tolist()

    live_latents.append({
        "latent_id": latent_idx,
        "max_act": max_act,
        "mean_act": acts.mean().item(),
        "top_images": top_paths,
        "top_values": top_values
    })

print(f"Найдено {len(live_latents)} живых латентов (max_act >= {MIN_MAX_ACT}) из {LATENT_DIM}")

# Сохраняем базовую информацию о живых латентах
df_basic = pd.DataFrame(live_latents)
df_basic.to_csv("live_latents_basic.csv", index=False)
print("Базовая информация о живых латентах сохранена в live_latents_basic.csv")

# ──────────────────────────────────────────────────────────────
# Интерпретация через LLM только для живых
def interpret_latent(top_paths):
    prompt = """Ты — эксперт по интерпретируемости нейросетей.  
Вот 8 изображений CIFAR-10, на которых этот латент SAE максимально активировался.  
Опиши одной короткой фразой (на русском, 5–10 слов), какой визуальный паттерн, объект, цвет, текстура или концепт этот латент скорее всего кодирует.  
Не пиши лишнего, только одну фразу."""

    for path in top_paths:
        prompt += f"\n- {path}"

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "qwen/qwen2.5-7b-instruct:free",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 60,
                "temperature": 0.7
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Ошибка: {str(e)}"

# Интерпретация всех живых латентов
df = pd.DataFrame(live_latents)
df["interpretation"] = ""
for idx, row in df.iterrows():
    print(f"Интерпретирую latent {row['latent_id']} (max_act = {row['max_act']:.3f})...")
    desc = interpret_latent(row["top_images"])
    df.at[idx, "interpretation"] = desc
    print(f" → {desc}")
    time.sleep(1.5)  # пауза для лимита API

df.to_csv(OUTPUT_CSV, index=False)
print(f"Готово! Интерпретации сохранены в {OUTPUT_CSV}")
print(f"Всего живых латентов: {len(df)}")