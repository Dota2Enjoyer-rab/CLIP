import warnings

# Подавляем FutureWarning и DeprecationWarning от MLflow и PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*artifact_path is deprecated.*")
warnings.filterwarnings("ignore", message=".*Saving pytorch model by Pickle.*")
warnings.filterwarnings("ignore", message=".*Found torch version.*local version label.*")
warnings.filterwarnings("ignore", message=".*The filesystem tracking backend.*")

from src.clip_architecture import get_sae
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import optuna
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from transformers import CLIPModel
from torch.amp import autocast, GradScaler
import os
import time
from torch.utils.data import random_split

# ──────────────────────────────────────────────────────────────
# MLflow config
MLFLOW_TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("CLIP_TopK_SAE_Tuning_4GB")

# Ускорение на CUDA
torch.backends.cudnn.benchmark = True


def train_one_run(params: dict, parent_run_id: str = None):
    with mlflow.start_run(nested=bool(parent_run_id)) as run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device} | Run ID: {run.info.run_id}")

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)

        # Логируем параметры после создания clip_model
        mlflow.log_params(params)
        mlflow.log_param("dataset", "CIFAR10")
        mlflow.log_param("epochs_total", params["epochs"])
        mlflow.log_param("input_dim", clip_model.config.projection_dim)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [test_size, train_size], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

        sae = get_sae(
            input_dim=clip_model.config.projection_dim,
            latent_dim=params["latent_dim"],
            k=params["k"]
        ).to(device)

        optimizer = torch.optim.AdamW(
            sae.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"]
        )

        criterion = nn.MSELoss()
        scaler = GradScaler(enabled=torch.cuda.is_available())

        os.makedirs("./checkpoints", exist_ok=True)
        best_loss = float("inf")
        best_epoch = 0

        for epoch in range(params["epochs"]):
            epoch_start = time.time()
            epoch_loss = 0.0
            n_batches = 0

            # Running average активаций z для dead feature detection
            z_sum = torch.zeros(params["latent_dim"], device=device)
            z_count = 0

            for imgs, _ in train_loader:
                imgs = imgs.to(device, non_blocking=True)

                with torch.no_grad():
                    vision_outputs = clip_model.vision_model(imgs, output_hidden_states=False)
                    pooled = (
                        vision_outputs.pooler_output
                        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None
                        else vision_outputs.last_hidden_state.mean(dim=1)
                    )
                    clip_features = clip_model.visual_projection(pooled)

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type='cuda', dtype=torch.float16):
                    x_hat, z, aux = sae(clip_features)
                    recon_loss = criterion(x_hat, clip_features)
                    loss = recon_loss + params["aux_coeff"] * aux

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Нормализация декодера
                with torch.no_grad():
                    sae.decoder.weight.div_(
                        sae.decoder.weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    )

                # Собираем статистику активаций z
                z_sum += z.mean(dim=0) * z.size(0)  # взвешенная сумма
                z_count += z.size(0)

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches if n_batches > 0 else float('inf')

            # Средние активации по эпохе
            z_avg = z_sum / z_count if z_count > 0 else torch.zeros_like(z_sum)

            # Порог для мёртвых фич
            dead_threshold = 1e-6
            dead_mask = (z_avg < dead_threshold)
            num_dead = dead_mask.sum().item()

            if num_dead > 0:
                print(f"Эпоха {epoch+1}: Перезапуск {num_dead} мёртвых фич...")
                with torch.no_grad():
                    dead_indices = torch.nonzero(dead_mask).squeeze()
                    sae.encoder.weight[dead_indices] = nn.init.kaiming_uniform_(
                        torch.empty_like(sae.encoder.weight[dead_indices])
                    )
                    sae.decoder.weight[:, dead_indices] = nn.init.kaiming_uniform_(
                        torch.empty_like(sae.decoder.weight[:, dead_indices])
                    )

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:2d}/{params['epochs']}   Loss: {avg_loss:.6f}   Time: {epoch_time:.1f}s")

            mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)
            mlflow.log_metric("current_epoch", epoch + 1, step=epoch + 1)
            mlflow.log_metric("dead_features_percent", num_dead / params["latent_dim"] * 100, step=epoch + 1)
            mlflow.log_metric("epoch_time_sec", epoch_time, step=epoch + 1)

            progress = (epoch + 1) / params["epochs"] * 100
            mlflow.log_metric("progress_percent", progress, step=epoch + 1)

            # Валидация после эпохи
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for imgs, _ in test_loader:
                    imgs = imgs.to(device, non_blocking=True)

                    vision_outputs = clip_model.vision_model(imgs, output_hidden_states=False)
                    pooled = (
                        vision_outputs.pooler_output
                        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None
                        else vision_outputs.last_hidden_state.mean(dim=1)
                    )
                    clip_features = clip_model.visual_projection(pooled)

                    x_hat, z, aux = sae(clip_features)
                    recon_loss = criterion(x_hat, clip_features)
                    val_loss += recon_loss.item()
                    val_batches += 1

            val_avg = val_loss / val_batches if val_batches > 0 else float('inf')
            print(f"Val Loss: {val_avg:.6f}")
            mlflow.log_metric("val_loss", val_avg, step=epoch + 1)


            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1

                checkpoint_path = (
                    f"./checkpoints/sae_k{params['k']}_lr{params['lr']:.1e}"
                    f"_wd{params['weight_decay']:.1e}_aux{params['aux_coeff']:.3f}"
                    f"_e{best_epoch}_loss{avg_loss:.5f}.pth"
                )

                torch.save({
                    "epoch": best_epoch,
                    "model_state_dict": sae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "params": params,
                }, checkpoint_path)

                mlflow.log_artifact(checkpoint_path, "best_checkpoint")
                mlflow.pytorch.log_model(
                    sae,
                    name="sae_model",
                )

        mlflow.log_metric("final_loss", avg_loss)
        mlflow.log_metric("best_loss", best_loss)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_param("completed_epochs", best_epoch)

        print(f"Завершено. Best loss: {best_loss:.6f} @ epoch {best_epoch}")
        return best_loss  


def objective(trial: optuna.Trial):
    params = {
        "lr": trial.suggest_float("lr", 3e-5, 8e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
        "aux_coeff": trial.suggest_float("aux_coeff", 0.001, 0.08, log=True),
        "k": trial.suggest_categorical("k", [8, 16, 24, 32, 48]),
        "latent_dim": trial.suggest_categorical("latent_dim", [2048, 4096]),
        "batch_size": 128,
        "epochs": 30,
    }

    final_loss = train_one_run(params)
    return final_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "tune"], default="single")
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "single":
        default_params = {
            "lr": 3e-4,
            "weight_decay": 1e-5,
            "aux_coeff": 0.05,
            "k": 48,
            "latent_dim": 4096,
            "batch_size": 128,
            "epochs": 30,
        }
        print("Запуск одиночного эксперимента...")
        train_one_run(default_params)

    else:
        print(f"Запуск Optuna тюнинга ({args.n_trials} trials)...")
        study = optuna.create_study(direction="minimize", study_name="CLIP_SAE_4GB")
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
        print("Лучшие параметры:", study.best_params)
        print("Лучшая loss:", study.best_value)