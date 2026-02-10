# utils.py
# Утилиты для сохранения логов и чекпоинтов модели

import os
import json
from datetime import datetime
import torch
from typing import Optional, Dict, Any


def save_logs(
    epoch: int,
    loss: float,
    log_dir: str = "./logs",
    extra_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Сохраняет метрики одной эпохи в jsonl-файл (дописывает в конец).
    """
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "training_logs.jsonl")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    entry = {
        "timestamp": timestamp,
        "epoch": epoch,
        "mse_loss": float(loss),
    }
    
    if extra_info:
        entry.update(extra_info)
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        print(f"Ошибка при записи лога: {e}")


def save_model(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    overwrite: bool = True
) -> str:
    """
    Сохраняет состояние модели и (опционально) оптимизатора.
    Возвращает фактический путь, по которому сохранена модель.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "last_loss": loss,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    save_path = path
    
    # Если не перезаписывать и файл уже существует → добавляем эпоху в имя
    if not overwrite and os.path.exists(path):
        base, ext = os.path.splitext(path)
        save_path = f"{base}_e{epoch if epoch is not None else 'final'}{ext}"
    
    try:
        torch.save(checkpoint, save_path)
        print(f"Модель сохранена: {save_path}")
    except Exception as e:
        print(f"Ошибка сохранения модели: {e}")
        save_path = path  # возвращаем исходный путь в случае ошибки
    
    return save_path