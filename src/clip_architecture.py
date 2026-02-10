import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder с auxiliary reconstruction loss.
    
    Рекомендуемые значения для CLIP ViT-B/32:
    · input_dim    = 512
    · latent_dim   = 2048–8192 (на 4 ГБ видеокарте лучше 1024–2048)
    · k            = 8–48 (чаще 16–32)
    """
    def __init__(self, input_dim: int, latent_dim: int, k: int = 32):
        super().__init__()
        self.k = k

        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)  # bias=True — важно!

        # Хорошая инициализация декодера (unit norm по выходам)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.encoder.weight)
            nn.init.kaiming_uniform_(self.decoder.weight)
            self.decoder.weight.div_(self.decoder.weight.norm(dim=1, keepdim=True).clamp(min=1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = self.encoder(x)  # [batch, latent_dim]
        values, indices = torch.topk(pre_acts, k=self.k, dim=-1)
        
        z = torch.zeros_like(pre_acts)
        z.scatter_(dim=-1, index=indices, src=values)
        
        return z, pre_acts  # возвращаем z и pre_acts для aux loss

    def forward(self, x: torch.Tensor):
        z, pre_acts = self.encode(x)
        x_hat = self.decoder(z)

        # Auxiliary reconstruction на всех предактивациях
        aux_loss = ((pre_acts - z.detach()) ** 2).mean()

        return x_hat, z, aux_loss

    def normalize_decoder(self):
        """Вызывать после каждого шага оптимизатора"""
        with torch.no_grad():
            norm = self.decoder.weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.div_(norm)


def get_sae(input_dim: int, latent_dim: int, k: int = 32):
    return TopKSAE(input_dim=input_dim, latent_dim=latent_dim, k=k)