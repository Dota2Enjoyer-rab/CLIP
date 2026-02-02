from transformers import CLIPModel, CLIPTokenizer
import torch.nn as nn
class Clip(nn.Module):
    def __init__(self):
        super(self).__init__()
