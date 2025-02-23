import torch
from torch import nn
import torch.nn.functional as F
import math
from layers.Invertible import RevIN
from layers.ModernTCN_Layer import series_decomp, Flatten_Head
import matplotlib.pyplot as plt
import seaborn as sns

class FourierLayer(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.fft_conv = nn.Conv1d(2, 64, kernel_size=3, padding=1)  # Magnitude+Phase
        
    def forward(self, x):
        fft = torch.fft.rfft(x, dim=-1)
        mag_phase = torch.cat([fft.abs(), fft.angle()], dim=1)
        return self.fft_conv(mag_phase)
    
    def visualize_attention(self, attention_map, filename):
        plt.figure(figsize=(10,5))
        sns.heatmap(attention_map.cpu().numpy(), cmap='viridis')
        plt.savefig(filename)
        plt.close()
