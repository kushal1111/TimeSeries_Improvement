import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Invertible import RevIN

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.task_name = args.task_name
        self.seq_len = args.seq_len
        self.args = args
        self.pred_len = args.pred_len      

        self.time_dim = len(args.time_feature_types)
    
        self.histroy_proj = nn.Linear(self.seq_len, self.pred_len)
        self.time_proj = nn.Linear(self.seq_len, self.pred_len)
        self.fft_layer = nn.Linear(self.seq_len, args.c_out//4)
        self.freq_proj = nn.Linear(args.c_out//4, args.c_out)
        self.time_enc = nn.Sequential(
                                      nn.Linear(self.time_dim, args.c_out//args.rda), 
                                      nn.LayerNorm(args.c_out//args.rda),
                                      nn.ReLU(),
                                      nn.Linear(args.c_out//args.rda, args.c_out//args.rdb), 
                                      nn.LayerNorm(args.c_out//args.rdb),
                                      nn.ReLU(),
                                      nn.Conv1d(in_channels=self.seq_len, 
                                                out_channels=self.seq_len, 
                                                kernel_size=args.ksize, 
                                                padding='same'),
                                      nn.Linear(args.c_out//args.rdb, args.c_out),
                                      )
        
        self.beta = args.beta

    def fft_features(self, x):
        # Apply FFT and extract features
        fft = torch.fft.rfft(x, dim=1)
        magnitudes = torch.abs(fft)
        phases = torch.angle(fft)
        return torch.cat([magnitudes, phases], dim=-1)

    def encoder(self, x, x_mark_enc, y_mark_dec):
        # x: [B, L, D]
        means = torch.mean(x, dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
        x = (x - means) / stdev

        time_embed = self.time_enc(x_mark_enc)
        time_out = self.time_proj(time_embed.transpose(1, 2)).transpose(1, 2)

        pred = self.histroy_proj(x.transpose(1, 2)).transpose(1, 2)
        pred = self.beta * pred + (1 - self.beta) * time_out

        pred = pred * stdev + means
        # FFT Feature Extraction
        fft_feats = self.fft_features(x)
        fft_feats = F.relu(self.fft_layer(fft_feats))
        fft_feats = self.freq_proj(fft_feats)
        
        # Existing temporal processing
        time_embed = self.time_enc(x_mark_enc)
        time_out = self.time_proj(time_embed.transpose(1, 2)).transpose(1, 2)
        
        # Combine features
        combined = self.beta * (pred + fft_feats) + (1 - self.beta) * time_out
        return combined * stdev + means

    def forecast(self, x, x_mark_enc, y_mark_dec):
        # Encoder
        return self.encoder(x, x_mark_enc, y_mark_dec)

    def forward(self, x, x_mark_enc, x_dec, y_mark_dec, mask=None):
        dec_out = self.forecast(x, x_mark_enc, y_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
