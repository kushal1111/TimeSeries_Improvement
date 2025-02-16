import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Invertible import RevIN
from captum.attr import  IntegratedGradients, LayerConductance
import shap
from lime import lime_tabular
from torch.utils.tensorboard import SummaryWriter

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db1', level=1):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        x_transformed = []

        for b in range(batch_size):
            feature_transformed = []
            for d in range(num_features):
                coeffs = pywt.wavedec(x[b, :, d].cpu().numpy(), self.wavelet, level=self.level)
                feature_transformed.append(np.concatenate(coeffs))
            x_transformed.append(np.stack(feature_transformed, axis=1))

        x_transformed = np.stack(x_transformed, axis=0)
        return torch.tensor(x_transformed, dtype=x.dtype, device=x.device)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.task_name = args.task_name
        self.seq_len = args.seq_len
        self.args = args
        self.pred_len = args.pred_len   
        self.shap_explainer = None
        self.lime_explainer = None   
        self.writer = SummaryWriter(log_dir=f"./tensorboard_logs/{args.model}/model")

        self.time_dim = len(args.time_feature_types)
        self.attention_weights = None  # For storing attention

        # Initialize Wavelet Transform
        self.wavelet_transform = WaveletTransform(wavelet='db1', level=1)
        
        # XAI-specific components
        self.attn_probe = nn.Linear(args.c_out, 1)  # Attention probe
        self.grad_cam_conv = nn.Conv1d(args.c_out, 1, kernel_size=3, padding=1)
    
        self.histroy_proj = nn.Linear(self.seq_len, self.pred_len)
        self.time_proj = nn.Linear(self.seq_len, self.pred_len)
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

    def encoder(self, x, x_mark_enc, y_mark_dec):
        # x: [B, L, D]
        # Apply Wavelet Transform
        x = self.wavelet_transform(x)

        # Normalize the transformed data
        means = torch.mean(x, dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
        x = (x - means) / stdev

        time_embed = self.time_enc(x_mark_enc)
        # Add attention visualization
        attn_weights = F.softmax(self.attn_probe(time_embed), dim=1)
        self.attention_weights = attn_weights.detach().cpu().numpy()
        
        # Add Grad-CAM hooks
        if self.training:
            time_embed.register_hook(self._grad_cam_hook)
        time_out = self.time_proj(time_embed.transpose(1, 2)).transpose(1, 2)

        pred = self.histroy_proj(x.transpose(1, 2)).transpose(1, 2)
        pred = self.beta * pred + (1 - self.beta) * time_out

        pred = pred * stdev + means
        return pred

    def forecast(self, x, x_mark_enc, y_mark_dec):
        # Encoder
        return self.encoder(x, x_mark_enc, y_mark_dec)

    def forward(self, x, x_mark_enc, x_dec, y_mark_dec, mask=None):
        dec_out = self.forecast(x, x_mark_enc, y_mark_dec)
        # Log attention weights
        if self.attention_weights is not None:
            self.writer.add_image("Attention_Weights", self.attention_weights, global_step=0)

        # Log Grad-CAM
        grad_cam = self.get_grad_cam()
        self.writer.add_image("Grad_CAM", grad_cam, global_step=0)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    
    def _grad_cam_hook(self, grad):
        """For Grad-CAM visualization"""
        self.gradients = grad.detach().mean(dim=2, keepdim=True)
        
    def get_grad_cam(self):
        """Return Grad-CAM heatmap"""
        return (self.gradients * self.time_enc[-1].weight).sum(dim=1)
    
    def integrate_shap(self, data_loader):
        if self.shap_explainer is None:
            background = next(iter(data_loader))[0][:100]  # Use first 100 samples as background
            self.shap_explainer = shap.DeepExplainer(self, background)
        
        shap_values = []
        for batch_x, _, batch_x_mark, _ in data_loader:
            batch_shap_values = self.shap_explainer.shap_values(batch_x)
            shap_values.append(batch_shap_values)
        
        return shap_values

    def integrate_lime(self, data_loader):
        if self.lime_explainer is None:
            # Assuming the first element of the batch is the input data
            train_data = next(iter(data_loader))[0].numpy()
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                training_data=train_data,
                mode="regression",
                feature_names=[f"Feature_{i}" for i in range(train_data.shape[1])]
            )
        
        lime_explanations = []
        for batch_x, _, _, _ in data_loader:
            for single_x in batch_x:
                explanation = self.lime_explainer.explain_instance(
                    single_x.numpy(),
                    self.predict_single,
                    num_features=10
                )
                lime_explanations.append(explanation)
        
        return lime_explanations

    def predict_single(self, x):
        # Helper method for LIME
        x = torch.FloatTensor(x).unsqueeze(0)
        with torch.no_grad():
            return self.forward(x, None, None, None).squeeze().numpy()
