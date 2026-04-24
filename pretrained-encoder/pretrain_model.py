import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.rq_vae import ResidualVQ


class IMUEncoder(nn.Module):
    """
    Input:
        raw:   [B, D, T, F]
        patch: [B, D, P, L, F]
    Output:
        token features [B, S, C]
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 256, n_layers: int = 4, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            # [B, D, P, L, F] -> [B, D*P*L, F]
            b, d, p, l, f = x.shape
            x = x.reshape(b, d * p * l, f)
        elif x.dim() == 4:
            # [B, D, T, F] -> [B, D*T, F]
            b, d, t, f = x.shape
            x = x.reshape(b, d * t, f)
        else:
            raise ValueError(f"Unexpected IMU shape: {x.shape}")

        x = self.in_proj(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x


class IMUDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, output_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 1024, hidden_dim: int = 256, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, text: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(text)
        key_padding_mask = mask.eq(0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        valid = mask.unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return pooled


class IMURQVAEAlignModel(nn.Module):
    """
    Joint model for:
      1) IMU reconstruction via Residual VQ
      2) IMU-text alignment via contrastive learning
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.temperature = args.temperature

        self.imu_encoder = IMUEncoder(
            input_dim=args.imu_input_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.imu_encoder_layers,
            n_heads=args.imu_encoder_heads,
            dropout=args.dropout,
        )
        self.rq_vae = ResidualVQ(
            num_embeddings=args.rq_num_embeddings,
            embedding_dim=args.hidden_dim,
            num_quantizers=args.rq_num_quantizers,
        )
        self.imu_decoder = IMUDecoder(hidden_dim=args.hidden_dim, output_dim=args.imu_input_dim)
        self.text_encoder = TextEncoder(
            vocab_size=args.text_vocab_size,
            hidden_dim=args.hidden_dim,
            n_layers=args.text_encoder_layers,
            n_heads=args.text_encoder_heads,
            dropout=args.dropout,
        )
        self.imu_proj = nn.Linear(args.hidden_dim, args.proj_dim)
        self.text_proj = nn.Linear(args.hidden_dim, args.proj_dim)

    def encode_imu(self, imu_x: torch.Tensor) -> torch.Tensor:
        return self.imu_encoder(imu_x)

    def forward(
        self,
        imu_x: torch.Tensor,
        text: torch.Tensor,
        mask: torch.Tensor,
        compute_align: bool = True,
    ) -> Dict[str, torch.Tensor]:
        token_feats = self.encode_imu(imu_x)  # [B, S, C]

        # RQ-VAE expects [B, C, S]
        token_feats_bcs = token_feats.transpose(1, 2).contiguous() 
        quantized_bcs, vq_loss, code_indices = self.rq_vae(token_feats_bcs)
        quantized = quantized_bcs.transpose(1, 2).contiguous()  # [B, S, C]

        recon = self.imu_decoder(quantized)  # [B, S, F]
        target = self._flatten_imu(imu_x)    # [B, S, F]
        recon_loss = F.mse_loss(recon, target)

        imu_z = torch.empty(0, device=imu_x.device)
        text_z = torch.empty(0, device=imu_x.device)
        align_loss = torch.zeros((), device=imu_x.device)
        if compute_align:
            imu_global = quantized.mean(dim=1)           # [B, C]
            text_global = self.text_encoder(text, mask)  # [B, C]
            imu_z = F.normalize(self.imu_proj(imu_global), dim=-1)
            text_z = F.normalize(self.text_proj(text_global), dim=-1)
            align_loss = self._contrastive_loss(imu_z, text_z, self.temperature)

        return {
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "align_loss": align_loss,
            "imu_z": imu_z,
            "text_z": text_z,
            "codes": code_indices,
        }

    @staticmethod
    def _flatten_imu(imu_x: torch.Tensor) -> torch.Tensor:
        if imu_x.dim() == 5:
            b, d, p, l, f = imu_x.shape
            return imu_x.reshape(b, d * p * l, f)
        if imu_x.dim() == 4:
            b, d, t, f = imu_x.shape
            return imu_x.reshape(b, d * t, f)
        raise ValueError(f"Unexpected IMU shape: {imu_x.shape}")

    @staticmethod
    def _contrastive_loss(imu_z: torch.Tensor, text_z: torch.Tensor, temperature: float) -> torch.Tensor:
        logits = imu_z @ text_z.t() / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_i2t + loss_t2i)
