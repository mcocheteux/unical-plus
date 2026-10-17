"""
Temporal fusion: aggregates a window of per-frame backbone features into a
single feature vector before the regression head.

The decalibration being corrected is constant across a short window of
consecutive frames (it is a property of the sensor mounts, not of any one
observation), so fusing multiple frames' features gives the head several
noisy observations of the same target to average over.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalFusion(nn.Module):
    """
    Aggregates a (B, T, D) feature stack into a (B, D) vector.

    Args:
        feature_dim:          Size of the per-frame feature vector (must match
                               the backbone's GAP output / head's in_features).
        fusion_type:           "none" | "gru" | "transformer".
                               "none" is a parameter-free mean over T and is the
                               backward-compatible default: at T=1 it is exactly
                               the identity, so models trained without a window
                               are unaffected.
        gru_hidden:            Hidden size of the GRU (only used if fusion_type="gru").
        gru_layers:            Number of GRU layers.
        bidirectional:         Whether the GRU is bidirectional.
        transformer_layers:    Number of Transformer encoder layers (only used if
                               fusion_type="transformer").
        transformer_heads:     Number of attention heads.
        transformer_ff_dim:    Feed-forward dimension inside each encoder layer.
        dropout:               Dropout used inside the GRU/Transformer.
        max_seq_len:           Maximum window length supported by the learned
                               positional embedding (transformer only).
    """

    def __init__(
        self,
        feature_dim: int = 640,
        fusion_type: str = "none",
        gru_hidden: int = 640,
        gru_layers: int = 1,
        bidirectional: bool = False,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 16,
    ) -> None:
        super().__init__()
        if fusion_type not in ("none", "gru", "transformer"):
            raise ValueError(f"Unknown fusion_type: {fusion_type!r}")
        self.fusion_type = fusion_type
        self.feature_dim = feature_dim

        if fusion_type == "gru":
            self.gru = nn.GRU(
                input_size=feature_dim,
                hidden_size=gru_hidden,
                num_layers=gru_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if gru_layers > 1 else 0.0,
            )
            gru_out_dim = gru_hidden * (2 if bidirectional else 1)
            self.proj = (
                nn.Identity() if gru_out_dim == feature_dim else nn.Linear(gru_out_dim, feature_dim)
            )
        elif fusion_type == "transformer":
            self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, feature_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, feature_dim)

        Returns:
            (B, feature_dim)
        """
        if self.fusion_type == "none":
            return x.mean(dim=1)

        if self.fusion_type == "gru":
            _, h_n = self.gru(x)
            # h_n: (num_layers * num_directions, B, gru_hidden)
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.gru.bidirectional else h_n[-1]
            return self.proj(h_last)

        # transformer
        T = x.shape[1]
        if T > self.pos_embedding.shape[1]:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.pos_embedding.shape[1]}"
            )
        x = x + self.pos_embedding[:, :T]
        out = self.encoder(x)
        return out.mean(dim=1)


__all__ = ["TemporalFusion"]
