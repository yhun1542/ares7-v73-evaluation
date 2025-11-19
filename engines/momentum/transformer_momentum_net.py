import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from rotary_embedding_torch import RotaryEmbedding
import math

class TransformerMomentumNet(nn.Module):
    """
    ARES-7 v73 FlashAttention2 Momentum Transformer
    ------------------------------------------------
    특징:
    - FlashAttention-2 기반 초고속 Attention
    - Rotary Position Embedding (RoPE)
    - ALiBi positional bias
    - Transformer Encoder Layer 다중 스택
    - 모멘텀 / 추세 / 가속도 / 품질 / 신뢰도 / GEX / VPIN / WhisperZ 출력
    """
    def __init__(
        self,
        d_model=768,
        n_heads=12,
        n_layers=8,
        seq_length=128,
        n_features=140
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_length = seq_length
        self.head_dim = d_model // n_heads

        # 입력 선형 변환
        self.input_proj = nn.Linear(n_features, d_model)

        # Rotary embedding
        self.rotary = RotaryEmbedding(dim=self.head_dim)

        # ALiBi slopes
        self.alibi_slopes = self._build_alibi_slopes(n_heads)

        # Transformer 블록 구성
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=3072,
                dropout=0.10,
                batch_first=True,
                activation="gelu",
                norm_first=True
            )
            for _ in range(n_layers)
        ])

        # 출력 헤드: 8개 아웃풋
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 8)  # output channels
        )

    # ----------------------------------------------------
    # ALiBi bias 계산
    # ----------------------------------------------------
    def _build_alibi_slopes(self, n_heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n_heads).is_integer():
            slopes = get_slopes_power_of_2(n_heads)
        else:
            closest_power = 2 ** math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power)
            slopes = slopes + slopes[0::2][: n_heads - closest_power]
        return torch.tensor(slopes)

    # ----------------------------------------------------
    def _alibi_bias(self, seq_len, device):
        i = torch.arange(seq_len, device=device)
        j = torch.arange(seq_len, device=device)
        bias = i.view(-1, 1) - j.view(1, -1)
        bias = bias.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
        bias = bias * self.alibi_slopes.view(1, self.n_heads, 1, 1)
        return bias

    # ----------------------------------------------------
    # Forward
    # ----------------------------------------------------
    def forward(self, x):
        """
        x: (batch, seq_length, n_features)
        """
        B, L, _ = x.size()
        device = x.device

        # 1) Feature projection
        x = self.input_proj(x)  # (B, L, d_model)

        # 2) Rotary embedding 적용
        x = self.rotary.rotate_queries_or_keys(x)

        # 3) ALiBi bias
        alibi = self._alibi_bias(L, device)

        # 4) Transformer 레이어 반복
        for layer in self.layers:
            # q,k,v 분리
            qkv = x.view(B, L, self.n_heads, self.head_dim)
            q = qkv
            k = qkv
            v = qkv

            # FlashAttention-2 실행
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=0.0 if not self.training else 0.1,
                softmax_scale=None,
                attention_bias=alibi
            )
            attn_output = attn_output.reshape(B, L, self.d_model)

            # Transformer layer 통과
            x = layer(x + attn_output)

        # 5) Sequence pooling
        pooled = x.mean(dim=1)

        # 6) Output head
        out = self.head(pooled)

        return {
            "momentum": torch.tanh(out[:, 0]),
            "trend": torch.tanh(out[:, 1]),
            "acceleration": out[:, 2],
            "quality": torch.sigmoid(out[:, 3]),
            "confidence": torch.sigmoid(out[:, 4]),
            "gex_score": out[:, 5],
            "vpin_toxicity": torch.sigmoid(out[:, 6]),
            "whisper_z": out[:, 7]
        }
