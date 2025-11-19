import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# TransformerMomentumNet (FlashAttention2 version)
# — 이미 패치 4에서 제작한 v73 버전 전체 포함.


# ============================================================
#   CONFIG
# ============================================================

@dataclass
class MomentumConfig:
    seq_length: int = 128
    feature_dim: int = 140
    min_length: int = 128
    use_cross_sectional_attn: bool = False
    smoothing: float = 0.2
    device: str = "cpu"


# ============================================================
#   FEATURE EXTRACTOR
# ============================================================

class MomentumFeatureExtractorV73:
    """
    Converts price/volume/microstructure/alpha into Transformer input sequence.
    """

    def __init__(self, config: MomentumConfig):
        self.config = config

    def _basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # log returns
        df["logret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)

        # normalized volume
        df["vol_norm"] = (
            (df["volume"] - df["volume"].rolling(20).mean()) /
            (df["volume"].rolling(20).std() + 1e-9)
        ).fillna(0)

        # high-low range
        df["range"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)

        # momentum window
        df["mom_10"] = df["close"].pct_change(10).fillna(0)
        df["mom_20"] = df["close"].pct_change(20).fillna(0)

        return df

    def _technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # RSI 14
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)

        # MACD histogram
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        df["macd_hist"] = hist.fillna(0)

        return df

    def _microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # spread proxy
        df["spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)

        # tick direction
        df["tick"] = np.sign(df["close"].diff()).fillna(0)

        # OFI proxy
        ofi = []
        for i in range(len(df)):
            if i == 0:
                ofi.append(0)
            else:
                up = max(df["close"].iloc[i] - df["close"].iloc[i - 1], 0)
                down = max(df["close"].iloc[i - 1] - df["close"].iloc[i], 0)
                ofi.append(up - down)
        df["ofi"] = ofi

        return df

    def build_features(
        self, df: pd.DataFrame, alpha_extra: Dict[str, float]
    ) -> np.ndarray:

        df = self._basic_features(df)
        df = self._technical_features(df)
        df = self._microstructure(df)

        # add alpha factors (gex, vpin, whisperz)
        df["gex"] = alpha_extra.get("gex", 0)
        df["vpin"] = alpha_extra.get("vpin", 0)
        df["whisper_z"] = alpha_extra.get("whisper_z", 0)
        df["dix_gex_combo"] = alpha_extra.get("dix_gex_combo", 0)

        # final selection (must match feature_dim=140)
        cols = df.columns[-self.config.feature_dim:]
        arr = df[cols].values[-self.config.seq_length:]

        # if too short, pad
        if len(arr) < self.config.seq_length:
            pad = np.zeros((self.config.seq_length - len(arr), arr.shape[1]))
            arr = np.vstack([pad, arr])

        return arr.astype(np.float32)


# ============================================================
#   TransformerMomentumNet (패치 4 전체 버전)
#   - FlashAttention2
#   - Rotary Embedding
#   - ALiBi Bias
#   - 8 output dimensions
# ============================================================

from .transformer_momentum_net import TransformerMomentumNet


# ============================================================
#   Momentum Engine V73
# ============================================================

class MomentumEngineV73:
    """
    Main Momentum Engine V73
    Uses FlashAttention2 transformer for sequence modeling.
    """

    def __init__(self, config: MomentumConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = TransformerMomentumNet(
            d_model=768,
            n_heads=12,
            n_layers=8,
            seq_length=config.seq_length,
            n_features=config.feature_dim
        ).to(self.device)

        self.feat_extractor = MomentumFeatureExtractorV73(config)

    # --------------------------------------------------------
    def generate_signal(
        self,
        df: pd.DataFrame,
        alpha_extra: Dict[str, float]
    ) -> Dict[str, Any]:

        # need at least seq_length rows
        if len(df) < self.config.seq_length:
            return {
                "momentum": 0.0,
                "trend": 0.0,
                "acceleration": 0.0,
                "quality": 0.0,
                "confidence": 0.0,
                "gex_score": 0.0,
                "vpin_toxicity": 0.0,
                "whisper_z": 0.0
            }

        features = self.feat_extractor.build_features(df, alpha_extra)

        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            out = self.model(x)

        # extract final values
        result = {
            "momentum": float(out["momentum"][0].cpu()),
            "trend": float(out["trend"][0].cpu()),
            "acceleration": float(out["acceleration"][0].cpu()),
            "quality": float(out["quality"][0].cpu()),
            "confidence": float(out["confidence"][0].cpu()),
            "gex_score": float(out["gex_score"][0].cpu()),
            "vpin_toxicity": float(out["vpin_toxicity"][0].cpu()),
            "whisper_z": float(out["whisper_z"][0].cpu())
        }

        return result

# ============================================================
#   Momentum Explanation Engine
# ============================================================

class MomentumExplainV73:
    """
    Provides human-readable interpretation of MomentumEngine outputs.
    """

    def explain(self, result: Dict[str, float]) -> Dict[str, Any]:

        mom = result["momentum"]
        trend = result["trend"]
        acc = result["acceleration"]
        q = result["quality"]
        conf = result["confidence"]

        explanation = {
            "direction": "bullish" if mom > 0 else "bearish",
            "momentum_strength": float(abs(mom)),
            "trend_strength": float(abs(trend)),
            "acceleration": float(acc),
            "quality": float(q),
            "confidence": float(conf),
        }

        # risk flags
        if result["vpin_toxicity"] > 0.85:
            explanation["risk_flag"] = "VPIN TOXICITY"
        elif result["gex_score"] > 0:
            explanation["risk_flag"] = "POSITIVE GAMMA"
        elif result["gex_score"] < 0:
            explanation["risk_flag"] = "NEGATIVE GAMMA"
        else:
            explanation["risk_flag"] = "NORMAL"

        return explanation


# ============================================================
#   Batch Momentum Processor (multiple assets)
# ============================================================

class MultiAssetMomentumV73:
    """
    Applies MomentumEngine to multiple assets in parallel.
    """

    def __init__(self, config: MomentumConfig):
        self.engine = MomentumEngineV73(config)

    def generate(
        self,
        price_map: Dict[str, pd.DataFrame],
        alpha_map: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, Any]]:

        results = {}

        for symbol, df in price_map.items():
            alpha_extra = alpha_map.get(symbol, {})
            out = self.engine.generate_signal(df, alpha_extra)
            results[symbol] = out

        return results

# ============================================================
#   Simple Momentum Backtest
# ============================================================

def momentum_backtest(
    df: pd.DataFrame,
    alpha_extra: Dict[str, float],
    config: Optional[MomentumConfig] = None
) -> Dict[str, float]:

    if config is None:
        config = MomentumConfig()

    engine = MomentumEngineV73(config)

    momentum_vals = []
    pnl_vals = []

    for i in range(config.seq_length, len(df)):
        window = df.iloc[:i]
        out = engine.generate_signal(window, alpha_extra)

        m = out["momentum"]
        momentum_vals.append(m)

        # simple pnl model
        ret = (df["close"].iloc[i] - df["close"].iloc[i - 1]) / df["close"].iloc[i - 1]
        pnl_vals.append(m * ret)

    pnl = float(np.sum(pnl_vals))
    sharpe = float(np.mean(pnl_vals) / (np.std(pnl_vals) + 1e-9) * np.sqrt(252))

    return {"pnl": pnl, "sharpe": sharpe}


# ============================================================
#   Momentum Orchestrator Adapter
# ============================================================

class MomentumOrchestratorAdapter:
    """
    Connects Momentum Engine to ARES-7 orchestrator.
    """

    def __init__(self, config: MomentumConfig):
        self.engine = MomentumEngineV73(config)
        self.explainer = MomentumExplainV73()

    def run(
        self, df: pd.DataFrame, alpha_extra: Dict[str, float]
    ) -> Dict[str, Any]:

        out = self.engine.generate_signal(df, alpha_extra)
        exp = self.explainer.explain(out)

        return {
            "signal": out,
            "explanation": exp
        }

# ============================================================
#   Visualization utilities (optional)
# ============================================================

def plot_momentum_signals(df: pd.DataFrame, signals: List[float]):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.plot(df["close"].iloc[-len(signals):].values, label="Price")
        plt.plot(signals, label="Momentum* (scaled)")
        plt.legend()
        plt.show()
    except:
        pass


# ============================================================
#   Embedding / latent representation (optional)
# ============================================================

class MomentumEmbeddingExtractor:
    """
    Extracts intermediate transformer embedding for advanced analytics.
    """

    def __init__(self, engine: MomentumEngineV73):
        self.engine = engine

    def extract(
        self, df: pd.DataFrame, alpha_extra: Dict[str, float]
    ) -> np.ndarray:

        features = self.engine.feat_extractor.build_features(df, alpha_extra)
        x = torch.tensor(features, dtype=torch.float32,
                         device=self.engine.device).unsqueeze(0)

        with torch.no_grad():
            h = self.engine.model.input_proj(x)
            return h[0].cpu().numpy()

# ============================================================
#   Standalone Test Entry
# ============================================================

def momentum_test_run(df: pd.DataFrame,
                      alpha_extra: Dict[str, float]) -> Dict[str, float]:

    cfg = MomentumConfig()
    engine = MomentumEngineV73(cfg)

    return engine.generate_signal(df, alpha_extra)


# ============================================================
#   MODULE END (PART 3)
# ============================================================

