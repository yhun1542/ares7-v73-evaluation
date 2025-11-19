import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional

from dataclasses import dataclass

import torch
import torch.nn as nn

from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor

# LLM Alpha integration
from ..llm_alpha.llm_feature_builder import LLMFeatureBuilder


# ============================================================
#   ARES-7 v73 Ensemble Meta Engine
# ============================================================
#  - Combines all engines into unified final signal
#  - CatBoost + TabNet + Phoenix + Momentum + MeanRev
#  - LLM Alpha integration
#  - WhisperZ, GEX, VPIN included
# ============================================================


@dataclass
class MetaConfig:
    llm_provider: str = "openai"
    smoothing: float = 0.15
    feature_dropout: float = 0.05
    device: str = "cpu"


class FeatureNormalizer:
    """
    Normalizes feature vectors for ML models.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-9

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return X
        return (X - self.mean) / self.std


# ============================================================
#   META ENGINE
# ============================================================

class EnsembleMetaEngineV73:
    """
    ARES-7 v73 Meta-Ensemble Engine.
    Combines outputs from:
      - Phoenix Engine
      - Momentum Engine
      - Mean-Reversion Engine
      - CatBoost Regresor
      - TabNet Regressor
      - LLM Alpha Features
      - GEX / VPIN / WhisperZ / DIX-GEX Combo

    Produces:
      - final_signal ∈ [-1, 1]
      - confidence
      - risk_score
    """

    def __init__(self, config: MetaConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # ML Models
        self.cb = CatBoostRegressor(
            iterations=700,
            depth=6,
            learning_rate=0.04,
            verbose=False,
            loss_function='MAE'
        )

        self.tabnet = TabNetRegressor()

        # Feature utilities
        self.normalizer = FeatureNormalizer()

        # LLM alpha module
        self.llm = LLMFeatureBuilder(provider=config.llm_provider)

        # Adaptive weights
        self.weights = {
            "momentum": 0.33,
            "phoenix": 0.27,
            "meanrev": 0.15,
            "catboost": 0.15,
            "tabnet": 0.10
        }

        self.fitted = False

    # --------------------------------------------------------
    # Build feature vector for ML models
    # --------------------------------------------------------
    def _build_feature_vector(
        self,
        phoenix_out: float,
        momentum_out: float,
        meanrev_out: float,
        alpha: Dict[str, Any]
    ) -> np.ndarray:

        base_feats = [
            phoenix_out,
            momentum_out,
            meanrev_out,

            # key alpha factors
            alpha.get("whisper_z", 0),
            alpha.get("gex", 0),
            alpha.get("vpin", 0),
            alpha.get("dix_gex_combo", 0),
            alpha.get("earnings_factor", 0),
            alpha.get("risk_scalar", 0),
            alpha.get("sentiment_score", 0),
            alpha.get("impact_score", 0),
            alpha.get("vol_shock", 0),
            alpha.get("uncertainty", 0),
        ]

        embed = alpha.get("global_embedding", [])
        feats = np.array(base_feats + embed, dtype=float)
        return feats

    # --------------------------------------------------------
    # Core predict method (ensemble blend)
    # --------------------------------------------------------
    def predict(
        self,
        *,
        phoenix_out: float,
        momentum_out: float,
        meanrev_out: float,
        alpha_features: Dict[str, Any]
    ) -> Dict[str, Any]:

        # 1) feature vector
        X = self._build_feature_vector(
            phoenix_out,
            momentum_out,
            meanrev_out,
            alpha_features
        )

        # normalization
        Xn = self.normalizer.transform(X)

        Xn = Xn.reshape(1, -1)

        # 2) CatBoost prediction
        try:
            cb_pred = float(self.cb.predict(Xn)[0])
        except Exception:
            cb_pred = 0.0

        # 3) TabNet prediction
        try:
            tn_pred = float(self.tabnet.predict(Xn)[0][0])
        except Exception:
            tn_pred = 0.0

        # 4) weighted ensemble
        final_raw = (
            self.weights["momentum"] * momentum_out +
            self.weights["phoenix"] * phoenix_out +
            self.weights["meanrev"] * meanrev_out +
            self.weights["catboost"] * cb_pred +
            self.weights["tabnet"] * tn_pred
        )

        final_signal = float(np.clip(final_raw, -1, 1))

        # 5) confidence (agreement)
        components = np.array([
            momentum_out,
            phoenix_out,
            meanrev_out,
            cb_pred,
            tn_pred
        ])

        conf = float(
            1 - (np.std(components) / (np.abs(np.mean(components)) + 1e-9))
        )
        conf = float(np.clip(conf, 0, 1))

        risk_score = float(1 - conf)

        return {
            "signal": final_signal,
            "confidence": conf,
            "risk_score": risk_score,
            "components": {
                "phoenix": phoenix_out,
                "momentum": momentum_out,
                "meanrev": meanrev_out,
                "catboost": cb_pred,
                "tabnet": tn_pred
            }
        }


    # --------------------------------------------------------
    #  Training the meta-models
    # --------------------------------------------------------
    def fit(
        self,
        feature_list: List[np.ndarray],
        target_list: List[float]
    ):

        X = np.vstack(feature_list).astype(float)
        y = np.array(target_list).astype(float)

        # normalize
        self.normalizer.fit(X)
        Xn = self.normalizer.transform(X)

        # CatBoost fit
        try:
            self.cb.fit(Xn, y)
        except Exception as e:
            self.logger.error(f"CatBoost fitting error: {e}")

        # TabNet fit
        try:
            self.tabnet.fit(
                X_train=Xn,
                y_train=y.reshape(-1, 1),
                max_epochs=30,
                patience=5,
                batch_size=256,
                virtual_batch_size=128,
            )
        except Exception as e:
            self.logger.error(f"TabNet fitting error: {e}")

        self.fitted = True


    # --------------------------------------------------------
    # LLM Alpha Feature Builder
    # --------------------------------------------------------
    def build_llm_features(
        self,
        *,
        market_text: Optional[str],
        news_headline: Optional[str],
        news_body: Optional[str],
        earnings_text: Optional[str],
        whisper_number: Optional[float]
    ) -> Dict[str, Any]:

        return self.llm.build_features(
            market_text=market_text,
            news_headline=news_headline,
            news_body=news_body,
            earnings_text=earnings_text,
            whisper_number=whisper_number
        )

    # --------------------------------------------------------
    # Full run (all engines + features)
    # --------------------------------------------------------
    def run(
        self,
        *,
        phoenix_out: float,
        momentum_out: float,
        meanrev_out: float,
        extra_alpha: Dict[str, Any],
        llm_texts: Dict[str, Optional[str]] = None
    ) -> Dict[str, Any]:

        llm_alpha = {}
        if llm_texts:
            llm_alpha = self.build_llm_features(
                market_text=llm_texts.get("market"),
                news_headline=llm_texts.get("news_headline"),
                news_body=llm_texts.get("news_body"),
                earnings_text=llm_texts.get("earnings_text"),
                whisper_number=extra_alpha.get("whisper_number")
            )

        # merge alphafeatures
        merged_alpha = {**extra_alpha, **llm_alpha}

        out = self.predict(
            phoenix_out=phoenix_out,
            momentum_out=momentum_out,
            meanrev_out=meanrev_out,
            alpha_features=merged_alpha
        )

        out["llm_alpha"] = llm_alpha
        out["merged_alpha"] = merged_alpha

        return out


# ============================================================
#   Multi-Asset Meta Engine
# ============================================================

class MultiAssetMetaEngineV73:
    """
    Applies meta-engine to multiple tickers.
    """

    def __init__(self, config: MetaConfig):
        self.engine = EnsembleMetaEngineV73(config)

    def generate(
        self,
        phoenix_map: Dict[str, float],
        momentum_map: Dict[str, float],
        meanrev_map: Dict[str, float],
        alpha_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:

        out = {}

        for sym in phoenix_map.keys():
            out[sym] = self.engine.predict(
                phoenix_out=phoenix_map[sym],
                momentum_out=momentum_map[sym],
                meanrev_out=meanrev_map[sym],
                alpha_features=alpha_map.get(sym, {})
            )

        return out

# ============================================================
#   Standalone test entry
# ============================================================

def meta_test_run(
    phoenix: float,
    momentum: float,
    meanrev: float,
    alpha_extra: Dict[str, Any]
) -> Dict[str, Any]:

    cfg = MetaConfig()
    engine = EnsembleMetaEngineV73(cfg)

    # This test will produce near-zero results unless model is trained,
    # but structural correctness is ensured.
    return engine.predict(
        phoenix_out=phoenix,
        momentum_out=momentum,
        meanrev_out=meanrev,
        alpha_features=alpha_extra
    )


# ============================================================
#   MODULE END (PART 4)
# ============================================================
