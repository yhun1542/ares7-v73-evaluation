import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import logging


# ============================================================
#  ARES-7 v73 — PHOENIX 10D ENGINE (FULL MODULE)
# ============================================================
#  - Overnight Momentum (패치1)
#  - Option Flow (Vanna/Charm) 강화
#  - Liquidity Fragmentation Feature
#  - Market Microstructure 10D expansion
#  - Regime-aware adaptive weighting
#  - LLM Alpha Integration Hook
#  - DIX/GEX integration hook
# ============================================================


@dataclass
class PhoenixConfig:
    lookback: int = 200
    vol_lookback: int = 20
    smoothing: float = 0.25
    vix_threshold: float = 25
    overnight_scale: float = 35.0
    fragmentation_window: int = 20
    optionflow_window: int = 20
    use_llm_alpha: bool = True
    use_gex: bool = True
    use_dix: bool = True
    use_whisperz: bool = True
    use_momentum_boost: bool = True


class PhoenixEngineV73:
    """
    ARES-7 v73 Phoenix 10D Engine
    Main multi-factor tactical trading engine.
    """

    def __init__(self, config: PhoenixConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    # ========================================================
    #  OVERNIGHT MOMENTUM  (패치1)
    # ========================================================
    def _overnight_momentum(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return 0.0

        close_prev = df["close"].iloc[-2]
        open_today = df["open"].iloc[-1]
        if close_prev <= 0:
            return 0.0

        overnight_ret = (open_today / close_prev) - 1

        vix = df.get("VIX", pd.Series([20] * len(df))).iloc[-1]
        weight = 0.45 if vix < self.config.vix_threshold else 0.20

        sig = np.tanh(overnight_ret * self.config.overnight_scale) * weight
        return float(np.clip(sig, -1, 1))

    # ========================================================
    #  OPTION FLOW FACTORS (VANNA / CHARM 등)
    # ========================================================
    def _option_flow_factors(self, df: pd.DataFrame) -> Dict[str, float]:
        # 필요한 컬럼 체크
        cols = ["vanna_flow_proxy", "charm_flow_proxy",
                "dealer_hedging_flow", "dealer_hedging_flow_z"]
        out = {c: 0.0 for c in cols}

        if not all(c in df.columns for c in cols):
            return out

        out = {
            "vanna": float(df["vanna_flow_proxy"].iloc[-1]),
            "charm": float(df["charm_flow_proxy"].iloc[-1]),
            "hedge_flow": float(df["dealer_hedging_flow"].iloc[-1]),
            "hedge_z": float(df["dealer_hedging_flow_z"].iloc[-1])
        }
        return out

    # ========================================================
    #  LIQUIDITY FRAGMENTATION INDEX
    # ========================================================
    def _liquidity_fragmentation(self, df: pd.DataFrame) -> float:
        if "nbbo_spread" not in df.columns:
            return 0.0
        if "fragmentation_ratio" not in df.columns:
            return 0.0

        frag = df["fragmentation_ratio"].iloc[-1]
        spread = df["nbbo_spread"].iloc[-1]

        score = np.tanh(frag * 2 - spread * 5)
        return float(np.clip(score, -1, 1))

    # ========================================================
    #  10D MICROSTRUCTURE SIGNAL
    # ========================================================
    def _microstructure_signal(self, df: pd.DataFrame) -> float:
        req = [
            "spread", "depth_imbalance", "order_flow_imbalance",
            "tick_direction", "volatility"
        ]
        if not all(r in df.columns for r in req):
            return 0.0

        spread = df["spread"].iloc[-1]
        flow = df["order_flow_imbalance"].iloc[-1]
        depth = df["depth_imbalance"].iloc[-1]
        tick = df["tick_direction"].iloc[-1]
        vol = df["volatility"].iloc[-1]

        raw = 0.3 * (-spread) + 0.25 * depth + 0.2 * flow + 0.15 * tick - 0.1 * vol
        return float(np.tanh(raw))

    # ========================================================
    #  LLM ALPHA HOOK
    # ========================================================
    def _llm_alpha_boost(self, llm_alpha: Optional[Dict[str, float]]) -> float:
        if llm_alpha is None:
            return 0.0
        if not self.config.use_llm_alpha:
            return 0.0

        return float(
            0.45 * llm_alpha.get("risk_scalar", 0) +
            0.25 * llm_alpha.get("sentiment_score", 0) +
            0.15 * llm_alpha.get("earnings_factor", 0) -
            0.15 * llm_alpha.get("uncertainty", 0)
        )

    # ========================================================
    #  PRIMARY ENGINE SCORE
    # ========================================================
    def generate_signal(
            self,
            df: pd.DataFrame,
            *,
            gex: float = 0.0,
            dix: float = 0.0,
            whisper_z: float = 0.0,
            llm_alpha: Optional[Dict[str, Any]] = None
    ) -> float:

        scores = []

        # 1) Overnight
        scores.append(self._overnight_momentum(df))

        # 2) Microstructure 10D
        scores.append(self._microstructure_signal(df))

        # 3) Fragmentation
        scores.append(self._liquidity_fragmentation(df))

        # 4) OptionFlow (Vanna/Charm)
        opt = self._option_flow_factors(df)
        opt_score = np.tanh(0.35 * opt["vanna"] + 0.35 * opt["charm"]
                            + 0.3 * opt["hedge_z"])
        scores.append(float(opt_score))

        # 5) GEX / DIX
        if self.config.use_gex:
            scores.append(np.tanh(gex / 2e9))
        if self.config.use_dix:
            scores.append(np.tanh((dix - 42) / 5))

        # 6) WhisperZ
        if self.config.use_whisperz:
            scores.append(np.tanh(whisper_z / 2))

        # 7) LLM Alpha boost
        scores.append(self._llm_alpha_boost(llm_alpha))

        base = float(np.mean(scores))

        # Smoothing
        final_sig = float(np.clip(base * (1 - self.config.smoothing) + base * self.config.smoothing,
                                  -1, 1))

        return final_sig

    # ========================================================
    #  SIGNAL EXPLANATION (for monitoring or debugging)
    # ========================================================
    def explain_signal(
            self,
            df: pd.DataFrame,
            *,
            gex: float = 0.0,
            dix: float = 0.0,
            whisper_z: float = 0.0,
            llm_alpha: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        explanation = {}

        explanation["overnight"] = self._overnight_momentum(df)
        explanation["microstructure"] = self._microstructure_signal(df)
        explanation["fragmentation"] = self._liquidity_fragmentation(df)

        opt = self._option_flow_factors(df)
        explanation["option_flow_score"] = float(
            np.tanh(0.35 * opt["vanna"] + 0.35 * opt["charm"]
                    + 0.3 * opt["hedge_z"])
        )

        explanation["gex"] = float(np.tanh(gex / 2e9)) if self.config.use_gex else 0.0
        explanation["dix"] = float(np.tanh((dix - 42) / 5)) if self.config.use_dix else 0.0
        explanation["whisper_z"] = float(np.tanh(whisper_z / 2)) if self.config.use_whisperz else 0.0
        explanation["llm_boost"] = self._llm_alpha_boost(llm_alpha)

        # final
        raw_scores = list(explanation.values())
        base = float(np.mean(raw_scores))
        final = float(np.clip(base * (1 - self.config.smoothing) + base * self.config.smoothing,
                              -1, 1))

        explanation["final_signal"] = final

        return explanation

# =====================================================================
#  PHOENIX FEATURE EXTRACTION SUBMODULES
# =====================================================================

class PhoenixFeatureExtractorV73:
    """
    Gathers additional feature sets:
      - microstructure
      - option flow
      - fragmentation
      - volatility
      - volume-based factors
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------------
    # Microstructure Features
    # ---------------------------------------------------------------
    def compute_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 3:
            df["spread"] = 0
            df["depth_imbalance"] = 0
            df["order_flow_imbalance"] = 0
            df["tick_direction"] = 0
            return df

        # Spread proxy
        df["spread"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

        # Depth imbalance (proxy)
        df["depth_imbalance"] = (
            df.get("bid_size", pd.Series([0] * len(df))) -
            df.get("ask_size", pd.Series([0] * len(df)))
        ) / (
            df.get("bid_size", pd.Series([1] * len(df))) +
            df.get("ask_size", pd.Series([1] * len(df)))
        )

        # Order flow imbalance (OFI proxy)
        ofi = []
        for i in range(len(df)):
            if i == 0:
                ofi.append(0)
            else:
                up = max(df["close"].iloc[i] - df["close"].iloc[i - 1], 0)
                down = max(df["close"].iloc[i - 1] - df["close"].iloc[i], 0)
                ofi.append(up - down)
        df["order_flow_imbalance"] = ofi

        # tick direction
        tick = []
        for i in range(len(df)):
            if i == 0:
                tick.append(0)
            else:
                direction = np.sign(df["close"].iloc[i] - df["close"].iloc[i - 1])
                tick.append(direction)
        df["tick_direction"] = tick

        return df

    # ---------------------------------------------------------------
    # Liquidity Fragmentation
    # ---------------------------------------------------------------
    def compute_fragmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        if "nbbo_spread" not in df.columns:
            df["nbbo_spread"] = 0

        if "fragmentation_ratio" not in df.columns:
            # Fallback fragmentation approximation
            df["fragmentation_ratio"] = (
                df["volume"].rolling(10).std() /
                (df["volume"].rolling(10).mean() + 1e-9)
            ).fillna(0)

        return df

    # ---------------------------------------------------------------
    # Option flow extension
    # ---------------------------------------------------------------
    def compute_option_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        # Proxy implementations if not available
        if "vanna_flow_proxy" not in df.columns:
            df["vanna_flow_proxy"] = (
                df["close"].pct_change().rolling(10).mean().fillna(0)
            )

        if "charm_flow_proxy" not in df.columns:
            df["charm_flow_proxy"] = (
                df["volume"].pct_change().rolling(10).mean().fillna(0)
            )

        if "dealer_hedging_flow" not in df.columns:
            df["dealer_hedging_flow"] = (
                df["close"].diff().rolling(10).sum().fillna(0)
            )

        if "dealer_hedging_flow_z" not in df.columns:
            series = df["dealer_hedging_flow"]
            df["dealer_hedging_flow_z"] = (
                (series - series.rolling(20).mean()) /
                (series.rolling(20).std() + 1e-9)
            ).fillna(0)

        return df

    # ---------------------------------------------------------------
    # Volatility features
    # ---------------------------------------------------------------
    def compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        returns = df["close"].pct_change().fillna(0)
        df["volatility"] = (
            returns.rolling(20).std().fillna(0)
        )
        return df

    # ---------------------------------------------------------------
    # Full feature stack
    # ---------------------------------------------------------------
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.compute_microstructure(df)
        df = self.compute_fragmentation(df)
        df = self.compute_option_flow(df)
        df = self.compute_volatility(df)
        return df


# =====================================================================
#  PHOENIX DATA WRAPPER (OPTIONAL)
# =====================================================================

class PhoenixDataWrapperV73:
    """
    Wraps OHLCV + microstructure + LLM alpha + GEX/DIX into
    a single unified feature-ready DataFrame.
    """

    def __init__(self, feature_extractor: PhoenixFeatureExtractorV73):
        self.fe = feature_extractor

    def prepare(
        self,
        df: pd.DataFrame,
        *,
        gex: float = 0,
        dix: float = 0,
        whisper_z: float = 0,
        llm_alpha: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:

        df = df.copy()

        # Feature expansion
        df = self.fe.add_all_features(df)

        # Add macro factors
        df["gex"] = gex
        df["dix"] = dix
        df["whisper_z"] = whisper_z

        if llm_alpha:
            df["llm_risk_scalar"] = llm_alpha.get("risk_scalar", 0)
            df["llm_sentiment"] = llm_alpha.get("sentiment_score", 0)
            df["llm_uncertainty"] = llm_alpha.get("uncertainty", 0)
        else:
            df["llm_risk_scalar"] = 0
            df["llm_sentiment"] = 0
            df["llm_uncertainty"] = 0

        return df

# =====================================================================
#  END OF PHOENIX ENGINE MODULE (PART 1)
# =====================================================================

# Phoenix Engine integration example (optional)
# This demonstrates how PhoenixEngineV73 and the DataWrapper combine
# inside the ARES-7 orchestrator.

class PhoenixOrchestratorAdapter:
    """
    Optional adapter class to connect Phoenix to orchestrator.
    """

    def __init__(self, config: PhoenixConfig):
        self.engine = PhoenixEngineV73(config)
        self.wrapper = PhoenixDataWrapperV73(PhoenixFeatureExtractorV73())

    def run(
        self,
        df: pd.DataFrame,
        *,
        gex: float = 0.0,
        dix: float = 0.0,
        whisper_z: float = 0.0,
        llm_alpha: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        # Wrap / feature expand
        data = self.wrapper.prepare(
            df,
            gex=gex,
            dix=dix,
            whisper_z=whisper_z,
            llm_alpha=llm_alpha
        )

        # Generate signal
        sig = self.engine.generate_signal(
            data,
            gex=gex,
            dix=dix,
            whisper_z=whisper_z,
            llm_alpha=llm_alpha
        )

        # Optional debugging summary
        explanation = self.engine.explain_signal(
            data,
            gex=gex,
            dix=dix,
            whisper_z=whisper_z,
            llm_alpha=llm_alpha
        )

        return {
            "signal": sig,
            "explanation": explanation,
            "data_snapshot": data.tail(1).to_dict(orient='records')[0]
        }



# ============================================================
#  STANDALONE UTILITY FUNCTIONS FOR TESTING
# ============================================================

def phoenix_test_run(df: pd.DataFrame) -> float:
    """
    Convenience test function for quick standalone checks.
    """
    cfg = PhoenixConfig()
    engine = PhoenixEngineV73(cfg)

    # dummy values (can be replaced by real alpha pipeline outputs)
    gex = 0.0
    dix = 45.2
    whisper_z = 0.0
    llm_alpha = {
        "risk_scalar": 0.1,
        "sentiment_score": 0.05,
        "uncertainty": 0.2,
        "earnings_factor": 0.0
    }

    return engine.generate_signal(
        df,
        gex=gex,
        dix=dix,
        whisper_z=whisper_z,
        llm_alpha=llm_alpha
    )


# ============================================================
#  MODULE END
# ============================================================

