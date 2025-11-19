import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ============================================================
#   ARES-7 v73 Mean Reversion Engine (Full Module)
# ============================================================
# 포함 기능:
#   - NSV 기반 변동성 모델 (패치10 연동)
#   - Johansen Cointegration (VECM 기반)
#   - 확장형 MR 시그널 (spread Z-score)
#   - drift / volatility scaling
#   - multi-asset pair/triple 지원
# ============================================================


@dataclass
class MeanRevConfig:
    lookback: int = 120          # price window
    z_entry: float = 1.5         # z-score entry
    z_exit: float = 0.3          # mean reversion exit
    vol_smoothing: float = 0.25
    min_samples: int = 60
    use_nsv: bool = True         # NSV 변동성 사용 여부
    use_vecm: bool = True        # Johansen cointegration 사용 여부
    max_leverage: float = 1.6
    base_position: float = 100000


# ============================================================
#   NSV Model (패치10에서 따로 제공된 모듈을 import하는 구조)
# ============================================================

from .nsr_volatility import NSVVolatilityModel


class MRVolatilityAdapter:
    """
    NSV + Rolling Volatility fallback
    """

    def __init__(self):
        self.nsv = NSVVolatilityModel()

    def forecast(self, prices: pd.Series) -> float:
        if len(prices) < 40:
            return float(prices.pct_change().rolling(10).std().iloc[-1])

        return float(self.nsv.predict_vol(prices))


# ============================================================
#   Johansen Cointegration (VECM)
# ============================================================

def detect_vecm_cointegration(series_list: List[pd.Series]) -> Dict[str, Any]:
    """
    series_list: [asset1, asset2, ...] 형태
    """

    if len(series_list) < 2:
        return {"rank": 0}

    df = pd.concat(series_list, axis=1).dropna()
    if df.shape[0] < 60:
        return {"rank": 0}

    try:
        result = coint_johansen(df.values, det_order=0, k_ar_diff=1)
        trace_stats = result.lr1
        crit_vals = result.cvt[:, 1]   # 95%

        rank = 0
        for i in range(len(trace_stats)):
            if trace_stats[i] > crit_vals[i]:
                rank = i + 1

        if rank == 0:
            return {"rank": 0}

        hedge_vec = result.evec[:, 0]
        spread = df.values @ hedge_vec

        return {
            "rank": rank,
            "hedge": hedge_vec,
            "spread_mean": float(np.mean(spread)),
            "spread_std": float(np.std(spread)),
            "spread_series": spread
        }

    except Exception:
        return {"rank": 0}


# ============================================================
#   Mean Reversion Engine 본체 (v73)
# ============================================================

class MeanReversionEngineV73:
    """
    Multi-asset MR engine using:
      - VECM cointegration
      - NSV volatility forecast
      - Z-score based signal
    """

    def __init__(self, config: MeanRevConfig):
        self.config = config
        self.vol_adapter = MRVolatilityAdapter()

    # --------------------------------------------------------
    # Spread 계산
    # --------------------------------------------------------
    def _compute_spread(self, prices: List[pd.Series]) -> Dict[str, Any]:
        if self.config.use_vecm:
            res = detect_vecm_cointegration(prices)
            if res.get("rank", 0) > 0:
                return res

        # fallback: simple pair hedge
        s1 = prices[0]
        s2 = prices[1]
        beta = np.polyfit(s1, s2, 1)[0]
        spread = s2 - beta * s1

        return {
            "rank": 1,
            "hedge": np.array([1, -beta]),
            "spread_series": spread.values,
            "spread_mean": float(spread.mean()),
            "spread_std": float(spread.std() + 1e-9)
        }

    # --------------------------------------------------------
    # MR Signal 계산
    # --------------------------------------------------------
    def _compute_signal(self, spread: np.ndarray, vol_forecast: float) -> float:
        if len(spread) < 10:
            return 0.0

        mean = np.mean(spread)
        std = np.std(spread) + 1e-9
        z = (spread[-1] - mean) / std

        # NSV 기반 scaling
        scaled_z = z / (1 + 3 * vol_forecast)

        return float(np.clip(-scaled_z, -1, 1))

    # --------------------------------------------------------
    # PUBLIC: Generate MR signal (single pair or multi-asset)
    # --------------------------------------------------------
    def generate_signal(
        self,
        price_dict: Dict[str, pd.Series]   # {"AAPL": series, "MSFT": series}
    ) -> Dict[str, Any]:

        # 1) 최소 샘플 체크
        if any(len(s) < self.config.min_samples for s in price_dict.values()):
            return {
                "signal": 0.0,
                "spread_z": 0.0,
                "vol_forecast": 0.0,
                "hedge": None,
                "assets": list(price_dict.keys())
            }

        # 2) 가격 시리즈 정렬
        series_list = [price_dict[k].iloc[-self.config.lookback:] for k in price_dict]
        assets = list(price_dict.keys())

        # 3) Spread 계산 (vecm / fallback)
        spread_info = self._compute_spread(series_list)
        spread_series = spread_info["spread_series"]

        # 4) 변동성 예측 (NSV)
        vol_forecast = self.vol_adapter.forecast(series_list[0])

        # 5) 신호 계산
        sig = self._compute_signal(spread_series, vol_forecast)

        return {
            "signal": sig,
            "spread_z": float((spread_series[-1] - spread_info["spread_mean"]) /
                              (spread_info["spread_std"] + 1e-9)),
            "vol_forecast": vol_forecast,
            "hedge": spread_info["hedge"],
            "assets": assets
        }


# =====================================================================
#  Optional MR Orchestrator Adapter
# =====================================================================

class MeanReversionAdapterV73:
    """
    Optional adapter between MR engine and orchestrator
    """

    def __init__(self, config: MeanRevConfig):
        self.engine = MeanReversionEngineV73(config)

    def run(self, price_dict: Dict[str, pd.Series]) -> Dict[str, Any]:
        return self.engine.generate_signal(price_dict)


# ============================================================
#  Spread Diagnostics (optional)
# ============================================================

def spread_diagnostics(spread: np.ndarray) -> Dict[str, float]:
    """
    Useful for debugging or visualization.
    """
    if len(spread) < 20:
        return {"mean": 0.0, "std": 0.0, "last": 0.0}

    mean = float(np.mean(spread))
    std = float(np.std(spread) + 1e-9)
    last = float(spread[-1])

    return {
        "mean": mean,
        "std": std,
        "last": last,
        "z_score": float((last - mean) / std)
    }


# ============================================================
#  Backtest Utility (optional)
# ============================================================

def mr_backtest(prices_a: pd.Series, prices_b: pd.Series,
                 config: Optional[MeanRevConfig] = None) -> Dict[str, Any]:
    """
    Quick backtest between two assets for demonstration.
    """

    if config is None:
        config = MeanRevConfig()

    engine = MeanReversionEngineV73(config)

    returns = []
    spreads = []

    for i in range(config.lookback, len(prices_a)):
        pa = prices_a.iloc[:i]
        pb = prices_b.iloc[:i]

        out = engine.generate_signal({"A": pa, "B": pb})
        sig = out["signal"]

        # simple return model
        retn = sig * (prices_b.iloc[i] - prices_b.iloc[i - 1]) / prices_b.iloc[i - 1]
        returns.append(retn)

        spread = out.get("spread_z", 0)
        spreads.append(spread)

    if len(returns) == 0:
        return {"pnl": 0, "sharpe": 0}

    pnl = float(np.sum(returns))
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252))

    return {"pnl": pnl, "sharpe": sharpe}

# ============================================================
#  Helper: Normalize price series length
# ============================================================

def align_price_series(series_list: List[pd.Series], window: int) -> List[pd.Series]:
    """
    Cut/align price series to a uniform window for MR processing.
    """
    aligned = []
    for s in series_list:
        if len(s) >= window:
            aligned.append(s.iloc[-window:])
        else:
            aligned.append(s.copy())
    return aligned


# ============================================================
#  Multi-Asset MR Extension
# ============================================================

class MultiAssetMeanReversionV73:
    """
    Handles pair / triple / basket mean reversion.
    """

    def __init__(self, config: MeanRevConfig):
        self.engine = MeanReversionEngineV73(config)
        self.config = config

    def generate(self, price_map: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        price_map: {"SPY": series, "QQQ": series, "IWM": series}
        """

        if len(price_map) < 2:
            return {"signal": 0.0, "detail": {}}

        keys = list(price_map.keys())
        aligned = align_price_series(
            [price_map[k] for k in keys],
            self.config.lookback
        )
        mapping = {k: aligned[i] for i, k in enumerate(keys)}

        res = self.engine.generate_signal(mapping)
        return res


# ============================================================
#  Visualization Helper (optional)
# ============================================================

def plot_spread(spread: np.ndarray):
    """
    Plots a spread series using matplotlib (optional).
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(spread, label="Spread")
        plt.axhline(np.mean(spread), color='r', linestyle='--')
        plt.legend()
        plt.show()
    except Exception:
        pass

# ============================================================
#  STANDALONE TEST ENTRY
# ============================================================

def meanrev_test_run(price_a: pd.Series, price_b: pd.Series) -> float:
    cfg = MeanRevConfig()
    engine = MeanReversionEngineV73(cfg)

    out = engine.generate_signal({"A": price_a, "B": price_b})
    return out.get("signal", 0.0)


# ============================================================
#  MODULE END (PART 2)
# ============================================================

