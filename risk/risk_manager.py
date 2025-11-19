import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

import logging


# ============================================================
#   ARES-7 v73 RISK MANAGEMENT ENGINE
# ============================================================
# 포함 기능:
#   - VPIN Toxicity Kill-Switch (패치3)
#   - Continuous-Time Kelly Sizing (패치5)
#   - GEX Crash/Trend Filter (패치2)
#   - Regime-aware leverage modulation
#   - Volatility-aware scaling
# ============================================================


# ============================================================
#   VPIN MODULE
# ============================================================

class VPINCalculator:
    """
    VPIN (Volume-Synchronized Probability of Informed Trading)
    v73 version.
    """

    def __init__(self, window: int = 50):
        self.window = window

    def calculate(self, volume_series: pd.Series) -> float:
        if len(volume_series) < self.window:
            return 0.5

        vol_sum = volume_series.rolling(self.window).sum()
        signed_volume = np.sign(volume_series.diff().fillna(0)) * volume_series

        vpin = signed_volume.rolling(self.window).sum().abs() / (vol_sum + 1e-9)

        return float(np.clip(vpin.iloc[-1], 0, 1))


# ============================================================
#   GEX FILTER MODULE
# ============================================================

class GEXFilter:
    """
    GEX (Gamma Exposure) filter for crash detection & trend enhancement.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply(self, size: float, gex: float) -> float:
        # Crash zone
        if gex < -2_000_000_000:
            if size > 0:  # block long
                return 0.0
            return size

        # Trend fuel zone
        if gex > 3_000_000_000:
            return size * 1.35

        return size


# ============================================================
#   RISK MANAGER CORE
# ============================================================

class RiskManagerV73:
    """
    Full ARES-7 v73 Risk Manager (Momentum/MR/Phoenix 공통)
    """

    def __init__(self, base_position_size: float = 1_000_000):
        self.base_position_size = base_position_size
        self.vpin_calc = VPINCalculator()
        self.gex_filter = GEXFilter()

    # --------------------------------------------------------
    def calculate_position_size(
        self,
        *,
        signal: float,            # model raw signal (-1 to 1)
        confidence: float,        # model confidence 0 to 1
        quality: float,           # quality score 0 to 1
        volatility: float,        # realized vol
        volume_series: pd.Series, # for VPIN
        gex: float,               # gamma exposure
        regime: Dict[str, Any]    # {"type": "..."}
    ) -> float:

        # ================================================
        # 1) VPIN Toxicity Kill-Switch (패치3)
        # ================================================
        vpin = self.vpin_calc.calculate(volume_series)
        if vpin > 0.85:
            return 0.0

        # ================================================
        # 2) Continuous-Time Kelly (패치5)
        # ================================================
        SR = confidence * quality * abs(signal) / (volatility + 1e-9)
        f_star = SR / (1.0 + SR * SR) * np.sign(signal)

        size = self.base_position_size * f_star

        # Clamp leverage
        max_notional = self.base_position_size * 1.6
        size = float(np.clip(size, -max_notional, max_notional))

        # ================================================
        # 3) Regime-aware adjustments
        # ================================================
        regime_type = regime.get("type", "normal")

        if regime_type == "trending":
            size *= 1.25
        elif regime_type == "choppy":
            size *= 0.6
        elif regime_type == "volatile":
            size *= 0.5

        # ================================================
        # 4) GEX Filtering (패치2)
        # ================================================
        size = self.gex_filter.apply(size, gex)

        return float(size)

# ============================================================
#   Risk Score Computation
# ============================================================

def compute_risk_score(
    *,
    volatility: float,
    vpin: float,
    gex: float,
    drawdown: float
) -> float:
    """
    Computes a normalized risk score combining key microstructure,
    volatility, and gamma signals.
    """

    vol_component = np.tanh(volatility * 5)
    vpin_component = float(vpin)
    gex_component = float(-np.tanh(gex / 2e9))   # negative GEX increases risk
    dd_component = float(drawdown)

    composite = (
        0.35 * vol_component +
        0.25 * vpin_component +
        0.25 * gex_component +
        0.15 * dd_component
    )

    return float(np.clip(composite, 0, 1))


# ============================================================
#   Stop-Loss + Take-Profit Logic
# ============================================================

class RiskLimitControllerV73:
    """
    Applies dynamic SL/TP based on volatility + VPIN.
    """

    def __init__(self):
        pass

    def compute_limits(
        self,
        *,
        entry_price: float,
        volatility: float,
        vpin: float
    ) -> Dict[str, float]:

        # wider stops in low-vol, tighter in high-vol
        vol_adj = 1.0 + volatility * 5
        vpin_adj = 1.0 + vpin

        stop_loss = float(entry_price * (1 - 0.02 * vol_adj * vpin_adj))
        take_profit = float(entry_price * (1 + 0.03 / (vol_adj)))

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }


# ============================================================
#   Portfolio-Level Constraints
# ============================================================

class PortfolioConstraintEngineV73:
    """
    Controls overall exposure limits (portfolio level).
    """

    def __init__(
        self,
        max_portfolio_risk: float = 0.025,
        max_concentration: float = 0.25
    ):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_concentration = max_concentration

    def enforce(
        self,
        allocations: Dict[str, float],
        risk_scores: Dict[str, float],
        total_capital: float
    ) -> Dict[str, float]:

        adjusted = {}
        cap = total_capital

        for sym, alloc in allocations.items():
            risk = risk_scores.get(sym, 0)

            # reduce position size if risk too high
            if risk > self.max_portfolio_risk:
                alloc *= 0.5

            # concentration limit
            alloc = min(alloc, cap * self.max_concentration)
            adjusted[sym] = alloc

        return adjusted

# ============================================================
#   Transaction Cost / Slippage Model
# ============================================================

class SlippageModelV73:
    """
    Estimates slippage based on:
      - volatility
      - order book depth (proxy)
      - VPIN toxicity
    """

    def __init__(self):
        pass

    def estimate(
        self,
        *,
        volatility: float,
        depth_imbalance: float,
        vpin: float,
        notional: float
    ) -> float:

        base = abs(notional) * (0.0002 + volatility * 0.5)

        imbalance_adj = (1 - abs(depth_imbalance))
        toxicity_adj = (1 + vpin)

        return float(base * imbalance_adj * toxicity_adj)


# ============================================================
#   Risk Event Flags
# ============================================================

def detect_risk_events(
    *,
    vpin: float,
    gex: float,
    volatility: float,
    regime_type: str
) -> List[str]:

    flags = []

    if vpin > 0.85:
        flags.append("VPIN_TOXICITY")

    if gex < -2_000_000_000:
        flags.append("NEGATIVE_GAMMA_CRASH_ZONE")

    if volatility > 0.04:
        flags.append("HIGH_VOLATILITY")

    if regime_type == "volatile":
        flags.append("VOLATILE_REGIME")

    return flags


# ============================================================
#   Multi-Asset Risk Manager (optional)
# ============================================================

class MultiAssetRiskManagerV73:
    """
    Applies RiskManagerV73 to multiple tickers.
    """

    def __init__(self, base_position_size: float = 1_000_000):
        self.rm = RiskManagerV73(base_position_size)

    def generate_allocations(
        self,
        *,
        signals: Dict[str, float],
        confidence: Dict[str, float],
        quality: Dict[str, float],
        volatility: Dict[str, float],
        volume_map: Dict[str, pd.Series],
        gex_map: Dict[str, float],
        regime_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:

        alloc = {}

        for sym in signals:
            alloc[sym] = self.rm.calculate_position_size(
                signal=signals[sym],
                confidence=confidence[sym],
                quality=quality[sym],
                volatility=volatility[sym],
                volume_series=volume_map[sym],
                gex=gex_map.get(sym, 0.0),
                regime=regime_map.get(sym, {"type": "normal"})
            )

        return alloc

# ============================================================
#   Portfolio Optimizer (Mean-Variance)
# ============================================================

class PortfolioOptimizerV73:
    """
    Simplified mean-variance optimizer with risk & GEX awareness.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.rf = risk_free_rate

    def optimize(
        self,
        *,
        expected_returns: Dict[str, float],
        risk_scores: Dict[str, float],
        capital: float
    ) -> Dict[str, float]:

        symbols = list(expected_returns.keys())
        n = len(symbols)
        if n == 0:
            return {}

        rets = np.array([expected_returns[s] for s in symbols])
        risks = np.array([risk_scores[s] for s in symbols])

        # inverse-risk weighting
        inv = 1 / (risks + 1e-9)
        inv /= inv.sum()

        alloc = {symbols[i]: float(capital * inv[i]) for i in range(n)}
        return alloc


# ============================================================
#   Portfolio Risk Tracking
# ============================================================

class PortfolioRiskTrackerV73:
    """
    Tracks rolling volatility, drawdown, PnL at portfolio level.
    """

    def __init__(self):
        self.equity_curve = []
        self.peak = 0.0

    def update(self, equity: float):
        self.equity_curve.append(equity)
        if equity > self.peak:
            self.peak = equity

    def compute(self) -> Dict[str, float]:
        if len(self.equity_curve) < 2:
            return {"vol": 0, "dd": 0}

        arr = np.array(self.equity_curve)
        vol = float(np.std(np.diff(arr)) * np.sqrt(252))
        dd = float((self.peak - arr[-1]) / (self.peak + 1e-9))
        return {"vol": vol, "dd": dd}

# ============================================================
#   Standalone Test
# ============================================================

def risk_test_run(
    signal: float,
    confidence: float,
    quality: float,
    volatility: float,
    volume_series: pd.Series,
    gex: float,
    regime: Dict[str, Any]
) -> float:

    rm = RiskManagerV73()
    return rm.calculate_position_size(
        signal=signal,
        confidence=confidence,
        quality=quality,
        volatility=volatility,
        volume_series=volume_series,
        gex=gex,
        regime=regime
    )


# ============================================================
#   MODULE END (PART 5)
# ============================================================

