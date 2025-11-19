import numpy as np
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

# 패치 9: Orderbook IRL 강화 모델 포함
from .orderbook_intent import OrderbookIntentModelV73


# ============================================================
#   ARES-7 v73 EXECUTION ENGINE
# ============================================================
# 포함 기능:
#   - IRL 기반 Order Intent 추론 (패치9)
#   - Depth Imbalance 기반 aggression 조절
#   - VPIN/GEX 기반 execution 리스크 필터
#   - TWAP/Sniper/Limit/Routing 자동 전환
#   - 슬리피지 최소화 로직
# ============================================================


class ExecutionConfigV73:
    def __init__(
        self,
        default_aggression: float = 0.5,
        max_slice: float = 0.15,            # order size max fraction
        min_slice: float = 0.05,
        vpin_threshold: float = 0.85,
        routing_mode: str = "auto"
    ):
        self.default_aggression = default_aggression
        self.max_slice = max_slice
        self.min_slice = min_slice
        self.vpin_threshold = vpin_threshold
        self.routing_mode = routing_mode


# ============================================================
#   MICROSTRUCTURE PARSER
# ============================================================

def parse_orderbook(book: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardizes orderbook structure for intent model.
    Expected keys:
      - bids: [(price, size), ...]
      - asks: [(price, size), ...]
      - last_trade: {price, size, side}
    """
    if "bids" not in book or "asks" not in book:
        return {"bids": [], "asks": [], "last_trade": {}}

    return {
        "bids": book.get("bids", []),
        "asks": book.get("asks", []),
        "last_trade": book.get("last_trade", {})
    }


# ============================================================
#   EXECUTION ENGINE CORE
# ============================================================

class ExecutionEngineV73:
    """
    Main Execution Engine with IRL-based order intent
    """

    def __init__(self, config: ExecutionConfigV73):
        self.config = config
        self.intent_model = OrderbookIntentModelV73()
        self.logger = logging.getLogger(self.__class__.__name__)


    # --------------------------------------------------------
    async def execute_order(
        self,
        order: Dict[str, Any],
        orderbook_snapshot: Dict[str, Any],
        *,
        vpin: float = 0.0,
        gex: float = 0.0
    ) -> Dict[str, Any]:

        """
        order: {
            "symbol": str,
            "side": "buy"|"sell",
            "quantity": float,
            "order_type": "limit"|"market"|"auto",
            "limit_price": float (optional)
        }
        """

        side = order["side"].lower()
        qty = order["quantity"]

        # ================================================
        # 1) VPIN Toxicity: execution pause
        # ================================================
        if vpin > self.config.vpin_threshold:
            return {
                "executed_qty": 0.0,
                "price": None,
                "intent_score": 0.0,
                "aggression": 0.0,
                "reason": "VPIN_TOXICITY_BLOCK",
                "timestamp": datetime.now().isoformat()
            }

        # ================================================
        # 2) Orderbook IRL Intent
        # ================================================
        ob = parse_orderbook(orderbook_snapshot)
        intent = self.intent_model.score_orderbook(ob)

        intent_score = intent["intent"]
        aggression = intent["aggression"]
        confidence = intent["confidence"]

        # ================================================
        # 3) Compute final aggression score
        # ================================================
        base = self.config.default_aggression

        if side == "buy":
            final_aggr = np.clip(base + aggression + intent_score, 0.0, 1.0)
        else:
            final_aggr = np.clip(base + aggression - intent_score, 0.0, 1.0)

        # ================================================
        # 4) Price computation
        # ================================================
        best_bid = ob["bids"][0][0] if ob["bids"] else None
        best_ask = ob["asks"][0][0] if ob["asks"] else None

        if side == "buy":
            if order["order_type"] == "market":
                price = best_ask
            else:
                price = best_ask * (1 + 0.001 * (1 - final_aggr))
        else:
            if order["order_type"] == "market":
                price = best_bid
            else:
                price = best_bid * (1 - 0.001 * (1 - final_aggr))

        # ================================================
        # 5) Slice the order according to aggression
        # ================================================
        slice_qty = max(
            qty * max(final_aggr, self.config.min_slice),
            qty * self.config.min_slice
        )
        slice_qty = min(slice_qty, qty * self.config.max_slice)

        # ================================================
        # 6) Output
        # ================================================
        result = {
            "executed_qty": float(slice_qty),
            "price": float(price) if price is not None else None,
            "intent_score": float(intent_score),
            "aggression": float(final_aggr),
            "confidence": float(confidence),
            "timestamp": datetime.now().isoformat()
        }

        return result

# ============================================================
#   ROUTING ENGINE (Dark Pool / Lit / Midpoint)
# ============================================================

class OrderRouterV73:
    """
    Chooses execution venue based on:
      - intent score
      - aggression
      - liquidity
      - market conditions
    """

    def __init__(self):
        pass

    def route(
        self,
        *,
        symbol: str,
        intent_score: float,
        aggression: float,
        volatility: float,
        vpin: float
    ) -> str:

        # High toxicity → Midpoint routing
        if vpin > 0.75:
            return "MIDPOINT"

        # Strong aggressive intent → Lit markets
        if aggression > 0.7:
            return "LIT"

        # Weak intent or sideways → Dark pools
        if abs(intent_score) < 0.1:
            return "DARK"

        # Default
        return "AUTO"


# ============================================================
#   SLIPPAGE ESTIMATION
# ============================================================

class SlippageEstimatorV73:
    """
    Estimates slippage for executed slices.
    """

    def __init__(self):
        pass

    def estimate(
        self,
        *,
        volatility: float,
        depth_imbalance: float,
        size: float
    ) -> float:

        base = abs(size) * (0.0002 + volatility * 0.4)
        depth_adj = (1 - abs(depth_imbalance))

        return float(base * depth_adj)


# ============================================================
#   LATENCY MODEL
# ============================================================

class LatencyModelV73:
    """
    Models latency impact on execution quality.
    """

    def __init__(self):
        pass

    def penalize(self, signal: float, latency_ms: float) -> float:
        penalty = np.tanh(latency_ms / 1000)
        return float(signal * (1 - penalty))

# ============================================================
#   FULL EXECUTION PIPELINE
# ============================================================

class FullExecutionPipelineV73:
    """
    Combines:
      - ExecutionEngineV73
      - OrderRouterV73
      - SlippageEstimatorV73
      - LatencyModelV73
    """

    def __init__(self, exec_config: ExecutionConfigV73):
        self.exec_engine = ExecutionEngineV73(exec_config)
        self.router = OrderRouterV73()
        self.slip = SlippageEstimatorV73()
        self.latency = LatencyModelV73()

    async def execute(
        self,
        *,
        order: Dict[str, Any],
        orderbook_snapshot: Dict[str, Any],
        volatility: float,
        depth_imbalance: float,
        latency_ms: float,
        vpin: float,
        gex: float
    ) -> Dict[str, Any]:

        # 1) routing mode
        route = self.router.route(
            symbol=order["symbol"],
            intent_score=0.0,     # placeholder (intent comes later)
            aggression=0.5,       # placeholder
            volatility=volatility,
            vpin=vpin
        )

        # 2) Execute slice
        result = await self.exec_engine.execute_order(
            order,
            orderbook_snapshot,
            vpin=vpin,
            gex=gex
        )

        # 3) Slippage estimate
        slip = self.slip.estimate(
            volatility=volatility,
            depth_imbalance=depth_imbalance,
            size=result["executed_qty"]
        )

        # 4) Latency penalty
        adjusted_price = self.latency.penalize(
            signal=result["price"],
            latency_ms=latency_ms
        )

        result.update({
            "route": route,
            "slippage_estimate": slip,
            "adjusted_price": adjusted_price
        })

        return result

# ============================================================
#   MULTI-ASSET EXECUTION ENGINE
# ============================================================

class MultiAssetExecutionV73:
    """
    Handles execution for multiple tickers sequentially or in parallel.
    """

    def __init__(self, exec_config: ExecutionConfigV73):
        self.pipeline = FullExecutionPipelineV73(exec_config)

    async def run(
        self,
        order_map: Dict[str, Dict[str, Any]],
        orderbook_map: Dict[str, Dict[str, Any]],
        volatility_map: Dict[str, float],
        depth_map: Dict[str, float],
        latency_map: Dict[str, float],
        vpin_map: Dict[str, float],
        gex_map: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:

        results = {}

        tasks = []
        for sym, order in order_map.items():
            tasks.append(
                asyncio.create_task(
                    self.pipeline.execute(
                        order=order,
                        orderbook_snapshot=orderbook_map[sym],
                        volatility=volatility_map[sym],
                        depth_imbalance=depth_map[sym],
                        latency_ms=latency_map[sym],
                        vpin=vpin_map[sym],
                        gex=gex_map[sym]
                    )
                )
            )

        outs = await asyncio.gather(*tasks)

        for i, sym in enumerate(order_map.keys()):
            results[sym] = outs[i]

        return results


# ============================================================
#   ROUTING / EXECUTION VISUALIZATION UTIL (optional)
# ============================================================

def plot_execution_prices(prices: List[float]):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(prices, label="Executed Prices")
        plt.legend()
        plt.show()
    except:
        pass

# ============================================================
#   Standalone Test Entry
# ============================================================

async def execution_test_run(order, ob_snapshot):
    cfg = ExecutionConfigV73()
    engine = ExecutionEngineV73(cfg)

    out = await engine.execute_order(
        order,
        ob_snapshot,
        vpin=0.2,
        gex=0.0
    )
    return out


# ============================================================
#   MODULE END (PART 7)
# ============================================================

