"""
Orderbook Intent Model for ARES-7 v73
Execution Engine Component

This model analyzes orderbook dynamics to detect institutional intent
and optimize trade execution timing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderbookSnapshot:
    """Orderbook snapshot data"""
    timestamp: float
    bid_prices: List[float]
    bid_sizes: List[float]
    ask_prices: List[float]
    ask_sizes: List[float]
    last_price: float
    volume: float


class OrderbookIntentModelV73:
    """
    Orderbook Intent Analysis Model
    
    Analyzes Level 2 orderbook data (or proxies) to detect:
    - Institutional order flow
    - Hidden liquidity
    - Optimal execution timing
    - Market impact estimation
    
    When Level 2 data is unavailable, uses OHLCV proxies.
    """
    
    def __init__(self, 
                 depth_levels: int = 5,
                 imbalance_threshold: float = 0.3,
                 pressure_window: int = 10):
        """
        Initialize orderbook intent model
        
        Args:
            depth_levels: Number of price levels to analyze
            imbalance_threshold: Threshold for significant imbalance (0-1)
            pressure_window: Window for pressure accumulation
        """
        self.depth_levels = depth_levels
        self.imbalance_threshold = imbalance_threshold
        self.pressure_window = pressure_window
        
        self.orderbook_history = []
        
        logger.info(f"Orderbook Intent Model initialized: "
                   f"depth={depth_levels}, threshold={imbalance_threshold}")
    
    def calculate_depth_imbalance(self, snapshot: OrderbookSnapshot) -> float:
        """
        Calculate bid-ask depth imbalance
        
        Positive imbalance = more buying pressure
        Negative imbalance = more selling pressure
        
        Args:
            snapshot: Orderbook snapshot
        
        Returns:
            Imbalance ratio (-1 to 1)
        """
        total_bid_size = sum(snapshot.bid_sizes[:self.depth_levels])
        total_ask_size = sum(snapshot.ask_sizes[:self.depth_levels])
        
        if total_bid_size + total_ask_size == 0:
            return 0.0
        
        imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
        return imbalance
    
    def calculate_spread(self, snapshot: OrderbookSnapshot) -> Tuple[float, float]:
        """
        Calculate bid-ask spread
        
        Args:
            snapshot: Orderbook snapshot
        
        Returns:
            (absolute_spread, relative_spread_bps)
        """
        if not snapshot.bid_prices or not snapshot.ask_prices:
            return 0.0, 0.0
        
        best_bid = snapshot.bid_prices[0]
        best_ask = snapshot.ask_prices[0]
        
        absolute_spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        if mid_price > 0:
            relative_spread_bps = (absolute_spread / mid_price) * 10000
        else:
            relative_spread_bps = 0.0
        
        return absolute_spread, relative_spread_bps
    
    def detect_hidden_liquidity(self, snapshot: OrderbookSnapshot) -> Dict:
        """
        Detect potential hidden liquidity using iceberg order patterns
        
        Args:
            snapshot: Orderbook snapshot
        
        Returns:
            Dictionary with hidden liquidity indicators
        """
        # Look for repeated order sizes at same price level
        # (simplified detection - real implementation would be more sophisticated)
        
        bid_sizes = snapshot.bid_sizes[:self.depth_levels]
        ask_sizes = snapshot.ask_sizes[:self.depth_levels]
        
        # Check for unusually consistent sizes (potential icebergs)
        bid_consistency = np.std(bid_sizes) / (np.mean(bid_sizes) + 1e-8) if bid_sizes else 0
        ask_consistency = np.std(ask_sizes) / (np.mean(ask_sizes) + 1e-8) if ask_sizes else 0
        
        # Low coefficient of variation suggests hidden orders
        hidden_bid_score = 1.0 / (1.0 + bid_consistency)
        hidden_ask_score = 1.0 / (1.0 + ask_consistency)
        
        return {
            'hidden_bid_score': hidden_bid_score,
            'hidden_ask_score': hidden_ask_score,
            'likely_hidden_liquidity': (hidden_bid_score > 0.7 or hidden_ask_score > 0.7)
        }
    
    def calculate_order_flow_toxicity(self, recent_snapshots: List[OrderbookSnapshot]) -> float:
        """
        Calculate order flow toxicity (adverse selection risk)
        
        High toxicity = informed traders present = higher execution risk
        
        Args:
            recent_snapshots: Recent orderbook snapshots
        
        Returns:
            Toxicity score (0-1)
        """
        if len(recent_snapshots) < 2:
            return 0.5  # Neutral
        
        # Calculate VPIN (Volume-Synchronized Probability of Informed Trading)
        # Simplified version
        
        buy_volume = 0
        sell_volume = 0
        
        for i in range(1, len(recent_snapshots)):
            prev = recent_snapshots[i-1]
            curr = recent_snapshots[i]
            
            price_change = curr.last_price - prev.last_price
            volume = curr.volume
            
            if price_change > 0:
                buy_volume += volume
            elif price_change < 0:
                sell_volume += volume
        
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return 0.5
        
        # Volume imbalance as proxy for toxicity
        toxicity = abs(buy_volume - sell_volume) / total_volume
        
        return toxicity
    
    def estimate_market_impact(self, order_size: float, 
                               snapshot: OrderbookSnapshot) -> Dict:
        """
        Estimate market impact of an order
        
        Args:
            order_size: Size of order (positive for buy, negative for sell)
            snapshot: Current orderbook snapshot
        
        Returns:
            Dictionary with impact estimates
        """
        is_buy = order_size > 0
        abs_size = abs(order_size)
        
        if is_buy:
            prices = snapshot.ask_prices[:self.depth_levels]
            sizes = snapshot.ask_sizes[:self.depth_levels]
        else:
            prices = snapshot.bid_prices[:self.depth_levels]
            sizes = snapshot.bid_sizes[:self.depth_levels]
        
        if not prices or not sizes:
            return {
                'estimated_price': snapshot.last_price,
                'impact_bps': 0,
                'liquidity_available': 0
            }
        
        # Calculate volume-weighted average price for order
        cumulative_size = 0
        weighted_price_sum = 0
        
        for price, size in zip(prices, sizes):
            available = min(size, abs_size - cumulative_size)
            weighted_price_sum += price * available
            cumulative_size += available
            
            if cumulative_size >= abs_size:
                break
        
        if cumulative_size > 0:
            avg_execution_price = weighted_price_sum / cumulative_size
        else:
            avg_execution_price = prices[0]
        
        # Calculate impact in basis points
        mid_price = (snapshot.bid_prices[0] + snapshot.ask_prices[0]) / 2
        impact_bps = abs((avg_execution_price - mid_price) / mid_price) * 10000
        
        return {
            'estimated_price': avg_execution_price,
            'impact_bps': impact_bps,
            'liquidity_available': cumulative_size,
            'liquidity_sufficient': cumulative_size >= abs_size
        }
    
    def get_execution_timing_score(self, snapshot: OrderbookSnapshot) -> Tuple[float, Dict]:
        """
        Get execution timing score (0-1, higher is better)
        
        Args:
            snapshot: Current orderbook snapshot
        
        Returns:
            (score, metrics) where score indicates execution favorability
        """
        # Calculate key metrics
        imbalance = self.calculate_depth_imbalance(snapshot)
        abs_spread, rel_spread_bps = self.calculate_spread(snapshot)
        hidden_liq = self.detect_hidden_liquidity(snapshot)
        
        # Add to history
        self.orderbook_history.append(snapshot)
        if len(self.orderbook_history) > self.pressure_window:
            self.orderbook_history.pop(0)
        
        # Calculate toxicity if we have history
        if len(self.orderbook_history) >= 2:
            toxicity = self.calculate_order_flow_toxicity(self.orderbook_history)
        else:
            toxicity = 0.5
        
        # Execution timing score components:
        # 1. Low spread is good (0-40 points)
        # 2. Balanced orderbook is good (0-30 points)
        # 3. Low toxicity is good (0-30 points)
        
        # Spread score (lower is better, normalize to 0-1)
        # Assume typical spread is 5-20 bps
        spread_score = max(0, 1.0 - (rel_spread_bps - 5) / 15)
        spread_score = min(1.0, spread_score)
        
        # Balance score (closer to 0 imbalance is better)
        balance_score = 1.0 - abs(imbalance)
        
        # Toxicity score (lower toxicity is better)
        toxicity_score = 1.0 - toxicity
        
        # Combined score
        timing_score = (
            0.4 * spread_score +
            0.3 * balance_score +
            0.3 * toxicity_score
        )
        
        metrics = {
            'depth_imbalance': imbalance,
            'spread_bps': rel_spread_bps,
            'spread_score': spread_score,
            'balance_score': balance_score,
            'toxicity': toxicity,
            'toxicity_score': toxicity_score,
            'timing_score': timing_score,
            'hidden_liquidity': hidden_liq['likely_hidden_liquidity'],
            'execution_favorable': timing_score > 0.6
        }
        
        return timing_score, metrics
    
    def recommend_execution_strategy(self, order_size: float, 
                                    snapshot: OrderbookSnapshot) -> Dict:
        """
        Recommend execution strategy based on orderbook analysis
        
        Args:
            order_size: Desired order size
            snapshot: Current orderbook snapshot
        
        Returns:
            Dictionary with execution recommendations
        """
        timing_score, metrics = self.get_execution_timing_score(snapshot)
        impact = self.estimate_market_impact(order_size, snapshot)
        
        # Determine strategy
        if timing_score > 0.7 and impact['liquidity_sufficient']:
            strategy = "IMMEDIATE"
            reason = "Favorable conditions, sufficient liquidity"
        elif timing_score > 0.5 and impact['impact_bps'] < 10:
            strategy = "VWAP"
            reason = "Good conditions, moderate liquidity"
        elif impact['impact_bps'] > 20:
            strategy = "TWAP"
            reason = "High impact, spread execution over time"
        else:
            strategy = "PATIENT"
            reason = "Wait for better conditions"
        
        return {
            'strategy': strategy,
            'reason': reason,
            'timing_score': timing_score,
            'estimated_impact_bps': impact['impact_bps'],
            'estimated_price': impact['estimated_price'],
            'metrics': metrics
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample orderbook snapshot
    snapshot = OrderbookSnapshot(
        timestamp=1234567890.0,
        bid_prices=[99.95, 99.94, 99.93, 99.92, 99.91],
        bid_sizes=[1000, 1500, 2000, 1200, 800],
        ask_prices=[100.00, 100.01, 100.02, 100.03, 100.04],
        ask_sizes=[800, 1200, 1800, 1000, 1500],
        last_price=99.98,
        volume=50000
    )
    
    # Initialize model
    intent_model = OrderbookIntentModelV73(
        depth_levels=5,
        imbalance_threshold=0.3,
        pressure_window=10
    )
    
    # Analyze orderbook
    print("\n=== Orderbook Analysis ===")
    timing_score, metrics = intent_model.get_execution_timing_score(snapshot)
    
    print(f"Execution Timing Score: {timing_score:.3f}")
    print(f"Depth Imbalance: {metrics['depth_imbalance']:.3f}")
    print(f"Spread: {metrics['spread_bps']:.2f} bps")
    print(f"Toxicity: {metrics['toxicity']:.3f}")
    print(f"Execution Favorable: {metrics['execution_favorable']}")
    
    # Test execution strategy
    print("\n=== Execution Strategy ===")
    order_size = 5000  # Buy 5000 shares
    recommendation = intent_model.recommend_execution_strategy(order_size, snapshot)
    
    print(f"Recommended Strategy: {recommendation['strategy']}")
    print(f"Reason: {recommendation['reason']}")
    print(f"Estimated Impact: {recommendation['estimated_impact_bps']:.2f} bps")
    print(f"Estimated Price: ${recommendation['estimated_price']:.2f}")
