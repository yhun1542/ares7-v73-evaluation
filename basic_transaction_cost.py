"""
Basic Transaction Cost Model for ARES-7 v73
Addresses P0 critical issue: No transaction cost modeling

This implements essential cost estimation to prevent strategy bleeding:
- Bid-ask spread costs
- Market impact estimation
- Slippage modeling
- Commission fees
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CostParameters:
    """Transaction cost parameters"""
    # Spread costs (basis points)
    spread_bps_liquid: float = 5.0  # Liquid stocks (S&P 500)
    spread_bps_normal: float = 10.0  # Normal stocks
    spread_bps_illiquid: float = 25.0  # Illiquid stocks
    
    # Slippage (basis points)
    slippage_bps_liquid: float = 5.0
    slippage_bps_normal: float = 10.0
    slippage_bps_illiquid: float = 20.0
    
    # Market impact (basis points per $1M traded)
    impact_bps_per_million: float = 2.0
    
    # Commission (per share or percentage)
    commission_per_share: float = 0.005  # $0.005 per share
    min_commission: float = 1.0  # Minimum $1 per trade
    
    # Liquidity thresholds (average daily volume)
    liquid_threshold: float = 5_000_000  # 5M shares
    normal_threshold: float = 1_000_000  # 1M shares


class BasicTransactionCostModel:
    """
    Basic transaction cost estimation model
    
    Estimates total cost of trading including:
    - Bid-ask spread
    - Market impact
    - Slippage
    - Commissions
    
    Based on industry-standard assumptions from Grok's recommendation:
    - Liquid instruments: 5-10 bps
    - Illiquid instruments: 20-50 bps
    """
    
    def __init__(self, params: Optional[CostParameters] = None):
        self.params = params or CostParameters()
        logger.info("Transaction Cost Model initialized")
    
    def classify_liquidity(self, avg_daily_volume: float) -> str:
        """Classify stock liquidity based on average daily volume"""
        if avg_daily_volume >= self.params.liquid_threshold:
            return "liquid"
        elif avg_daily_volume >= self.params.normal_threshold:
            return "normal"
        else:
            return "illiquid"
    
    def estimate_spread_cost(self, order_value: float, liquidity: str) -> float:
        """
        Estimate bid-ask spread cost
        
        Args:
            order_value: Dollar value of the order
            liquidity: 'liquid', 'normal', or 'illiquid'
        
        Returns:
            Spread cost in dollars
        """
        if liquidity == "liquid":
            spread_bps = self.params.spread_bps_liquid
        elif liquidity == "normal":
            spread_bps = self.params.spread_bps_normal
        else:
            spread_bps = self.params.spread_bps_illiquid
        
        cost = order_value * (spread_bps / 10000)
        return cost
    
    def estimate_slippage(self, order_value: float, liquidity: str, 
                         volatility: Optional[float] = None) -> float:
        """
        Estimate slippage cost
        
        Args:
            order_value: Dollar value of the order
            liquidity: 'liquid', 'normal', or 'illiquid'
            volatility: Optional volatility multiplier
        
        Returns:
            Slippage cost in dollars
        """
        if liquidity == "liquid":
            slippage_bps = self.params.slippage_bps_liquid
        elif liquidity == "normal":
            slippage_bps = self.params.slippage_bps_normal
        else:
            slippage_bps = self.params.slippage_bps_illiquid
        
        # Adjust for volatility if provided
        if volatility is not None and volatility > 0:
            slippage_bps *= (1 + volatility)
        
        cost = order_value * (slippage_bps / 10000)
        return cost
    
    def estimate_market_impact(self, order_value: float, 
                               avg_daily_volume: float,
                               avg_price: float) -> float:
        """
        Estimate market impact cost using simplified Almgren-Chriss
        
        Args:
            order_value: Dollar value of the order
            avg_daily_volume: Average daily trading volume (shares)
            avg_price: Average stock price
        
        Returns:
            Market impact cost in dollars
        """
        if avg_daily_volume == 0 or avg_price == 0:
            return 0.0
        
        # Calculate order size as fraction of daily volume
        order_shares = order_value / avg_price
        volume_fraction = order_shares / avg_daily_volume
        
        # Impact is proportional to square root of volume fraction
        # This is a simplified Almgren-Chriss model
        impact_multiplier = np.sqrt(volume_fraction)
        
        # Base impact per $1M traded
        millions_traded = order_value / 1_000_000
        base_impact = millions_traded * self.params.impact_bps_per_million
        
        # Total impact
        impact_bps = base_impact * impact_multiplier
        cost = order_value * (impact_bps / 10000)
        
        return cost
    
    def estimate_commission(self, shares: int, price: float) -> float:
        """
        Estimate commission cost
        
        Args:
            shares: Number of shares
            price: Price per share
        
        Returns:
            Commission cost in dollars
        """
        commission = shares * self.params.commission_per_share
        return max(commission, self.params.min_commission)
    
    def estimate_total_cost(self, symbol: str, order_value: float, 
                           price: float, shares: int,
                           avg_daily_volume: float,
                           volatility: Optional[float] = None) -> Dict[str, float]:
        """
        Estimate total transaction cost
        
        Args:
            symbol: Stock symbol
            order_value: Dollar value of the order
            price: Current price
            shares: Number of shares
            avg_daily_volume: Average daily volume
            volatility: Optional volatility measure
        
        Returns:
            Dictionary with cost breakdown
        """
        # Classify liquidity
        liquidity = self.classify_liquidity(avg_daily_volume)
        
        # Calculate individual cost components
        spread_cost = self.estimate_spread_cost(order_value, liquidity)
        slippage_cost = self.estimate_slippage(order_value, liquidity, volatility)
        impact_cost = self.estimate_market_impact(order_value, avg_daily_volume, price)
        commission = self.estimate_commission(shares, price)
        
        # Total cost
        total_cost = spread_cost + slippage_cost + impact_cost + commission
        cost_bps = (total_cost / order_value * 10000) if order_value > 0 else 0
        
        result = {
            'symbol': symbol,
            'order_value': order_value,
            'liquidity': liquidity,
            'spread_cost': spread_cost,
            'slippage_cost': slippage_cost,
            'impact_cost': impact_cost,
            'commission': commission,
            'total_cost': total_cost,
            'cost_bps': cost_bps,
            'cost_pct': cost_bps / 100
        }
        
        logger.debug(f"Transaction cost for {symbol}: ${total_cost:.2f} ({cost_bps:.1f} bps)")
        
        return result
    
    def adjust_order_for_costs(self, target_pnl: float, estimated_cost: float) -> float:
        """
        Adjust target P&L to account for transaction costs
        
        Args:
            target_pnl: Target profit/loss
            estimated_cost: Estimated transaction cost
        
        Returns:
            Adjusted target P&L
        """
        # Need to achieve target_pnl + costs to actually realize target_pnl
        adjusted_target = target_pnl + estimated_cost
        return adjusted_target


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize cost model
    cost_model = BasicTransactionCostModel()
    
    # Example trade: $50k of AAPL
    print("\n=== Liquid Stock Example (AAPL) ===")
    costs = cost_model.estimate_total_cost(
        symbol="AAPL",
        order_value=50000,
        price=180.0,
        shares=278,
        avg_daily_volume=50_000_000,  # 50M shares (very liquid)
        volatility=0.15
    )
    
    for key, value in costs.items():
        if isinstance(value, float):
            print(f"{key}: ${value:.2f}" if 'cost' in key or 'value' in key else f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Example trade: $20k of illiquid stock
    print("\n=== Illiquid Stock Example ===")
    costs = cost_model.estimate_total_cost(
        symbol="ILLIQ",
        order_value=20000,
        price=25.0,
        shares=800,
        avg_daily_volume=500_000,  # 500k shares (illiquid)
        volatility=0.30
    )
    
    for key, value in costs.items():
        if isinstance(value, float):
            print(f"{key}: ${value:.2f}" if 'cost' in key or 'value' in key else f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Cost impact on strategy
    print("\n=== Cost Impact Analysis ===")
    target_profit = 500
    estimated_cost = costs['total_cost']
    adjusted_target = cost_model.adjust_order_for_costs(target_profit, estimated_cost)
    print(f"Target profit: ${target_profit:.2f}")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Adjusted target (to achieve net profit): ${adjusted_target:.2f}")
    print(f"Required gross return: {(adjusted_target/20000)*100:.2f}%")
