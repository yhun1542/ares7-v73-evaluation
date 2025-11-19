"""
NSV (Noise-to-Signal Volatility) Model for ARES-7 v73
Mean Reversion Engine Component

This model estimates the noise-to-signal ratio in price movements
to identify mean reversion opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NSVVolatilityModel:
    """
    Noise-to-Signal Volatility Model
    
    Estimates the ratio of noise (random fluctuations) to signal (true price movement)
    in stock prices to identify mean reversion opportunities.
    
    Key concepts:
    - High NSV ratio → More noise → Better mean reversion opportunities
    - Low NSV ratio → More signal → Trend following may be better
    
    Based on microstructure theory and statistical arbitrage principles.
    """
    
    def __init__(self, 
                 lookback_short: int = 20,
                 lookback_long: int = 60,
                 noise_threshold: float = 0.6):
        """
        Initialize NSV model
        
        Args:
            lookback_short: Short-term lookback period for noise estimation
            lookback_long: Long-term lookback period for signal estimation
            noise_threshold: Threshold for high noise regime (0-1)
        """
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.noise_threshold = noise_threshold
        
        logger.info(f"NSV Model initialized: short={lookback_short}, "
                   f"long={lookback_long}, threshold={noise_threshold}")
    
    def estimate_realized_volatility(self, returns: np.ndarray, 
                                     window: int) -> np.ndarray:
        """
        Estimate realized volatility using rolling standard deviation
        
        Args:
            returns: Array of returns
            window: Rolling window size
        
        Returns:
            Array of realized volatility estimates
        """
        if len(returns) < window:
            return np.full(len(returns), np.nan)
        
        # Calculate rolling standard deviation
        realized_vol = pd.Series(returns).rolling(window=window).std().values
        
        # Annualize (assuming daily returns)
        realized_vol = realized_vol * np.sqrt(252)
        
        return realized_vol
    
    def estimate_noise_component(self, prices: np.ndarray, 
                                 window: int) -> np.ndarray:
        """
        Estimate noise component using bid-ask bounce proxy
        
        Uses first-order autocorrelation of returns as proxy for noise.
        Negative autocorrelation suggests bid-ask bounce (noise).
        
        Args:
            prices: Array of prices
            window: Rolling window size
        
        Returns:
            Array of noise estimates
        """
        if len(prices) < window + 1:
            return np.full(len(prices), np.nan)
        
        returns = np.diff(np.log(prices))
        
        # Calculate rolling autocorrelation
        noise = np.full(len(prices), np.nan)
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            if len(window_returns) > 1:
                # First-order autocorrelation
                autocorr = np.corrcoef(window_returns[:-1], window_returns[1:])[0, 1]
                # Negative autocorrelation indicates noise
                noise[i] = max(0, -autocorr)
        
        return noise
    
    def estimate_signal_component(self, prices: np.ndarray, 
                                  window: int) -> np.ndarray:
        """
        Estimate signal component using trend strength
        
        Uses moving average deviation and R-squared of linear trend.
        
        Args:
            prices: Array of prices
            window: Rolling window size
        
        Returns:
            Array of signal estimates
        """
        if len(prices) < window:
            return np.full(len(prices), np.nan)
        
        signal = np.full(len(prices), np.nan)
        
        for i in range(window, len(prices)):
            window_prices = prices[i-window:i]
            
            # Fit linear trend
            x = np.arange(len(window_prices))
            coeffs = np.polyfit(x, window_prices, 1)
            trend = np.polyval(coeffs, x)
            
            # Calculate R-squared
            ss_res = np.sum((window_prices - trend) ** 2)
            ss_tot = np.sum((window_prices - np.mean(window_prices)) ** 2)
            
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
                signal[i] = max(0, r_squared)
            else:
                signal[i] = 0
        
        return signal
    
    def calculate_nsv_ratio(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate Noise-to-Signal Volatility ratio
        
        Args:
            prices: Array of prices
        
        Returns:
            Dictionary with NSV components and ratio
        """
        # Estimate components
        noise = self.estimate_noise_component(prices, self.lookback_short)
        signal = self.estimate_signal_component(prices, self.lookback_long)
        
        # Calculate NSV ratio
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        nsv_ratio = noise / (signal + epsilon)
        
        # Normalize to 0-1 range using sigmoid
        nsv_ratio_normalized = 1 / (1 + np.exp(-nsv_ratio))
        
        # Estimate volatility
        returns = np.diff(np.log(prices))
        realized_vol = self.estimate_realized_volatility(
            returns, self.lookback_short
        )
        
        return {
            'noise': noise,
            'signal': signal,
            'nsv_ratio': nsv_ratio,
            'nsv_ratio_normalized': nsv_ratio_normalized,
            'realized_volatility': realized_vol,
            'high_noise_regime': nsv_ratio_normalized > self.noise_threshold
        }
    
    def get_mean_reversion_score(self, prices: np.ndarray) -> Tuple[float, Dict]:
        """
        Get mean reversion opportunity score for current price
        
        Args:
            prices: Array of historical prices
        
        Returns:
            (score, metrics) where score is 0-1 and metrics is dict of components
        """
        if len(prices) < self.lookback_long:
            logger.warning(f"Insufficient data: {len(prices)} < {self.lookback_long}")
            return 0.0, {}
        
        # Calculate NSV components
        nsv_data = self.calculate_nsv_ratio(prices)
        
        # Get latest values
        latest_nsv = nsv_data['nsv_ratio_normalized'][-1]
        latest_vol = nsv_data['realized_volatility'][-1]
        latest_noise = nsv_data['noise'][-1]
        latest_signal = nsv_data['signal'][-1]
        
        # Mean reversion score is higher when:
        # 1. NSV ratio is high (more noise)
        # 2. Volatility is elevated (more opportunities)
        # 3. Signal is weak (no strong trend)
        
        # Normalize volatility (assume typical range 0.1-0.5)
        vol_score = min(1.0, max(0.0, (latest_vol - 0.1) / 0.4))
        
        # Combine scores
        mean_reversion_score = (
            0.5 * latest_nsv +  # 50% weight on NSV ratio
            0.3 * vol_score +    # 30% weight on volatility
            0.2 * latest_noise   # 20% weight on noise level
        )
        
        metrics = {
            'nsv_ratio': latest_nsv,
            'realized_vol': latest_vol,
            'noise_level': latest_noise,
            'signal_strength': latest_signal,
            'vol_score': vol_score,
            'mean_reversion_score': mean_reversion_score,
            'regime': 'high_noise' if latest_nsv > self.noise_threshold else 'low_noise'
        }
        
        return mean_reversion_score, metrics
    
    def estimate_reversion_target(self, prices: np.ndarray, 
                                  window: int = 20) -> Tuple[float, float]:
        """
        Estimate mean reversion target price and confidence
        
        Args:
            prices: Array of historical prices
            window: Lookback window for mean estimation
        
        Returns:
            (target_price, confidence) where confidence is 0-1
        """
        if len(prices) < window:
            return prices[-1], 0.0
        
        recent_prices = prices[-window:]
        
        # Use volume-weighted moving average as target
        # (simplified - in practice would use actual volume data)
        target = np.mean(recent_prices)
        
        # Confidence based on price stability around mean
        std = np.std(recent_prices)
        mean_abs_dev = np.mean(np.abs(recent_prices - target))
        
        # Lower deviation = higher confidence
        confidence = 1.0 / (1.0 + mean_abs_dev / target)
        
        return target, confidence


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample price data
    np.random.seed(42)
    n_points = 100
    
    # Simulate mean-reverting process with noise
    prices = np.zeros(n_points)
    prices[0] = 100
    mean_price = 100
    reversion_speed = 0.1
    noise_level = 0.5
    
    for i in range(1, n_points):
        # Mean reversion component
        reversion = reversion_speed * (mean_price - prices[i-1])
        # Noise component
        noise = np.random.normal(0, noise_level)
        # Update price
        prices[i] = prices[i-1] + reversion + noise
    
    # Initialize model
    nsv_model = NSVVolatilityModel(
        lookback_short=20,
        lookback_long=60,
        noise_threshold=0.6
    )
    
    # Calculate NSV metrics
    print("\n=== NSV Analysis ===")
    score, metrics = nsv_model.get_mean_reversion_score(prices)
    
    print(f"Mean Reversion Score: {score:.3f}")
    print(f"NSV Ratio: {metrics['nsv_ratio']:.3f}")
    print(f"Realized Volatility: {metrics['realized_vol']:.3f}")
    print(f"Noise Level: {metrics['noise_level']:.3f}")
    print(f"Signal Strength: {metrics['signal_strength']:.3f}")
    print(f"Regime: {metrics['regime']}")
    
    # Estimate reversion target
    target, confidence = nsv_model.estimate_reversion_target(prices)
    print(f"\nReversion Target: ${target:.2f}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Current Price: ${prices[-1]:.2f}")
    print(f"Distance to Target: {((target - prices[-1]) / prices[-1] * 100):.2f}%")
