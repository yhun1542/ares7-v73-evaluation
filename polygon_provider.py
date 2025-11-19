"""
Polygon.io Data Provider for ARES-7 v73
Replaces synthetic data with real market data

Provides:
- OHLCV data
- Options data (for GEX calculation)
- Real-time and historical data
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


class PolygonDataProvider:
    """
    Polygon.io API data provider
    
    Provides real market data to replace synthetic data:
    - Stock OHLCV
    - Options data for gamma exposure
    - Aggregates and snapshots
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polygon data provider
        
        Args:
            api_key: Polygon API key (or from environment)
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            logger.warning("No Polygon API key found. Set POLYGON_API_KEY environment variable.")
        
        self.base_url = "https://api.polygon.io"
        self.rate_limit_delay = 0.1  # 100ms between requests (free tier: 5 req/min)
        self.last_request_time = 0
        
        logger.info("Polygon Data Provider initialized")
    
    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make API request with error handling
        
        Args:
            endpoint: API endpoint
            params: Query parameters
        
        Returns:
            JSON response
        """
        self._rate_limit()
        
        if params is None:
            params = {}
        params['apiKey'] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {}
    
    def get_ohlcv(self, symbol: str, 
                   start_date: str,
                   end_date: str,
                   timespan: str = 'day',
                   multiplier: int = 1) -> pd.DataFrame:
        """
        Get OHLCV data for a symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timespan: 'minute', 'hour', 'day', 'week', 'month'
            multiplier: Size of timespan (e.g., 5 for 5-minute bars)
        
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        
        data = self._make_request(endpoint, {'adjusted': 'true', 'sort': 'asc'})
        
        if 'results' not in data or not data['results']:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data['results'])
        
        # Rename columns to standard names
        df = df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'transactions'
        })
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        logger.info(f"Retrieved {len(df)} bars for {symbol}")
        
        return df
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for a symbol
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Latest price or None
        """
        endpoint = f"/v2/last/trade/{symbol}"
        
        data = self._make_request(endpoint)
        
        if 'results' in data and data['results']:
            return data['results'].get('p')
        
        return None
    
    def get_options_chain(self, underlying: str, 
                         expiration_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get options chain for gamma exposure calculation
        
        Args:
            underlying: Underlying stock symbol
            expiration_date: Expiration date (YYYY-MM-DD), or None for all
        
        Returns:
            DataFrame with options data
        """
        # Get options contracts
        endpoint = f"/v3/reference/options/contracts"
        
        params = {
            'underlying_ticker': underlying,
            'limit': 1000
        }
        
        if expiration_date:
            params['expiration_date'] = expiration_date
        
        data = self._make_request(endpoint, params)
        
        if 'results' not in data or not data['results']:
            logger.warning(f"No options data for {underlying}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data['results'])
        
        logger.info(f"Retrieved {len(df)} options contracts for {underlying}")
        
        return df
    
    def calculate_gamma_exposure(self, symbol: str) -> Dict:
        """
        Calculate gamma exposure (GEX) from options data
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with GEX metrics
        """
        # Get current price
        current_price = self.get_latest_price(symbol)
        
        if current_price is None:
            logger.warning(f"Could not get price for {symbol}")
            return {'total_gamma': 0, 'call_gamma': 0, 'put_gamma': 0}
        
        # Get options chain
        options = self.get_options_chain(symbol)
        
        if options.empty:
            return {'total_gamma': 0, 'call_gamma': 0, 'put_gamma': 0}
        
        # Simplified GEX calculation
        # In production, would use Black-Scholes to calculate actual gamma
        
        call_gamma = 0
        put_gamma = 0
        
        for _, option in options.iterrows():
            strike = option.get('strike_price', 0)
            contract_type = option.get('contract_type', '')
            
            # Simplified gamma estimation (peaks at ATM)
            moneyness = abs(current_price - strike) / current_price
            gamma_estimate = np.exp(-10 * moneyness**2)  # Gaussian approximation
            
            if contract_type == 'call':
                call_gamma += gamma_estimate
            elif contract_type == 'put':
                put_gamma += gamma_estimate
        
        total_gamma = call_gamma - put_gamma  # Net gamma exposure
        
        result = {
            'symbol': symbol,
            'current_price': current_price,
            'total_gamma': total_gamma,
            'call_gamma': call_gamma,
            'put_gamma': put_gamma,
            'gamma_ratio': call_gamma / (put_gamma + 1e-8)
        }
        
        logger.info(f"GEX for {symbol}: {total_gamma:.2f}")
        
        return result
    
    def get_market_snapshot(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get market snapshot for multiple symbols
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dictionary mapping symbol to snapshot data
        """
        snapshots = {}
        
        for symbol in symbols:
            endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
            data = self._make_request(endpoint)
            
            if 'ticker' in data:
                ticker_data = data['ticker']
                snapshots[symbol] = {
                    'price': ticker_data.get('lastTrade', {}).get('p'),
                    'volume': ticker_data.get('day', {}).get('v'),
                    'change_pct': ticker_data.get('todaysChangePerc'),
                    'high': ticker_data.get('day', {}).get('h'),
                    'low': ticker_data.get('day', {}).get('l'),
                }
        
        return snapshots


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize provider
    provider = PolygonDataProvider()
    
    # Test OHLCV data
    print("\n=== Testing OHLCV Data ===")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    df = provider.get_ohlcv('AAPL', start_date, end_date)
    
    if not df.empty:
        print(f"Retrieved {len(df)} days of data")
        print(f"Latest close: ${df['close'].iloc[-1]:.2f}")
        print(f"30-day return: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.2f}%")
    
    # Test GEX calculation
    print("\n=== Testing Gamma Exposure ===")
    gex = provider.calculate_gamma_exposure('AAPL')
    print(f"Total Gamma: {gex['total_gamma']:.2f}")
    print(f"Call Gamma: {gex['call_gamma']:.2f}")
    print(f"Put Gamma: {gex['put_gamma']:.2f}")
