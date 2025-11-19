"""
Production-ready API endpoint management with proper error handling.
Removes all hardcoded fallback values and implements robust retry logic.
"""

import os
import time
import logging
import json
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
from functools import wraps
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib
import hmac

# Configure structured logging
logger = logging.getLogger(__name__)

class APIKeyManager:
    """Centralized API key management with validation."""
    
    def __init__(self):
        self.keys = {}
        self.validated = {}
        self._load_keys()
    
    def _load_keys(self):
        """Load API keys from environment or config file."""
        # Primary: Environment variables
        self.keys = {
            'ALPHA_VANTAGE': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'POLYGON': os.getenv('POLYGON_API_KEY'),
            'IEX_CLOUD': os.getenv('IEX_CLOUD_API_KEY'),
            'FRED': os.getenv('FRED_API_KEY'),
            'QUANDL': os.getenv('QUANDL_API_KEY'),
            'NEWSAPI': os.getenv('NEWSAPI_KEY'),
            'TWITTER': os.getenv('TWITTER_BEARER_TOKEN'),
            'REDDIT': {
                'client_id': os.getenv('REDDIT_CLIENT_ID'),
                'client_secret': os.getenv('REDDIT_CLIENT_SECRET')
            },
            'CBOE': os.getenv('CBOE_API_KEY'),
            'NYSE': os.getenv('NYSE_API_KEY')
        }
        
        # Fallback: Config file
        config_path = os.getenv('API_CONFIG_PATH', 'config/api_keys.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    for key, value in config.items():
                        if not self.keys.get(key):
                            self.keys[key] = value
            except Exception as e:
                logger.error(f"Failed to load API config: {e}")
    
    def validate_key(self, service: str) -> bool:
        """Validate API key exists and is properly formatted."""
        if service in self.validated:
            return self.validated[service]
        
        key = self.keys.get(service)
        if not key:
            logger.error(f"Missing API key for {service}")
            self.validated[service] = False
            return False
        
        # Service-specific validation
        validators = {
            'ALPHA_VANTAGE': lambda k: len(k) >= 16,
            'POLYGON': lambda k: k.startswith('_'),
            'IEX_CLOUD': lambda k: k.startswith('pk_') or k.startswith('sk_'),
            'FRED': lambda k: len(k) == 32,
            'QUANDL': lambda k: len(k) >= 20,
            'NEWSAPI': lambda k: len(k) == 32,
            'TWITTER': lambda k: k.startswith('Bearer '),
            'REDDIT': lambda k: isinstance(k, dict) and 'client_id' in k
        }
        
        validator = validators.get(service, lambda k: bool(k))
        is_valid = validator(key)
        
        if not is_valid:
            logger.error(f"Invalid API key format for {service}")
        
        self.validated[service] = is_valid
        return is_valid
    
    def get_key(self, service: str) -> Optional[Any]:
        """Get validated API key."""
        if self.validate_key(service):
            return self.keys[service]
        return None

class APIEndpoint:
    """Base class for API endpoint management."""
    
    def __init__(self, service_name: str, base_url: str, 
                 rate_limit: int = 5, timeout: int = 30):
        self.service_name = service_name
        self.base_url = base_url
        self.rate_limit = rate_limit  # requests per second
        self.timeout = timeout
        self.last_request_time = 0
        self.request_count = 0
        self.error_count = 0
        self.circuit_open = False
        self.circuit_open_until = None
        
        # Setup session with retry strategy
        self.session = self._create_session()
        
        # API key management
        self.key_manager = APIKeyManager()
        self.api_key = self.key_manager.get_key(service_name)
        
        if not self.api_key:
            raise ValueError(f"No valid API key for {service_name}")
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _rate_limit_check(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < (1.0 / self.rate_limit):
            sleep_time = (1.0 / self.rate_limit) - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.circuit_open:
            return True
        
        if datetime.now() > self.circuit_open_until:
            logger.info(f"Circuit breaker closed for {self.service_name}")
            self.circuit_open = False
            self.error_count = 0
            return True
        
        return False
    
    def _trip_circuit_breaker(self):
        """Trip the circuit breaker."""
        self.circuit_open = True
        self.circuit_open_until = datetime.now() + timedelta(minutes=5)
        logger.error(f"Circuit breaker opened for {self.service_name} until {self.circuit_open_until}")
    
    def request(self, endpoint: str, params: Dict = None, 
                method: str = 'GET', **kwargs) -> Optional[Dict]:
        """Make API request with full error handling."""
        
        # Check circuit breaker
        if not self._check_circuit_breaker():
            logger.warning(f"Circuit breaker open for {self.service_name}")
            return None
        
        # Rate limiting
        self._rate_limit_check()
        
        # Build URL
        url = f"{self.base_url}/{endpoint}"
        
        # Add API key to params or headers based on service
        if self.api_key:
            if self.service_name in ['ALPHA_VANTAGE', 'POLYGON', 'FRED']:
                params = params or {}
                params['apikey'] = self.api_key
            elif self.service_name == 'TWITTER':
                kwargs['headers'] = kwargs.get('headers', {})
                kwargs['headers']['Authorization'] = self.api_key
        
        # Exponential backoff retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=self.timeout,
                    **kwargs
                )
                
                response.raise_for_status()
                
                # Reset error count on success
                self.error_count = 0
                
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited on {self.service_name}, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                elif response.status_code == 401:  # Unauthorized
                    logger.error(f"Authentication failed for {self.service_name}")
                    self._trip_circuit_breaker()
                    return None
                else:
                    logger.error(f"HTTP error for {self.service_name}: {e}")
                    
            except requests.exceptions.Timeout:
                logger.error(f"Timeout for {self.service_name} (attempt {attempt + 1})")
                
            except requests.exceptions.ConnectionError:
                logger.error(f"Connection error for {self.service_name}")
                
            except Exception as e:
                logger.error(f"Unexpected error for {self.service_name}: {e}")
            
            # Exponential backoff
            if attempt < max_retries - 1:
                backoff_time = (2 ** attempt) + (time.time() % 1)  # Add jitter
                logger.info(f"Retrying {self.service_name} in {backoff_time:.2f}s")
                time.sleep(backoff_time)
            
            self.error_count += 1
        
        # Trip circuit breaker after max failures
        if self.error_count >= 5:
            self._trip_circuit_breaker()
        
        return None

class MarketDataAPI(APIEndpoint):
    """Specialized market data API handler."""
    
    def __init__(self, service_name: str, base_url: str):
        super().__init__(service_name, base_url)
        self.cache = {}
        self.cache_ttl = 60  # seconds
    
    def get_quote(self, symbol: str, use_cache: bool = True) -> Optional[Dict]:
        """Get stock quote with caching."""
        cache_key = f"quote_{symbol}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"Using cached quote for {symbol}")
                return cached_data
        
        # Fetch fresh data
        data = self.request(f"quote/{symbol}")
        
        if data:
            self.cache[cache_key] = (data, time.time())
        
        return data
    
    def get_options_chain(self, symbol: str, expiry: str = None) -> Optional[Dict]:
        """Get options chain data."""
        params = {'symbol': symbol}
        if expiry:
            params['expiry'] = expiry
        
        return self.request('options/chain', params=params)

class SentimentDataAPI(APIEndpoint):
    """Specialized sentiment data API handler."""
    
    def __init__(self, service_name: str, base_url: str):
        super().__init__(service_name, base_url, rate_limit=2)  # Lower rate limit
    
    def get_sentiment(self, query: str, lookback_hours: int = 24) -> Optional[Dict]:
        """Get sentiment data with proper error handling."""
        params = {
            'q': query,
            'from': (datetime.now() - timedelta(hours=lookback_hours)).isoformat(),
            'to': datetime.now().isoformat()
        }
        
        data = self.request('sentiment/analyze', params=params)
        
        if not data:
            logger.warning(f"No sentiment data available for {query}")
            return None
        
        return data

class APIOrchestrator:
    """Orchestrate multiple API endpoints with fallback strategies."""
    
    def __init__(self):
        self.endpoints = {}
        self._initialize_endpoints()
        self.health_status = {}
    
    def _initialize_endpoints(self):
        """Initialize all API endpoints."""
        configs = {
            'ALPHA_VANTAGE': ('https://www.alphavantage.co/query', MarketDataAPI),
            'POLYGON': ('https://api.polygon.io/v2', MarketDataAPI),
            'IEX_CLOUD': ('https://cloud.iexapis.com/stable', MarketDataAPI),
            'NEWSAPI': ('https://newsapi.org/v2', SentimentDataAPI),
            'TWITTER': ('https://api.twitter.com/2', SentimentDataAPI)
        }
        
        for service, (url, api_class) in configs.items():
            try:
                self.endpoints[service] = api_class(service, url)
                self.health_status[service] = 'healthy'
                logger.info(f"Initialized {service} endpoint")
            except ValueError as e:
                logger.error(f"Failed to initialize {service}: {e}")
                self.health_status[service] = 'unavailable'
    
    def get_market_data(self, symbol: str, data_type: str = 'quote') -> Optional[Dict]:
        """Get market data with automatic fallback."""
        # Priority order for market data sources
        priority = ['POLYGON', 'IEX_CLOUD', 'ALPHA_VANTAGE']
        
        for service in priority:
            if service not in self.endpoints:
                continue
            
            if self.health_status.get(service) == 'unavailable':
                continue
            
            try:
                if data_type == 'quote':
                    data = self.endpoints[service].get_quote(symbol)
                elif data_type == 'options':
                    data = self.endpoints[service].get_options_chain(symbol)
                else:
                    data = None
                
                if data:
                    logger.info(f"Successfully fetched {data_type} for {symbol} from {service}")
                    return data
                    
            except Exception as e:
                logger.error(f"Failed to get {data_type} from {service}: {e}")
                self.health_status[service] = 'degraded'
        
        # No fallback values - fail explicitly
        logger.error(f"All API sources failed for {symbol} {data_type}")
        raise RuntimeError(f"Unable to fetch {data_type} for {symbol} from any source")
    
    def get_sentiment_data(self, query: str) -> Optional[Dict]:
        """Get sentiment data with fallback."""
        priority = ['TWITTER', 'NEWSAPI']
        
        for service in priority:
            if service not in self.endpoints:
                continue
            
            try:
                data = self.endpoints[service].get_sentiment(query)
                if data:
                    return data
            except Exception as e:
                logger.error(f"Failed to get sentiment from {service}: {e}")
        
        # No fallback - return None to indicate no data
        logger.warning(f"No sentiment data available for {query}")
        return None
    
    def health_check(self) -> Dict[str, str]:
        """Check health of all endpoints."""
        for service, endpoint in self.endpoints.items():
            try:
                # Simple connectivity test
                test_data = endpoint.request('status', method='HEAD')
                self.health_status[service] = 'healthy' if test_data is not None else 'degraded'
            except Exception:
                self.health_status[service] = 'unhealthy'
        
        return self.health_status

# Global orchestrator instance
api_orchestrator = APIOrchestrator()

def get_market_data_safe(symbol: str, data_type: str = 'quote') -> Optional[Dict]:
    """Safe wrapper for market data retrieval."""
    try:
        return api_orchestrator.get_market_data(symbol, data_type)
    except RuntimeError as e:
        logger.critical(f"Critical: {e}")
        # Alert monitoring system
        send_alert(f"Market data failure: {e}")
        return None

def send_alert(message: str):
    """Send alert to monitoring system."""
    # Implement your alerting logic here
    logger.critical(f"ALERT: {message}")
    # Could integrate with PagerDuty, Slack, email, etc.

if __name__ == "__main__":
    # Test the implementation
    logging.basicConfig(level=logging.INFO)
    
    # Health check
    health = api_orchestrator.health_check()
    print(f"API Health Status: {health}")
    
    # Test market data
    try:
        quote = get_market_data_safe('AAPL')
        print(f"AAPL Quote: {quote}")
    except Exception as e:
        print(f"Failed to get quote: {e}")

"""
Comprehensive Test Suite for Trading System
Includes unit tests, integration tests, and performance tests
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional
import json
import tempfile
import os
from decimal import Decimal
import random
import string

# Import modules to test
from trading_system import TradingSystem
from risk_manager import RiskManager, PositionManager, RiskMetrics
from performance_monitor import PerformanceMonitor, MetricsCollector
from circuit_breaker import CircuitBreaker, CircuitBreakerState
from broker_resilience import BrokerConnectionPool, ConnectionManager
from backtesting_framework import BacktestEngine, BacktestResult
from logging_config import setup_logging, get_logger

# Test configuration
TEST_CONFIG = {
    "test_mode": True,
    "mock_broker": True,
    "test_database": "test_trading.db",
    "test_log_file": "test_trading.log"
}

# Fixtures
@pytest.fixture
def mock_broker():
    """Create mock broker for testing"""
    broker = Mock()
    broker.connect = AsyncMock(return_value=True)
    broker.disconnect = AsyncMock(return_value=True)
    broker.place_order = AsyncMock(return_value={"order_id": "TEST123", "status": "FILLED"})
    broker.get_positions = AsyncMock(return_value=[])
    broker.get_balance = AsyncMock(return_value={"cash": 100000, "equity": 100000})
    broker.get_market_data = AsyncMock(return_value={"price": 100.0, "volume": 1000})
    return broker

@pytest.fixture
def mock_data_generator():
    """Generate mock market data for testing"""
    def generate_ohlcv(symbol: str, days: int = 30) -> pd.DataFrame:
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        data = []
        price = 100.0
        
        for date in dates:
            # Random walk for price
            change = np.random.normal(0, 2)
            price *= (1 + change/100)
            
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            close = price
            volume = int(np.random.uniform(1000000, 5000000))
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    return generate_ohlcv

@pytest.fixture
def risk_manager_fixture():
    """Create RiskManager instance for testing"""
    config = {
        "max_position_size": 10000,
        "max_daily_loss": 5000,
        "max_leverage": 2.0,
        "risk_per_trade": 0.02
    }
    return RiskManager(config)

@pytest.fixture
def performance_monitor_fixture():
    """Create PerformanceMonitor instance for testing"""
    return PerformanceMonitor()

@pytest.fixture
async def trading_system_fixture(mock_broker):
    """Create TradingSystem instance for testing"""
    config = {
        "broker": mock_broker,
        "symbols": ["AAPL", "GOOGL"],
        "strategy": "test_strategy"
    }
    system = TradingSystem(config)
    await system.initialize()
    return system

# Mock Data Generators
class MockDataGenerator:
    """Generate various types of mock data for testing"""
    
    @staticmethod
    def generate_trade(symbol: str = "AAPL") -> Dict[str, Any]:
        """Generate mock trade data"""
        return {
            "trade_id": ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)),
            "symbol": symbol,
            "side": random.choice(["BUY", "SELL"]),
            "quantity": random.randint(1, 100),
            "price": round(random.uniform(100, 200), 2),
            "timestamp": datetime.now().isoformat(),
            "status": "FILLED",
            "commission": round(random.uniform(0.5, 2.0), 2)
        }
    
    @staticmethod
    def generate_position(symbol: str = "AAPL") -> Dict[str, Any]:
        """Generate mock position data"""
        quantity = random.randint(-100, 100)
        entry_price = round(random.uniform(100, 200), 2)
        current_price = entry_price * (1 + random.uniform(-0.1, 0.1))
        
        return {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": round(current_price, 2),
            "unrealized_pnl": round((current_price - entry_price) * quantity, 2),
            "realized_pnl": round(random.uniform(-100, 100), 2)
        }
    
    @staticmethod
    def generate_order(symbol: str = "AAPL") -> Dict[str, Any]:
        """Generate mock order data"""
        return {
            "order_id": ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)),
            "symbol": symbol,
            "side": random.choice(["BUY", "SELL"]),
            "order_type": random.choice(["MARKET", "LIMIT", "STOP"]),
            "quantity": random.randint(1, 100),
            "price": round(random.uniform(100, 200), 2),
            "status": random.choice(["PENDING", "FILLED", "CANCELLED", "REJECTED"]),
            "timestamp": datetime.now().isoformat()
        }

# Unit Tests
class TestRiskManager:
    """Test suite for RiskManager"""
    
    def test_position_sizing(self, risk_manager_fixture):
        """Test position sizing calculation"""
        account_value = 100000
        risk_per_trade = 0.02
        stop_loss_pct = 0.05
        
        position_size = risk_manager_fixture.calculate_position_size(
            account_value, risk_per_trade, stop_loss_pct
        )
        
        expected_size = (account_value * risk_per_trade) / stop_loss_pct
        assert abs(position_size - expected_size) < 0.01
    
    def test_max_position_limit(self, risk_manager_fixture):
        """Test maximum position size enforcement"""
        large_position = 20000  # Exceeds max_position_size
        is_valid = risk_manager_fixture.validate_position_size(large_position)
        assert not is_valid
    
    def test_leverage_calculation(self, risk_manager_fixture):
        """Test leverage calculation"""
        positions = [
            {"symbol": "AAPL", "value": 50000},
            {"symbol": "GOOGL", "value": 30000}
        ]
        account_value = 100000
        
        leverage = risk_manager_fixture.calculate_leverage(positions, account_value)
        expected_leverage = 80000 / 100000
        assert abs(leverage - expected_leverage) < 0.01
    
    def test_daily_loss_limit(self, risk_manager_fixture):
        """Test daily loss limit enforcement"""
        risk_manager_fixture.daily_pnl = -6000  # Exceeds max_daily_loss
        can_trade = risk_manager_fixture.can_trade()
        assert not can_trade
    
    @pytest.mark.asyncio
    async def test_risk_metrics_calculation(self, risk_manager_fixture):
        """Test risk metrics calculation"""
        trades = [MockDataGenerator.generate_trade() for _ in range(10)]
        metrics = await risk_manager_fixture.calculate_risk_metrics(trades)
        
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics

class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor"""
    
    @pytest.mark.asyncio
    async def test_metric_recording(self, performance_monitor_fixture):
        """Test metric recording functionality"""
        await performance_monitor_fixture.record_metric("latency", 10.5)
        await performance_monitor_fixture.record_metric("latency", 12.3)
        
        stats = await performance_monitor_fixture.get_metric_stats("latency")
        assert stats["count"] == 2
        assert stats["mean"] == pytest.approx(11.4, 0.1)
    
    @pytest.mark.asyncio
    async def test_alert_triggering(self, performance_monitor_fixture):
        """Test alert triggering on threshold breach"""
        performance_monitor_fixture.set_threshold("cpu_usage", 80)
        
        alerts = []
        performance_monitor_fixture.on_alert = lambda alert: alerts.append(alert)
        
        await performance_monitor_fixture.record_metric("cpu_usage", 85)
        assert len(alerts) == 1
        assert alerts[0]["metric"] == "cpu_usage"
    
    @pytest.mark.asyncio
    async def test_performance_report_generation(self, performance_monitor_fixture):
        """Test performance report generation"""
        # Record various metrics
        for _ in range(100):
            await performance_monitor_fixture.record_metric(
                "execution_time", 
                random.uniform(1, 10)
            )
        
        report = await performance_monitor_fixture.generate_report()
        assert "execution_time" in report
        assert "summary" in report
        assert "timestamp" in report

class TestCircuitBreaker:
    """Test suite for CircuitBreaker"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions"""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1,
            expected_exception=Exception
        )
        
        # Initial state should be CLOSED
        assert breaker.state == CircuitBreakerState.CLOSED
        
        # Simulate failures
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(self._failing_function)
        
        # Should be OPEN after threshold
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Should be HALF_OPEN
        assert breaker.state == CircuitBreakerState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery"""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=1,
            expected_exception=Exception
        )
        
        # Trigger circuit breaker
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(self._failing_function)
        
        # Wait and test recovery
        await asyncio.sleep(1.1)
        
        # Successful call should close circuit
        result = await breaker.call(self._successful_function)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED
    
    async def _failing_function(self):
        """Helper function that always fails"""
        raise Exception("Test failure")
    
    async def _successful_function(self):
        """Helper function that always succeeds"""
        return "success"

class TestBrokerResilience:
    """Test suite for Broker Resilience"""
    
    @pytest.mark.asyncio
    async def test_connection_pool(self, mock_broker):
        """Test connection pool management"""
        pool = BrokerConnectionPool(
            broker_class=type(mock_broker),
            pool_size=5,
            max_retries=3
        )
        
        await pool.initialize()
        
        # Test connection acquisition
        conn = await pool.acquire()
        assert conn is not None
        
        # Test connection release
        await pool.release(conn)
        
        # Test pool statistics
        stats = pool.get_stats()
        assert stats["total_connections"] == 5
        assert stats["available_connections"] >= 4
    
    @pytest.mark.asyncio
    async def test_auto_reconnection(self, mock_broker):
        """Test automatic reconnection with exponential backoff"""
        manager = ConnectionManager(mock_broker)
        
        # Simulate connection failure
        mock_broker.connect = AsyncMock(side_effect=[False, False, True])
        
        connected = await manager.connect_with_retry()
        assert connected
        assert mock_broker.connect.call_count == 3
    
    @pytest.mark.asyncio
    async def test_failover_logic(self):
        """Test failover between brokers"""
        primary_broker = Mock()
        backup_broker = Mock()
        
        primary_broker.connect = AsyncMock(return_value=False)
        backup_broker.connect = AsyncMock(return_value=True)
        
        manager = ConnectionManager([primary_broker, backup_broker])
        
        connected = await manager.connect_with_failover()
        assert connected
        assert backup_broker.connect.called

# Integration Tests
class TestIntegration:
    """Integration tests for complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_trade_flow(self, trading_system_fixture, mock_broker):
        """Test complete trade flow from signal to execution"""
        system = trading_system_fixture
        
        # Generate trading signal
        signal = {
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 100,
            "confidence": 0.8
        }
        
        # Execute trade
        result = await system.execute_signal(signal)
        
        assert result["status"] == "SUCCESS"
        assert mock_broker.place_order.called
    
    @pytest.mark.asyncio
    async def test_risk_management_integration(self, trading_system_fixture, risk_manager_fixture):
        """Test risk management integration with trading system"""
        system = trading_system_fixture
        system.risk_manager = risk_manager_fixture
        
        # Create high-risk signal
        signal = {
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 100000,  # Very large position
            "confidence": 0.5
        }
        
        # Should be rejected by risk manager
        result = await system.execute_signal(signal)
        assert result["status"] == "REJECTED"
        assert "risk" in result["reason"].lower()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, trading_system_fixture, performance_monitor_fixture):
        """Test performance monitoring integration"""
        system = trading_system_fixture
        system.performance_monitor = performance_monitor_fixture
        
        # Execute multiple trades
        for _ in range(10):
            signal = {
                "symbol": random.choice(["AAPL", "GOOGL"]),
                "action": random.choice(["BUY", "SELL"]),
                "quantity": random.randint(10, 100),
                "confidence": random.uniform(0.5, 1.0)
            }
            await system.execute_signal(signal)
        
        # Check performance metrics
        metrics = await performance_monitor_fixture.get_all_metrics()
        assert len(metrics) > 0

# Performance Tests
class TestPerformance:
    """Performance and load tests"""
    
    @pytest.mark.asyncio
    async def test_high_frequency_trading_performance(self, trading_system_fixture):
        """Test system performance under high-frequency trading conditions"""
        system = trading_system_fixture
        num_trades = 1000
        
        start_time = datetime.now()
        
        tasks = []
        for _ in range(num_trades):
            signal = {
                "symbol": random.choice(["AAPL", "GOOGL"]),
                "action": random.choice(["BUY", "SELL"]),
                "quantity": random.randint(1, 10),
                "confidence": random.uniform(0.5, 1.0)
            }
            tasks.append(system.execute_signal(signal))
        
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        trades_per_second = num_trades / duration
        assert trades_per_second > 100  # Should handle at least 100 trades/second
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, trading_system_fixture):
        """Test memory usage under load"""
        import psutil
        import os
        
        system = trading_system_fixture
        process = psutil.Process(os.getpid())
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate load
        for _ in range(10000):
            data = MockDataGenerator.generate_trade()
            await system.process_trade(data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 100  # Should not increase by more than 100MB

# Backtesting Tests
class TestBacktesting:
    """Test suite for backtesting framework"""
    
    @pytest.mark.asyncio
    async def test_backtest_execution(self, mock_data_generator):
        """Test backtest execution"""
        engine = BacktestEngine()
        
        # Generate test data
        data = mock_data_generator("AAPL", days=100)
        
        # Define simple strategy
        def simple_strategy(data):
            signals = []
            for i in range(1, len(data)):
                if data.iloc[i]['close'] > data.iloc[i-1]['close']:
                    signals.append({"action": "BUY", "quantity": 100})
                else:
                    signals.append({"action": "SELL", "quantity": 100})
            return signals
        
        # Run backtest
        result = await engine.run_backtest(
            strategy=simple_strategy,
            data=data,
            initial_capital=100000
        )
        
        assert result.total_trades > 0
        assert result.final_equity != result.initial_capital
    
    @pytest.mark.asyncio
    async def test_walk_forward_validation(self, mock_data_generator):
        """Test walk-forward validation"""
        engine = BacktestEngine()
        
        data = mock_data_generator("AAPL", days=365)
        
        results = await engine.walk_forward_validation(
            data=data,
            window_size=30,
            step_size=10
        )
        
        assert len(results) > 0
        assert all(r.sharpe_ratio is not None for r in results)

# Test Configuration
class TestConfiguration:
    """Test configuration and setup"""
    
    def test_logging_configuration(self):
        """Test logging configuration"""
        logger = setup_logging("test_module")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Check log file exists and contains messages
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Info message" in content
            assert "Warning message" in content
            assert "Error message" in content
        
        os.unlink(log_file)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        from config_validator import validate_config
        
        valid_config = {
            "broker": {"type": "alpaca", "api_key": "test"},
            "risk": {"max_position_size": 10000},
            "strategy": {"name": "momentum"}
        }
        
        assert validate_config(valid_config) == True
        
        invalid_config = {
            "broker": {"type": "invalid_broker"}
        }
        
        assert validate_config(invalid_config) == False

# Pytest Configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Test Runner
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--cov=.",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-x",
        "--tb=short"
    ])

"""
Comprehensive backtesting framework for ARES trading system.
Includes walk-forward validation, performance metrics, and report generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import logging
from pathlib import Path
import warnings
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    
    initial_capital: float = 100000.0
    position_size: float = 0.02  # 2% per position
    max_positions: int = 10
    commission: float = 0.001  # 0.1% per trade
    slippage_bps: float = 5  # 5 basis points
    min_holding_period: int = 1  # days
    max_holding_period: int = 30  # days
    stop_loss: float = 0.05  # 5% stop loss
    take_profit: float = 0.15  # 15% take profit
    use_kelly_sizing: bool = True
    kelly_fraction: float = 0.25  # Use 25% of Kelly criterion
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    benchmark_symbols: List[str] = field(default_factory=lambda: ['SPY', 'QQQ'])
    risk_free_rate: float = 0.02  # 2% annual risk-free rate


@dataclass
class TradeRecord:
    """Record of a single trade."""
    
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    shares: int
    position_value: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    holding_period: Optional[int] = None
    trade_type: str = 'long'  # 'long' or 'short'
    exit_reason: Optional[str] = None  # 'stop_loss', 'take_profit', 'signal', 'max_holding'
    commission_paid: float = 0.0
    slippage_cost: float = 0.0


class TransactionCostModel:
    """Model for calculating transaction costs including commission and slippage."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade."""
        return trade_value * self.config.commission
    
    def calculate_slippage(self, 
                          price: float, 
                          volume: float,
                          trade_size: float,
                          is_buy: bool) -> float:
        """
        Calculate slippage based on trade size and market conditions.
        
        Args:
            price: Current market price
            volume: Average daily volume
            trade_size: Number of shares to trade
            is_buy: True for buy orders, False for sell orders
        
        Returns:
            Adjusted price after slippage
        """
        # Base slippage
        slippage_pct = self.config.slippage_bps / 10000
        
        # Adjust for trade size relative to volume
        if volume > 0:
            impact_factor = min(trade_size / (volume * 0.01), 1.0)  # Max 1% of daily volume
            slippage_pct *= (1 + impact_factor)
        
        # Apply slippage
        if is_buy:
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)
    
    def get_execution_price(self,
                           price: float,
                           volume: float,
                           trade_size: float,
                           is_buy: bool) -> Tuple[float, float]:
        """
        Get execution price and slippage cost.
        
        Returns:
            Tuple of (execution_price, slippage_cost)
        """
        execution_price = self.calculate_slippage(price, volume, trade_size, is_buy)
        slippage_cost = abs(execution_price - price) * trade_size
        return execution_price, slippage_cost


class PerformanceMetrics:
    """Calculate comprehensive performance metrics."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, 
                              risk_free_rate: float = 0.02,
                              periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate / periods_per_year
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series,
                               risk_free_rate: float = 0.02,
                               periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, datetime, datetime]:
        """
        Calculate maximum drawdown and dates.
        
        Returns:
            Tuple of (max_drawdown_pct, peak_date, trough_date)
        """
        cumulative = (1 + equity_curve).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        if max_dd == 0:
            return 0.0, equity_curve.index[0], equity_curve.index[0]
        
        trough_date = drawdown.idxmin()
        peak_date = cumulative[:trough_date].idxmax()
        
        return abs(max_dd), peak_date, trough_date
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, 
                              periods_per_year: int = 252) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        annual_return = (1 + returns.mean()) ** periods_per_year - 1
        max_dd, _, _ = PerformanceMetrics.calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def calculate_win_rate(trades: List[TradeRecord]) -> float:
        """Calculate win rate from completed trades."""
        completed_trades = [t for t in trades if t.pnl is not None]
        if not completed_trades:
            return 0.0
        
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        return len(winning_trades) / len(completed_trades)
    
    @staticmethod
    def calculate_profit_factor(trades: List[TradeRecord]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        completed_trades = [t for t in trades if t.pnl is not None]
        if not completed_trades:
            return 0.0
        
        gross_profit = sum(t.pnl for t in completed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in completed_trades if t.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_expectancy(trades: List[TradeRecord]) -> float:
        """Calculate trade expectancy."""
        completed_trades = [t for t in trades if t.pnl_pct is not None]
        if not completed_trades:
            return 0.0
        
        return np.mean([t.pnl_pct for t in completed_trades])


class BacktestingEngine:
    """Main backtesting engine with walk-forward validation."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.transaction_model = TransactionCostModel(config)
        self.metrics = PerformanceMetrics()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # State tracking
        self.equity_curve = []
        self.trades: List[TradeRecord] = []
        self.open_positions: Dict[str, TradeRecord] = {}
        self.current_capital = config.initial_capital
        self.benchmark_data: Dict[str, pd.DataFrame] = {}
    
    def load_benchmark_data(self, start_date: datetime, end_date: datetime):
        """Load benchmark data for comparison."""
        self.logger.info(f"Loading benchmark data for {self.config.benchmark_symbols}")
        
        for symbol in self.config.benchmark_symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                self.benchmark_data[symbol] = data
                self.logger.info(f"Loaded {len(data)} days of data for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to load benchmark {symbol}: {e}")
    
    def run_backtest(self,
                    signals: pd.DataFrame,
                    price_data: pd.DataFrame,
                    volume_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            signals: DataFrame with columns ['symbol', 'date', 'signal', 'confidence']
            price_data: DataFrame with price data
            volume_data: Optional DataFrame with volume data
        
        Returns:
            Dictionary with backtest results
        """
        self.logger.info("Starting backtest")
        
        # Reset state
        self.equity_curve = []
        self.trades = []
        self.open_positions = {}
        self.current_capital = self.config.initial_capital
        
        # Sort signals by date
        signals = signals.sort_values('date')
        
        # Process each trading day
        for date in signals['date'].unique():
            daily_signals = signals[signals['date'] == date]
            
            # Update open positions
            self._update_positions(date, price_data, volume_data)
            
            # Process new signals
            self._process_signals(daily_signals, date, price_data, volume_data)
            
            # Record equity
            equity = self._calculate_equity(date, price_data)
            self.equity_curve.append({
                'date': date,
                'equity': equity,
                'cash': self.current_capital,
                'positions': len(self.open_positions)
            })
        
        # Close remaining positions
        self._close_all_positions(signals['date'].max(), price_data, volume_data)
        
        # Calculate metrics
        results = self._calculate_results()
        
        return results
    
    def _update_positions(self, 
                         date: datetime,
                         price_data: pd.DataFrame,
                         volume_data: Optional[pd.DataFrame]):
        """Update open positions and check exit conditions."""
        positions_to_close = []
        
        for symbol, position in self.open_positions.items():
            current_price = self._get_price(symbol, date, price_data)
            if current_price is None:
                continue
            
            # Calculate current P&L
            pnl_pct = (current_price - position.entry_price) / position.entry_price
            
            # Check exit conditions
            exit_reason = None
            
            # Stop loss
            if pnl_pct <= -self.config.stop_loss:
                exit_reason = 'stop_loss'
            
            # Take profit
            elif pnl_pct >= self.config.take_profit:
                exit_reason = 'take_profit'
            
            # Max holding period
            elif (date - position.entry_date).days >= self.config.max_holding_period:
                exit_reason = 'max_holding'
            
            if exit_reason:
                positions_to_close.append((symbol, exit_reason))
        
        # Close positions
        for symbol, exit_reason in positions_to_close:
            self._close_position(symbol, date, price_data, volume_data, exit_reason)
    
    def _process_signals(self,
                        signals: pd.DataFrame,
                        date: datetime,
                        price_data: pd.DataFrame,
                        volume_data: Optional[pd.DataFrame]):
        """Process new trading signals."""
        # Filter for strong signals
        strong_signals = signals[signals['confidence'] > 0.6].copy()
        
        # Sort by confidence
        strong_signals = strong_signals.sort_values('confidence', ascending=False)
        
        # Process signals up to max positions
        for _, signal in strong_signals.iterrows():
            if len(self.open_positions) >= self.config.max_positions:
                break
            
            if signal['symbol'] not in self.open_positions:
                self._open_position(signal, date, price_data, volume_data)
    
    def _open_position(self,
                      signal: pd.Series,
                      date: datetime,
                      price_data: pd.DataFrame,
                      volume_data: Optional[pd.DataFrame]):
        """Open a new position."""
        symbol = signal['symbol']
        price = self._get_price(symbol, date, price_data)
        
        if price is None or price <= 0:
            return
        
        # Calculate position size
        if self.config.use_kelly_sizing:
            position_size = self._calculate_kelly_size(signal['confidence'])
        else:
            position_size = self.config.position_size
        
        position_value = self.current_capital * position_size
        
        # Get volume for slippage calculation
        volume = self._get_volume(symbol, date, volume_data) if volume_data is not None else 1e6
        
        # Calculate shares and execution price
        shares = int(position_value / price)
        if shares <= 0:
            return
        
        execution_price, slippage_cost = self.transaction_model.get_execution_price(
            price, volume, shares, is_buy=True
        )
        
        # Calculate commission
        actual_position_value = shares * execution_price
        commission = self.transaction_model.calculate_commission(actual_position_value)
        
        # Check if we have enough capital
        total_cost = actual_position_value + commission + slippage_cost
        if total_cost > self.current_capital:
            return
        
        # Create trade record
        trade = TradeRecord(
            symbol=symbol,
            entry_date=date,
            exit_date=None,
            entry_price=execution_price,
            exit_price=None,
            shares=shares,
            position_value=actual_position_value,
            commission_paid=commission,
            slippage_cost=slippage_cost
        )
        
        # Update capital and positions
        self.current_capital -= total_cost
        self.open_positions[symbol] = trade
        
        self.logger.debug(f"Opened position: {symbol} @ {execution_price:.2f} ({shares} shares)")
    
    def _close_position(self,
                       symbol: str,
                       date: datetime,
                       price_data: pd.DataFrame,
                       volume_data: Optional[pd.DataFrame],
                       exit_reason: str):
        """Close an existing position."""
        if symbol not in self.open_positions:
            return
        
        position = self.open_positions[symbol]
        price = self._get_price(symbol, date, price_data)
        
        if price is None or price <= 0:
            return
        
        # Get volume for slippage calculation
        volume = self._get_volume(symbol, date, volume_data) if volume_data is not None else 1e6
        
        # Calculate execution price
        execution_price, slippage_cost = self.transaction_model.get_execution_price(
            price, volume, position.shares, is_buy=False
        )
        
        # Calculate proceeds and commission
        proceeds = position.shares * execution_price
        commission = self.transaction_model.calculate_commission(proceeds)
        
        # Update trade record
        position.exit_date = date
        position.exit_price = execution_price
        position.pnl = proceeds - position.position_value - position.commission_paid - commission - position.slippage_cost - slippage_cost
        position.pnl_pct = position.pnl / position.position_value
        position.holding_period = (date - position.entry_date).days
        position.exit_reason = exit_reason
        
        # Update capital
        self.current_capital += proceeds - commission - slippage_cost
        
        # Move to completed trades
        self.trades.append(position)
        del self.open_positions[symbol]
        
        self.logger.debug(f"Closed position: {symbol} @ {execution_price:.2f} "
                         f"(PnL: {position.pnl:.2f}, {position.pnl_pct:.2%})")
    
    def _close_all_positions(self,
                            date: datetime,
                            price_data: pd.DataFrame,
                            volume_data: Optional[pd.DataFrame]):
        """Close all remaining open positions."""
        symbols = list(self.open_positions.keys())
        for symbol in symbols:
            self._close_position(symbol, date, price_data, volume_data, 'end_of_backtest')
    
    def _calculate_equity(self, date: datetime, price_data: pd.DataFrame) -> float:
        """Calculate total equity (cash + positions)."""
        equity = self.current_capital
        
        for symbol, position in self.open_positions.items():
            current_price = self._get_price(symbol, date, price_data)
            if current_price:
                equity += position.shares * current_price
        
        return equity
    
    def _calculate_kelly_size(self, confidence: float) -> float:
        """Calculate position size using Kelly criterion."""
        # Estimate win probability and win/loss ratio from historical trades
        if len(self.trades) < 10:
            return self.config.position_size
        
        recent_trades = self.trades[-50:]  # Use last 50 trades
        win_rate = self.metrics.calculate_win_rate(recent_trades)
        
        if win_rate == 0 or win_rate == 1:
            return self.config.position_size
        
        # Calculate average win and loss
        wins = [t.pnl_pct for t in recent_trades if t.pnl_pct and t.pnl_pct > 0]
        losses = [abs(t.pnl_pct) for t in recent_trades if t.pnl_pct and t.pnl_pct < 0]
        
        if not wins or not losses:
            return self.config.position_size
        
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        b = avg_win / avg_loss
        kelly_pct = (win_rate * b - (1 - win_rate)) / b
        
        # Apply Kelly fraction and bounds
        kelly_pct = max(0, min(kelly_pct * self.config.kelly_fraction, 0.25))
        
        # Adjust by confidence
        return kelly_pct * confidence
    
    def _get_price(self, symbol: str, date: datetime, price_data: pd.DataFrame) -> Optional[float]:
        """Get price for symbol on date."""
        try:
            if symbol in price_data.columns and date in price_data.index:
                return price_data.loc[date, symbol]
        except:
            pass
        return None
    
    def _get_volume(self, symbol: str, date: datetime, volume_data: pd.DataFrame) -> float:
        """Get volume for symbol on date."""
        try:
            if symbol in volume_data.columns and date in volume_data.index:
                return volume_data.loc[date, symbol]
        except:
            pass
        return 1e6  # Default volume
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        if not self.equity_curve:
            return {}
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calculate metrics
        total_return = (equity_df['equity'].iloc[-1] / self.config.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(equity_df)) - 1
        
        sharpe = self.metrics.calculate_sharpe_ratio(equity_df['returns'].dropna(), self.config.risk_free_rate)
        sortino = self.metrics.calculate_sortino_ratio(equity_df['returns'].dropna(), self.config.risk_free_rate)
        max_dd, dd_start, dd_end = self.metrics.calculate_max_drawdown(equity_df['returns'].dropna())
        calmar = self.metrics.calculate_calmar_ratio(equity_df['returns'].dropna())
        
        win_rate = self.metrics.calculate_win_rate(self.trades)
        profit_factor = self.metrics.calculate_profit_factor(self.trades)
        expectancy = self.metrics.calculate_expectancy(self.trades)
        
        # Trade statistics
        completed_trades = [t for t in self.trades if t.pnl is not None]
        
        results = {
            'equity_curve': equity_df,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'max_dd_start': dd_start,
            'max_dd_end': dd_end,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'total_trades': len(completed_trades),
            'winning_trades': len([t for t in completed_trades if t.pnl > 0]),
            'losing_trades': len([t for t in completed_trades if t.pnl < 0]),
            'avg_win': np.mean([t.pnl_pct for t in completed_trades if t.pnl_pct and t.pnl_pct > 0]) if completed_trades else 0,
            'avg_loss': np.mean([t.pnl_pct for t in completed_trades if t.pnl_pct and t.pnl_pct < 0]) if completed_trades else 0,
            'largest_win': max([t.pnl for t in completed_trades if t.pnl], default=0),
            'largest_loss': min([t.pnl for t in completed_trades if t.pnl], default=0),
            'avg_holding_period': np.mean([t.holding_period for t in completed_trades if t.holding_period]) if completed_trades else 0,
            'trades': self.trades
        }
        
        return results
    
    def walk_forward_validation(self,
                               data: pd.DataFrame,
                               signals_generator,
                               window_size: int = 252,
                               step_size: int = 63,
                               n_windows: int = 4) -> List[Dict[str, Any]]:
        """
        Perform walk-forward validation.
        
        Args:
            data: Historical price data
            signals_generator: Function to generate signals
            window_size: Size of training window in days
            step_size: Step size for moving window
            n_windows: Number of windows to test
        
        Returns:
            List of backtest results for each window
        """
        self.logger.info(f"Starting walk-forward validation with {n_windows} windows")
        
        results = []
        dates = data.index.unique()
        
        for i in range(n_windows):
            # Define training and testing periods
            train_start = i * step_size
            train_end = train_start + window_size
            test_start = train_end
            test_end = test_start + step_size
            
            if test_end >= len(dates):
                break
            
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            self.logger.info(f"Window {i+1}: Training {dates[train_start]} to {dates[train_end-1]}, "
                           f"Testing {dates[test_start]} to {dates[test_end-1]}")
            
            # Generate signals on test data using model trained on train data
            signals = signals_generator(train_data, test_data)
            
            # Run backtest
            window_results = self.run_backtest(signals, test_data)
            window_results['window'] = i + 1
            window_results['train_period'] = (dates[train_start], dates[train_end-1])
            window_results['test_period'] = (dates[test_start], dates[test_end-1])
            
            results.append(window_results)
        
        return results
    
    def generate_report(self, 
                       results: Dict[str, Any],
                       output_dir: str = 'reports',
                       report_name: str = None) -> str:
        """
        Generate comprehensive backtest report with charts.
        
        Args:
            results: Backtest results dictionary
            output_dir: Directory to save report
            report_name: Name for the report files
        
        Returns:
            Path to generated report
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate report name
        if report_name is None:
            report_name = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Equity Curve
        ax1 = plt.subplot(3, 3, 1)
        equity_df = results['equity_curve']
        ax1.plot(equity_df.index, equity_df['equity'], label='Strategy', linewidth=2)
        
        # Add benchmark if available
        if self.benchmark_data:
            for symbol, data in self.benchmark_data.items():
                benchmark_equity = self.config.initial_capital * (data['Close'] / data['Close'].iloc[0])
                ax1.plot(data.index, benchmark_equity, label=symbol, alpha=0.7)
        
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = plt.subplot(3, 3, 2)
        cumulative = (1 + equity_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Returns Distribution
        ax3 = plt.subplot(3, 3, 3)
        returns = equity_df['returns'].dropna()
        ax3.hist(returns * 100, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(returns.mean() * 100, color='red', linestyle='--', label=f'Mean: {returns.mean()*100:.2f}%')
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Daily Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap
        ax4 = plt.subplot(3, 3, 4)
        monthly_returns = equity_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_pivot = pd.pivot_table(
            pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values * 100
            }),
            values='Return',
            index='Month',
            columns='Year'
        )
        sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax4)
        ax4.set_title('Monthly Returns (%)')
        
        # 5. Trade P&L Distribution
        ax5 = plt.subplot(3, 3, 5)
        trade_pnls = [t.pnl_pct * 100 for t in results['trades'] if t.pnl_pct is not None]
        if trade_pnls:
            ax5.hist(trade_pnls, bins=30, edgecolor='black', alpha=0.7)
            ax5.axvline(np.mean(trade_pnls), color='red', linestyle='--', label=f'Mean: {np.mean(trade_pnls):.2f}%')
            ax5.set_title('Trade P&L Distribution')
            ax5.set_xlabel('Trade Return (%)')
            ax5.set_ylabel('Frequency')
            ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Win/Loss by Exit Reason
        ax6 = plt.subplot(3, 3, 6)
        exit_reasons = {}
        for trade in results['trades']:
            if trade.exit_reason:
                if trade.exit_reason not in exit_reasons:
                    exit_reasons[trade.exit_reason] = {'wins': 0, 'losses': 0}
                if trade.pnl and trade.pnl > 0:
                    exit_reasons[trade.exit_reason]['wins'] += 1
                else:
                    exit_reasons[trade.exit_reason]['losses'] += 1
        
        if exit_reasons:
            reasons = list(exit_reasons.keys())
            wins = [exit_reasons[r]['wins'] for r in reasons]
            losses = [exit_reasons[r]['losses'] for r in reasons]
            
            x = np.arange(len(reasons))
            width = 0.35
            ax6.bar(x - width/2, wins, width, label='Wins', color='green', alpha=0.7)
            ax6.bar(x + width/2, losses, width, label='Losses', color='red', alpha=0.7)
            ax6.set_xlabel('Exit Reason')
            ax6.set_ylabel('Number of Trades')
            ax6.set_title('Trades by Exit Reason')
            ax6.set_xticks(x)
            ax6.set_xticklabels(reasons, rotation=45)
            ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Rolling Sharpe Ratio
        ax7 = plt.subplot(3, 3, 7)
        rolling_sharpe = equity_df['returns'].rolling(window=63).apply(
            lambda x: self.metrics.calculate_sharpe_ratio(x, self.config.risk_free_rate)
        )
        ax7.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2)
        ax7.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax7.axhline(y=1, color='g', linestyle='--', alpha=0.5)
        ax7.set_title('Rolling Sharpe Ratio (3-month)')
        ax7.set_xlabel('Date')
        ax7.set_ylabel('Sharpe Ratio')
        ax7.grid(True, alpha=0.3)
        
        # 8. Cumulative Returns Comparison
        ax8 = plt.subplot(3, 3, 8)
        strategy_cumret = (1 + equity_df['returns']).cumprod() - 1
        ax8.plot(strategy_cumret.index, strategy_cumret * 100, label='Strategy', linewidth=2)
        
        if self.benchmark_data:
            for symbol, data in self.benchmark_data.items():
                benchmark_returns = data['Close'].pct_change()
                benchmark_cumret = (1 + benchmark_returns).cumprod() - 1
                ax8.plot(data.index, benchmark_cumret * 100, label=symbol, alpha=0.7)
        
        ax8.set_title('Cumulative Returns')
        ax8.set_xlabel('Date')
        ax8.set_ylabel('Cumulative Return (%)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Performance Metrics Table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('tight')
        ax9.axis('off')
        
        metrics_data = [
            ['Total Return', f"{results['total_return']:.2%}"],
            ['Annual Return', f"{results['annual_return']:.2%}"],
            ['Sharpe Ratio', f"{results['sharpe_ratio']:.2f}"],
            ['Sortino Ratio', f"{results['sortino_ratio']:.2f}"],
            ['Max Drawdown', f"{results['max_drawdown']:.2%}"],
            ['Calmar Ratio', f"{results['calmar_ratio']:.2f}"],
            ['Win Rate', f"{results['win_rate']:.2%}"],
            ['Profit Factor', f"{results['profit_factor']:.2f}"],
            ['Total Trades', f"{results['total_trades']}"],
            ['Avg Win', f"{results['avg_win']:.2%}"],
            ['Avg Loss', f"{results['avg_loss']:.2%}"],
            ['Expectancy', f"{results['expectancy']:.2%}"]
        ]
        
        table = ax9.table(cellText=metrics_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax9.set_title('Performance Metrics', pad=20)
        
        plt.suptitle('Backtest Report', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        chart_path = output_path / f"{report_name}.png"
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Generate HTML report
        html_content = self._generate_html_report(results, chart_path.name)
        html_path = output_path / f"{report_name}.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        # Save results to JSON
        json_path = output_path / f"{report_name}.json"
        json_results = {k: v for k, v in results.items() if k not in ['equity_curve', 'trades']}
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        self.logger.info(f"Report generated: {html_path}")
        
        return str(html_path)
    
    def _generate_html_report(self, results: Dict[str, Any], chart_filename: str) -> str:
        """Generate HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric-positive {{ color: green; }}
                .metric-negative {{ color: red; }}
                .chart {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>ARES Trading System - Backtest Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Performance Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Return</td>
                    <td class="{'metric-positive' if results['total_return'] > 0 else 'metric-negative'}">
                        {results['total_return']:.2%}
                    </td>
                </tr>
                <tr>
                    <td>Annual Return</td>
                    <td class="{'metric-positive' if results['annual_return'] > 0 else 'metric-negative'}">
                        {results['annual_return']:.2%}
                    </td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{results['sharpe_ratio']:.2f}</td>
                </tr>
                <tr>
                    <td>Sortino Ratio</td>
                    <td>{results['sortino_ratio']:.2f}</td>
                </tr>
                <tr>
                    <td>Maximum Drawdown</td>
                    <td class="metric-negative">{results['max_drawdown']:.2%}</td>
                </tr>
                <tr>
                    <td>Calmar Ratio</td>
                    <td>{results['calmar_ratio']:.2f}</td>
                </tr>
            </table>
            
            <h2>Trading Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Trades</td>
                    <td>{results['total_trades']}</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{results['win_rate']:.2%}</td>
                </tr>
                <tr>
                    <td>Profit Factor</td>
                    <td>{results['profit_factor']:.2f}</td>
                </tr>
                <tr>
                    <td>Average Win</td>
                    <td class="metric-positive">{results['avg_win']:.2%}</td>
                </tr>
                <tr>
                    <td>Average Loss</td>
                    <td class="metric-negative">{results['avg_loss']:.2%}</td>
                </tr>
                <tr>
                    <td>Expectancy</td>
                    <td class="{'metric-positive' if results['expectancy'] > 0 else 'metric-negative'}">
                        {results['expectancy']:.2%}
                    </td>
                </tr>
                <tr>
                    <td>Average Holding Period</td>
                    <td>{results['avg_holding_period']:.1f} days</td>
                </tr>
            </table>
            
            <h2>Performance Charts</h2>
            <img src="{chart_filename}" class="chart" alt="Performance Charts">
            
        </body>
        </html>
        """
        return html


if __name__ == "__main__":
    # Example usage
    config = BacktestConfig(
        initial_capital=100000,
        position_size=0.02,
        max_positions=10,
        commission=0.001,
        slippage_bps=5
    )
    
    engine = BacktestingEngine(config)
    
    # Example: Generate dummy data for testing
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Create dummy price data
    price_data = pd.DataFrame(
        np.random.randn(len(dates), len(symbols)).cumsum() + 100,
        index=dates,
        columns=symbols
    )
    
    # Create dummy signals
    signals = []
    for i in range(100):
        signals.append({
            'date': np.random.choice(dates),
            'symbol': np.random.choice(symbols),
            'signal': np.random.choice(['buy', 'sell']),
            'confidence': np.random.uniform(0.5, 1.0)
        })
    
    signals_df = pd.DataFrame(signals)
    
    # Run backtest
    results = engine.run_backtest(signals_df, price_data)
    
    # Generate report
    report_path = engine.generate_report(results)
    print(f"Report generated: {report_path}")