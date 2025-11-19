"""
Centralized configuration management for all magic numbers and thresholds.
All hardcoded values moved here with documentation and validation.
"""

import os
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types for configuration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class ThresholdConfig:
    """Configuration for trading thresholds with validation."""
    
    # VPIN Thresholds
    VPIN_CRITICAL: float = 0.85  # Critical VPIN level indicating toxic flow
    VPIN_WARNING: float = 0.70   # Warning VPIN level
    VPIN_SAFE: float = 0.50      # Safe VPIN level
    
    # GEX (Gamma Exposure) Thresholds
    GEX_NEGATIVE_EXTREME: float = -2_000_000_000  # -2B extreme negative GEX
    GEX_NEGATIVE_HIGH: float = -1_000_000_000     # -1B high negative GEX
    GEX_NEUTRAL_RANGE: tuple = (-500_000_000, 500_000_000)  # Neutral zone
    GEX_POSITIVE_HIGH: float = 1_000_000_000      # 1B high positive GEX
    GEX_POSITIVE_EXTREME: float = 2_000_000_000   # 2B extreme positive GEX
    
    # DIX (Dark Pool Index) Thresholds
    DIX_BULLISH: float = 0.45    # Bullish dark pool activity
    DIX_NEUTRAL: float = 0.40    # Neutral dark pool activity
    DIX_BEARISH: float = 0.35    # Bearish dark pool activity
    
    # VIX Thresholds
    VIX_LOW: float = 12.0        # Low volatility environment
    VIX_NORMAL: float = 20.0     # Normal volatility
    VIX_HIGH: float = 30.0       # High volatility
    VIX_EXTREME: float = 40.0    # Extreme volatility
    
    # Risk Management Thresholds
    MAX_POSITION_SIZE: float = 0.02  # 2% max position size
    MAX_PORTFOLIO_RISK: float = 0.06  # 6% max portfolio risk
    MAX_CORRELATION: float = 0.70     # Maximum correlation threshold
    MIN_SHARPE_RATIO: float = 0.5    # Minimum acceptable Sharpe ratio
    
    # Stop Loss and Take Profit
    DEFAULT_STOP_LOSS: float = 0.02   # 2% stop loss
    DEFAULT_TAKE_PROFIT: float = 0.05  # 5% take profit
    TRAILING_STOP: float = 0.015      # 1.5% trailing stop
    
    # Volume Thresholds
    MIN_VOLUME_PERCENTILE: float = 0.20  # Minimum 20th percentile volume
    VOLUME_SPIKE_MULTIPLIER: float = 2.0  # 2x average volume = spike
    
    # Sentiment Thresholds
    SENTIMENT_VERY_BEARISH: float = -0.7
    SENTIMENT_BEARISH: float = -0.3
    SENTIMENT_NEUTRAL: tuple = (-0.3, 0.3)
    SENTIMENT_BULLISH: float = 0.3
    SENTIMENT_VERY_BULLISH: float = 0.7
    
    def validate(self) -> bool:
        """Validate threshold configurations."""
        validations = [
            (0 <= self.VPIN_SAFE <= self.VPIN_WARNING <= self.VPIN_CRITICAL <= 1,
             "VPIN thresholds must be between 0 and 1 and properly ordered"),
            
            (self.GEX_NEGATIVE_EXTREME < self.GEX_NEGATIVE_HIGH < 0,
             "Negative GEX thresholds must be properly ordered"),
            
            (self.GEX_POSITIVE_HIGH > 0 and self.GEX_POSITIVE_EXTREME > self.GEX_POSITIVE_HIGH,
             "Positive GEX thresholds must be properly ordered"),
            
            (0 < self.DIX_BEARISH < self.DIX_NEUTRAL < self.DIX_BULLISH < 1,
             "DIX thresholds must be between 0 and 1 and properly ordered"),
            
            (0 < self.VIX_LOW < self.VIX_NORMAL < self.VIX_HIGH < self.VIX_EXTREME,
             "VIX thresholds must be properly ordered"),
            
            (0 < self.MAX_POSITION_SIZE <= 0.1,
             "Max position size must be between 0 and 10%"),
            
            (0 < self.MAX_PORTFOLIO_RISK <= 0.2,
             "Max portfolio risk must be between 0 and 20%"),
        ]
        
        for condition, message in validations:
            if not condition:
                logger.error(f"Validation failed: {message}")
                return False
        
        return True

@dataclass
class SystemConfig:
    """System-wide configuration parameters."""
    
    # Timing Parameters (in seconds)
    MARKET_DATA_REFRESH: int = 1          # Market data refresh interval
    SENTIMENT_REFRESH: int = 60           # Sentiment analysis refresh
    RISK_CHECK_INTERVAL: int = 5          # Risk management check interval
    HEALTH_CHECK_INTERVAL: int = 30       # System health check interval
    
    # Cache Settings
    CACHE_TTL_QUOTES: int = 5             # Quote cache TTL in seconds
    CACHE_TTL_OPTIONS: int = 60           # Options cache TTL
    CACHE_TTL_SENTIMENT: int = 300        # Sentiment cache TTL
    
    # Rate Limiting
    API_RATE_LIMIT_DEFAULT: int = 5       # Default requests per second
    API_RATE_LIMIT_AGGRESSIVE: int = 10   # Aggressive rate limit
    API_RATE_LIMIT_CONSERVATIVE: int = 2  # Conservative rate limit
    
    # Retry Configuration
    MAX_RETRIES: int = 3                  # Maximum retry attempts
    RETRY_BACKOFF_BASE: float = 2.0       # Exponential backoff base
    RETRY_JITTER: float = 0.1             # Jitter factor for retries
    
    # Circuit Breaker
    CIRCUIT_BREAKER_THRESHOLD: int = 5    # Failures before opening
    CIRCUIT_BREAKER_TIMEOUT: int = 300    # Timeout in seconds (5 min)
    CIRCUIT_BREAKER_HALF_OPEN: int = 60   # Half-open test interval
    
    # Database Settings
    DB_CONNECTION_POOL_SIZE: int = 10     # Connection pool size
    DB_CONNECTION_TIMEOUT: int = 30       # Connection timeout
    DB_QUERY_TIMEOUT: int = 10            # Query timeout
    
    # Logging
    LOG_ROTATION_SIZE: int = 10_485_760   # 10MB log file size
    LOG_BACKUP_COUNT: int = 10            # Number of backup logs
    LOG_LEVEL_DEFAULT: str = "INFO"       # Default log level
    
    # Performance
    MAX_CONCURRENT_ORDERS: int = 10       # Max concurrent orders
    ORDER_QUEUE_SIZE: int = 100          # Order queue size
    WORKER_THREADS: int = 4               # Number of worker threads

@dataclass
class TradingConfig:
    """Trading-specific configuration."""
    
    # Order Types
    ALLOWED_ORDER_TYPES: list = field(default_factory=lambda: [
        "MARKET", "LIMIT", "STOP", "STOP_LIMIT", "TRAILING_STOP"
    ])
    
    # Time in Force
    DEFAULT_TIME_IN_FORCE: str = "DAY"
    ALLOWED_TIF: list = field(default_factory=lambda: [
        "DAY", "GTC", "IOC", "FOK", "GTX"
    ])
    
    # Position Sizing
    MIN_POSITION_SIZE: float = 100.0      # Minimum position size in dollars
    MAX_POSITION_SIZE: float = 100_000.0  # Maximum position size in dollars
    POSITION_SIZING_METHOD: str = "KELLY"  # KELLY, FIXED, VOLATILITY_BASED
    
    # Slippage and Costs
    DEFAULT_SLIPPAGE: float = 0.0005      # 0.05% slippage
    COMMISSION_PER_SHARE: float = 0.005   # $0.005 per share
    MIN_COMMISSION: float = 1.0           # $1 minimum commission
    
    # Trading Hours (Eastern Time)
    MARKET_OPEN: str = "09:30"
    MARKET_CLOSE: str = "16:00"
    PRE_MARKET_OPEN: str = "04:00"
    AFTER_MARKET_CLOSE: str = "20:00"
    
    # Risk Limits
    DAILY_LOSS_LIMIT: float = 0.02        # 2% daily loss limit
    DAILY_TRADE_LIMIT: int = 100          # Maximum trades per day
    CONCENTRATION_LIMIT: float = 0.20     # 20% max in single position

class ConfigManager:
    """Centralized configuration management with environment support."""
    
    def __init__(self, environment: Optional[str] = None):
        self.environment = Environment(
            environment or os.getenv('TRADING_ENV', 'development')
        )
        
        self.thresholds = ThresholdConfig()
        self.system = SystemConfig()
        self.trading = TradingConfig()
        
        # Load environment-specific overrides
        self._load_environment_config()
        
        # Validate all configurations
        self._validate_all()
    
    def _load_environment_config(self):
        """Load environment-specific configuration overrides."""
        config_file = f"config/{self.environment.value}.json"
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    overrides = json.load(f)
                    self._apply_overrides(overrides)
                    logger.info(f"Loaded {self.environment.value} configuration")
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
    
    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Apply configuration overrides from file or environment."""
        for section, values in overrides.items():
            if section == 'thresholds' and hasattr(self, 'thresholds'):
                for key, value in values.items():
                    if hasattr(self.thresholds, key):
                        setattr(self.thresholds, key, value)
            
            elif section == 'system' and hasattr(self, 'system'):
                for key, value in values.items():
                    if hasattr(self.system, key):
                        setattr(self.system, key, value)
            
            elif section == 'trading' and hasattr(self, 'trading'):
                for key, value in values.items():
                    if hasattr(self.trading, key):
                        setattr(self.trading, key, value)
    
    def _validate_all(self):
        """Validate all configuration sections."""
        if not self.thresholds.validate():
            raise ValueError("Invalid threshold configuration")
        
        logger.info("All configurations validated successfully")
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation path."""
        try:
            parts = path.split('.')
            value = self
            
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return default
            
            return value
        except Exception:
            return default
    
    def set(self, path: str, value: Any):
        """Set configuration value by dot-notation path."""
        parts = path.split('.')
        obj = self
        
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise ValueError(f"Invalid configuration path: {path}")
        
        if hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], value)
            logger.info(f"Updated configuration: {path} = {value}")
        else:
            raise ValueError(f"Invalid configuration key: {parts[-1]}")
    
    def export(self) -> Dict[str, Any]:
        """Export current configuration as dictionary."""
        return {
            'environment': self.environment.value,
            'thresholds': self.thresholds.__dict__,
            'system': self.system.__dict__,
            'trading': self.trading.__dict__
        }
    
    def reload(self):
        """Reload configuration from files."""
        self._load_environment_config()
        self._validate_all()
        logger.info("Configuration reloaded")

# Global configuration instance
config = ConfigManager()

# Convenience accessors
THRESHOLDS = config.thresholds
SYSTEM = config.system
TRADING = config.trading

def get_config(path: str, default: Any = None) -> Any:
    """Global accessor for configuration values."""
    return config.get(path, default)

def update_config(path: str, value: Any):
    """Global setter for configuration values."""
    config.set(path, value)

if __name__ == "__main__":
    # Test configuration
    logging.basicConfig(level=logging.INFO)
    
    print(f"Environment: {config.environment.value}")
    print(f"VPIN Critical: {THRESHOLDS.VPIN_CRITICAL}")
    print(f"GEX Negative Extreme: {THRESHOLDS.GEX_NEGATIVE_EXTREME:,}")
    print(f"Max Position Size: {THRESHOLDS.MAX_POSITION_SIZE:.1%}")
    print(f"Market Data Refresh: {SYSTEM.MARKET_DATA_REFRESH}s")
    print(f"Daily Loss Limit: {TRADING.DAILY_LOSS_LIMIT:.1%}")
    
    # Test configuration export
    config_dict = config.export()
    print(f"\nConfiguration sections: {list(config_dict.keys())}")

"""
Circuit Breaker Pattern Implementation
Provides fault tolerance and gradual degradation for external service calls
"""

import asyncio
import time
from enum import Enum
from typing import Callable, Optional, Any, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: List[Dict[str, Any]] = field(default_factory=list)

class CircuitBreaker:
    """
    Circuit breaker implementation with gradual degradation
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        success_threshold: int = 2,
        half_open_max_calls: int = 3,
        monitoring_window: int = 120,
        fallback_function: Optional[Callable] = None,
        state_persistence_path: Optional[str] = None
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
            success_threshold: Successes needed to close circuit
            half_open_max_calls: Max calls in half-open state
            monitoring_window: Time window for monitoring (seconds)
            fallback_function: Function to call when circuit is open
            state_persistence_path: Path to persist state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.half_open_max_calls = half_open_max_calls
        self.monitoring_window = monitoring_window
        self.fallback_function = fallback_function
        self.state_persistence_path = state_persistence_path
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._stats = CircuitBreakerStats()
        self._call_history = deque(maxlen=1000)
        self._health_check_function = None
        self._listeners = []
        
        # Load persisted state if available
        if state_persistence_path:
            self._load_state()
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current state with automatic transitions"""
        if self._state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
        return self._state
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        return (
            self._last_failure_time and
            time.time() - self._last_failure_time >= self.recovery_timeout
        )
    
    def _transition_to_half_open(self):
        """Transition to half-open state"""
        logger.info("Circuit breaker transitioning to HALF_OPEN")
        self._state = CircuitBreakerState.HALF_OPEN
        self._half_open_calls = 0
        self._success_count = 0
        self._record_state_change(CircuitBreakerState.HALF_OPEN)
        self._notify_listeners("state_change", self._state)
    
    def _record_state_change(self, new_state: CircuitBreakerState):
        """Record state change in statistics"""
        self._stats.state_changes.append({
            "timestamp": datetime.now().isoformat(),
            "from_state": self._state.value if self._state else None,
            "to_state": new_state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count
        })
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        # Check if circuit is open
        if self.state == CircuitBreakerState.OPEN:
            return await self._handle_open_circuit(func, *args, **kwargs)
        
        # Check half-open call limit
        if self._state == CircuitBreakerState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                return await self._handle_open_circuit(func, *args, **kwargs)
            self._half_open_calls += 1
        
        # Try to execute the function
        try:
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            self._on_success(execution_time)
            return result
            
        except self.expected_exception as e:
            self._on_failure(e)
            raise
    
    def _on_success(self, execution_time: float):
        """Handle successful call"""
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.last_success_time = datetime.now()
        self._stats.consecutive_successes += 1
        self._stats.consecutive_failures = 0
        
        self._call_history.append({
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "execution_time": execution_time
        })
        
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._close_circuit()
        elif self._state == CircuitBreakerState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)
    
    def _on_failure(self, exception: Exception):
        """Handle failed call"""
        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.last_failure_time = datetime.now()
        self._stats.consecutive_failures += 1
        self._stats.consecutive_successes = 0
        self._last_failure_time = time.time()
        
        self._call_history.append({
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": str(exception)
        })
        
        self._failure_count += 1
        
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._open_circuit()
        elif self._failure_count >= self.failure_threshold:
            self._open_circuit()
    
    def _open_circuit(self):
        """Open the circuit breaker"""
        logger.warning(f"Circuit breaker OPEN after {self._failure_count} failures")
        self._state = CircuitBreakerState.OPEN
        self._record_state_change(CircuitBreakerState.OPEN)
        self._notify_listeners("circuit_open", self._stats)
        self._persist_state()
    
    def _close_circuit(self):
        """Close the circuit breaker"""
        logger.info("Circuit breaker CLOSED")
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._record_state_change(CircuitBreakerState.CLOSED)
        self._notify_listeners("circuit_closed", self._stats)
        self._persist_state()
    
    async def _handle_open_circuit(self, func: Callable, *args, **kwargs) -> Any:
        """Handle call when circuit is open"""
        self._stats.rejected_calls += 1
        
        if self.fallback_function:
            logger.info("Circuit open, using fallback function")
            if asyncio.iscoroutinefunction(self.fallback_function):
                return await self.fallback_function(*args, **kwargs)
            return self.fallback_function(*args, **kwargs)
        
        raise CircuitBreakerOpenException(
            f"Circuit breaker is {self._state.value}, rejecting call"
        )
    
    def set_health_check(self, health_check: Callable[[], bool]):
        """Set health check function for proactive monitoring"""
        self._health_check_function = health_check
    
    async def check_health(self) -> bool:
        """Execute health check"""
        if not self._health_check_function:
            return True
        
        try:
            if asyncio.iscoroutinefunction(self._health_check_function):
                return await self._health_check_function()
            return self._health_check_function()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def add_listener(self, listener: Callable):
        """Add event listener"""
        self._listeners.append(listener)
    
    def _notify_listeners(self, event: str, data: Any):
        """Notify all listeners of an event"""
        for listener in self._listeners:
            try:
                listener(event, data)
            except Exception as e:
                logger.error(f"Error notifying listener: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "state": self._state.value,
            "total_calls": self._stats.total_calls,
            "successful_calls": self._stats.successful_calls,
            "failed_calls": self._stats.failed_calls,
            "rejected_calls": self._stats.rejected_calls,
            "success_rate": (
                self._stats.successful_calls / self._stats.total_calls
                if self._stats.total_calls > 0 else 0
            ),
            "last_failure": self._stats.last_failure_time.isoformat() if self._stats.last_failure_time else None,
            "last_success": self._stats.last_success_time.isoformat() if self._stats.last_success_time else None,
            "consecutive_failures": self._stats.consecutive_failures,
            "consecutive_successes": self._stats.consecutive_successes,
            "recent_calls": list(self._call_history)[-10:]
        }
    
    def reset(self):
        """Manually reset circuit breaker"""
        logger.info("Manually resetting circuit breaker")
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._stats = CircuitBreakerStats()
        self._persist_state()
    
    def _persist_state(self):
        """Persist circuit breaker state to disk"""
        if not self.state_persistence_path:
            return
        
        try:
            state_data = {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "stats": {
                    "total_calls": self._stats.total_calls,
                    "successful_calls": self._stats.successful_calls,
                    "failed_calls": self._stats.failed_calls,
                    "rejected_calls": self._stats.rejected_calls
                }
            }
            
            path = Path(self.state_persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(state_data, f)
                
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")
    
    def _load_state(self):
        """Load persisted state from disk"""
        if not self.state_persistence_path:
            return
        
        try:
            path = Path(self.state_persistence_path)
            if not path.exists():
                return
            
            with open(path, 'r') as f:
                state_data = json.load(f)
            
            self._state = CircuitBreakerState(state_data["state"])
            self._failure_count = state_data["failure_count"]
            self._success_count = state_data["success_count"]
            self._last_failure_time = state_data["last_failure_time"]
            
            stats = state_data.get("stats", {})
            self._stats.total_calls = stats.get("total_calls", 0)
            self._stats.successful_calls = stats.get("successful_calls", 0)
            self._stats.failed_calls = stats.get("failed_calls", 0)
            self._stats.rejected_calls = stats.get("rejected_calls", 0)
            
            logger.info(f"Loaded circuit breaker state: {self._state.value}")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class GradualDegradationStrategy:
    """Strategy for gradual service degradation"""
    
    def __init__(self, levels: List[Dict[str, Any]]):
        """
        Initialize degradation strategy
        
        Args:
            levels: List of degradation levels with thresholds and actions
        """
        self.levels = sorted(levels, key=lambda x: x["threshold"])
        self.current_level = 0
    
    def get_degradation_level(self, error_rate: float) -> Dict[str, Any]:
        """Get appropriate degradation level based on error rate"""
        for i, level in enumerate(self.levels):
            if error_rate <= level["threshold"]:
                self.current_level = i
                return level
        
        self.current_level = len(self.levels) - 1
        return self.levels[-1]
    
    def should_degrade(self, stats: Dict[str, Any]) -> bool:
        """Check if service should be degraded"""
        if stats["total_calls"] < 10:
            return False
        
        error_rate = stats["failed_calls"] / stats["total_calls"]
        level = self.get_degradation_level(error_rate)
        
        return level.get("degrade", False)

# Example usage
if __name__ == "__main__":
    async def main():
        # Create circuit breaker
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10,
            expected_exception=ValueError
        )
        
        # Example function that might fail
        async def risky_operation(should_fail: bool = False):
            if should_fail:
                raise ValueError("Operation failed")
            return "Success"
        
        # Test circuit breaker
        try:
            # Successful calls
            for _ in range(2):
                result = await breaker.call(risky_operation, False)
                print(f"Result: {result}")
            
            # Failed calls to open circuit
            for _ in range(3):
                try:
                    await breaker.call(risky_operation, True)
                except ValueError:
                    print("Call failed")
            
            # Circuit should be open now
            try:
                await breaker.call(risky_operation, False)
            except CircuitBreakerOpenException:
                print("Circuit is open!")
            
            # Print statistics
            print(json.dumps(breaker.get_stats(), indent=2))
            
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(main())

"""
Comprehensive logging configuration for ARES trading system.
Supports structured logging, file rotation, and ELK stack integration.
"""

import logging
import logging.handlers
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import traceback
from pythonjsonlogger import jsonlogger


class TradeLogger:
    """Specialized logger for trade execution and performance."""
    
    def __init__(self, log_dir: str = "logs/trades"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate logger for trades
        self.logger = logging.getLogger("trades")
        self.logger.setLevel(logging.INFO)
        
        # Trade log file with daily rotation
        trade_handler = logging.handlers.TimedRotatingFileHandler(
            filename=self.log_dir / "trades.log",
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        
        # JSON formatter for trades
        formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            rename_fields={'levelname': 'level', 'name': 'logger'}
        )
        trade_handler.setFormatter(formatter)
        self.logger.addHandler(trade_handler)
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade execution details."""
        self.logger.info("trade_executed", extra=trade_data)
    
    def log_order(self, order_data: Dict[str, Any]):
        """Log order placement details."""
        self.logger.info("order_placed", extra=order_data)
    
    def log_position(self, position_data: Dict[str, Any]):
        """Log position update."""
        self.logger.info("position_updated", extra=position_data)


class PerformanceLogger:
    """Logger for system performance metrics."""
    
    def __init__(self, log_dir: str = "logs/performance"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("performance")
        self.logger.setLevel(logging.INFO)
        
        # Performance log with size-based rotation
        perf_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "performance.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        formatter = jsonlogger.JsonFormatter()
        perf_handler.setFormatter(formatter)
        self.logger.addHandler(perf_handler)
    
    def log_latency(self, operation: str, latency_ms: float, metadata: Optional[Dict] = None):
        """Log operation latency."""
        data = {
            'operation': operation,
            'latency_ms': latency_ms,
            'timestamp': datetime.utcnow().isoformat()
        }
        if metadata:
            data.update(metadata)
        self.logger.info("latency_measurement", extra=data)
    
    def log_throughput(self, operation: str, items_per_second: float, metadata: Optional[Dict] = None):
        """Log throughput metrics."""
        data = {
            'operation': operation,
            'throughput': items_per_second,
            'timestamp': datetime.utcnow().isoformat()
        }
        if metadata:
            data.update(metadata)
        self.logger.info("throughput_measurement", extra=data)


class ErrorLogger:
    """Enhanced error logging with context capture."""
    
    def __init__(self, log_dir: str = "logs/errors"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("errors")
        self.logger.setLevel(logging.ERROR)
        
        # Error log file
        error_handler = logging.handlers.TimedRotatingFileHandler(
            filename=self.log_dir / "errors.log",
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        
        # Detailed formatter for errors
        formatter = jsonlogger.JsonFormatter()
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def log_exception(self, context: str, exception: Exception, additional_data: Optional[Dict] = None):
        """Log exception with full context."""
        error_data = {
            'context': context,
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if additional_data:
            error_data.update(additional_data)
        
        self.logger.error("exception_occurred", extra=error_data, exc_info=True)


def setup_logging(
    config_name: str = "production",
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = True,
    log_level: str = "INFO"
) -> Dict[str, logging.Logger]:
    """
    Setup comprehensive logging configuration.
    
    Args:
        config_name: Configuration name (development, staging, production)
        log_dir: Base directory for log files
        enable_console: Enable console output
        enable_file: Enable file logging
        enable_json: Enable JSON formatting
        log_level: Default log level
    
    Returns:
        Dictionary of configured loggers
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if enable_json:
            console_formatter = jsonlogger.JsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s'
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_file:
        # Main application log
        app_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_path / "application.log",
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        app_handler.setLevel(logging.DEBUG)
        
        if enable_json:
            app_formatter = jsonlogger.JsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(funcName)s %(message)s',
                timestamp=True
            )
        else:
            app_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s'
            )
        
        app_handler.setFormatter(app_formatter)
        root_logger.addHandler(app_handler)
    
    # Configure module-specific loggers
    loggers = {}
    
    # Module configurations
    module_configs = {
        'ares_orchestrator': logging.INFO,
        'data_pipeline': logging.INFO,
        'ml_models': logging.INFO,
        'backtesting': logging.INFO,
        'risk_management': logging.WARNING,
        'execution': logging.INFO,
        'monitoring': logging.DEBUG,
        'api': logging.INFO,
        'database': logging.WARNING,
        'websocket': logging.INFO
    }
    
    for module_name, level in module_configs.items():
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(level)
        loggers[module_name] = module_logger
    
    # Add specialized loggers
    loggers['trades'] = TradeLogger(log_dir=str(log_path))
    loggers['performance'] = PerformanceLogger(log_dir=str(log_path))
    loggers['errors'] = ErrorLogger(log_dir=str(log_path))
    
    # Configure third-party library logging
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Log startup message
    root_logger.info(
        "Logging initialized",
        extra={
            'config': config_name,
            'log_dir': str(log_path),
            'log_level': log_level,
            'handlers': {
                'console': enable_console,
                'file': enable_file,
                'json': enable_json
            }
        }
    )
    
    return loggers


class LogAggregator:
    """Aggregator for sending logs to ELK stack or similar systems."""
    
    def __init__(self, 
                 elasticsearch_host: str = "localhost",
                 elasticsearch_port: int = 9200,
                 index_prefix: str = "ares-logs"):
        self.es_host = elasticsearch_host
        self.es_port = elasticsearch_port
        self.index_prefix = index_prefix
        
        # Note: Requires elasticsearch package
        # from elasticsearch import Elasticsearch
        # self.es_client = Elasticsearch([{'host': elasticsearch_host, 'port': elasticsearch_port}])
    
    def ship_logs(self, log_file: str):
        """Ship logs to Elasticsearch."""
        # Implementation would parse log file and send to ES
        pass


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name."""
    return logging.getLogger(name)


# Convenience functions for common logging patterns
def log_function_call(logger: logging.Logger):
    """Decorator to log function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


def log_execution_time(logger: logging.Logger):
    """Decorator to log function execution time."""
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            logger.info(
                f"{func.__name__} execution time",
                extra={
                    'function': func.__name__,
                    'execution_time_ms': execution_time
                }
            )
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    loggers = setup_logging(
        config_name="development",
        log_dir="logs",
        enable_console=True,
        enable_file=True,
        enable_json=True,
        log_level="DEBUG"
    )
    
    # Test different loggers
    app_logger = get_logger("ares_orchestrator")
    app_logger.info("Application started")
    
    # Test trade logger
    trade_logger = loggers['trades']
    trade_logger.log_trade({
        'symbol': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
        'price': 150.50,
        'timestamp': datetime.utcnow().isoformat()
    })
    
    # Test performance logger
    perf_logger = loggers['performance']
    perf_logger.log_latency('model_inference', 45.2, {'model': 'ensemble'})
    
    # Test error logger
    error_logger = loggers['errors']
    try:
        1 / 0
    except Exception as e:
        error_logger.log_exception('test_division', e, {'test': True})