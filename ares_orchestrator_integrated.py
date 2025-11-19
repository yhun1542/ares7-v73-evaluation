# File: ares_orchestrator_integrated.py
# Purpose: Core orchestrator with real P&L tracking and database persistence
# Changes: Added real P&L calculation, database integration, position tracking, performance metrics

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import pickle
import gc
from concurrent.futures import ThreadPoolExecutor
import traceback

from momentum_engine import MomentumEngine
from phoenix_engine import PhoenixEngine
from risk_manager import RiskManager
from database import DatabaseManager
from monitoring_engine import MonitoringEngine

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    strategy: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def update_price(self, price: float):
        """Update current price and unrealized P&L"""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity
    
    def close_position(self, exit_price: float) -> float:
        """Close position and return realized P&L"""
        self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        return self.realized_pnl

@dataclass
class PerformanceMetrics:
    """Track performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    def update(self, trade_pnl: float):
        """Update metrics with new trade"""
        self.total_trades += 1
        self.total_pnl += trade_pnl
        
        if trade_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.win_rate = self.winning_trades / max(1, self.total_trades)
        self.daily_returns.append(trade_pnl)
        self.equity_curve.append(self.total_pnl)
        
        # Calculate drawdown
        if self.equity_curve:
            peak = max(self.equity_curve)
            current = self.equity_curve[-1]
            drawdown = (peak - current) / max(1, abs(peak)) if peak > 0 else 0
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Calculate Sharpe ratio
        if len(self.daily_returns) > 1:
            returns_array = np.array(self.daily_returns)
            if returns_array.std() > 0:
                self.sharpe_ratio = (returns_array.mean() / returns_array.std()) * np.sqrt(252)

class AresOrchestrator:
    """Main orchestrator coordinating all trading components"""
    
    def __init__(self, config: Dict, db_manager: DatabaseManager, monitoring: MonitoringEngine):
        self.config = config
        self.db_manager = db_manager
        self.monitoring = monitoring
        
        # Initialize components
        self.momentum_engine = None
        self.phoenix_engine = None
        self.risk_manager = None
        
        # Position and P&L tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.performance = PerformanceMetrics()
        
        # Order management
        self.pending_orders: Dict[str, Dict] = {}
        self.order_history: deque = deque(maxlen=1000)
        
        # State management
        self.is_initialized = False
        self.trading_enabled = True
        self.last_update = datetime.now()
        
        # Performance tracking
        self.signal_history: deque = deque(maxlen=5000)
        self.execution_times: deque = deque(maxlen=1000)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_ttl = 60  # seconds
        
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing ARES Orchestrator...")
            
            # Initialize engines
            self.momentum_engine = MomentumEngine(self.config.get('momentum', {}))
            await self.momentum_engine.initialize()
            
            self.phoenix_engine = PhoenixEngine(self.config.get('phoenix', {}))
            await self.phoenix_engine.initialize()
            
            self.risk_manager = RiskManager(self.config.get('risk', {}))
            await self.risk_manager.initialize()
            
            # Load existing positions from database
            await self.load_positions()
            
            # Load historical performance
            await self.load_performance_history()
            
            self.is_initialized = True
            logger.info("ARES Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def load_positions(self):
        """Load open positions from database"""
        try:
            positions_data = await self.db_manager.get_open_positions()
            
            for pos_data in positions_data:
                position = Position(
                    symbol=pos_data['symbol'],
                    quantity=pos_data['quantity'],
                    entry_price=pos_data['entry_price'],
                    entry_time=pos_data['entry_time'],
                    stop_loss=pos_data.get('stop_loss', 0),
                    take_profit=pos_data.get('take_profit', 0),
                    strategy=pos_data.get('strategy', ''),
                    metadata=pos_data.get('metadata', {})
                )
                self.positions[position.symbol] = position
            
            logger.info(f"Loaded {len(self.positions)} open positions")
            
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    async def load_performance_history(self):
        """Load historical performance metrics"""
        try:
            perf_data = await self.db_manager.get_performance_metrics()
            
            if perf_data:
                self.performance.total_trades = perf_data.get('total_trades', 0)
                self.performance.winning_trades = perf_data.get('winning_trades', 0)
                self.performance.losing_trades = perf_data.get('losing_trades', 0)
                self.performance.total_pnl = perf_data.get('total_pnl', 0.0)
                self.performance.max_drawdown = perf_data.get('max_drawdown', 0.0)
                self.performance.sharpe_ratio = perf_data.get('sharpe_ratio', 0.0)
                self.performance.equity_curve = perf_data.get('equity_curve', [])
                
            logger.info(f"Loaded performance history: {self.performance.total_trades} trades, P&L: ${self.performance.total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
    
    async def process_market_data(self, data: Dict) -> Optional[Dict]:
        """Process market data and generate trading signals"""
        start_time = datetime.now()
        
        try:
            if not self.is_initialized or not self.trading_enabled:
                return None
            
            symbol = data['symbol']
            price = data['price']
            
            # Update position prices
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
                await self.check_position_exits(symbol, price)
            
            # Generate signals from engines
            signals = await self.generate_signals(data)
            
            if not signals:
                return None
            
            # Combine and validate signals
            combined_signal = await self.combine_signals(signals)
            
            if combined_signal:
                # Risk check
                risk_approved = await self.risk_manager.evaluate_signal(
                    combined_signal,
                    self.positions,
                    self.performance
                )
                
                if risk_approved:
                    # Store signal
                    await self.store_signal(combined_signal)
                    
                    # Track execution time
                    execution_time = (datetime.now() - start_time).total_seconds()
                    self.execution_times.append(execution_time)
                    
                    # Update monitoring
                    self.monitoring.record_signal(combined_signal)
                    
                    return combined_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return None
    
    async def generate_signals(self, data: Dict) -> List[Dict]:
        """Generate signals from multiple engines"""
        signals = []
        
        try:
            # Run engines in parallel
            tasks = [
                self.momentum_engine.analyze(data),
                self.phoenix_engine.analyze(data)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Engine {i} error: {result}")
                elif result:
                    signals.append(result)
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    async def combine_signals(self, signals: List[Dict]) -> Optional[Dict]:
        """Combine multiple signals into a consensus signal"""
        if not signals:
            return None
        
        try:
            # Weight signals by confidence
            total_weight = sum(s.get('confidence', 0) for s in signals)
            
            if total_weight == 0:
                return None
            
            # Calculate weighted consensus
            symbol = signals[0]['symbol']
            
            weighted_action = defaultdict(float)
            for signal in signals:
                action = signal.get('action', 'hold')
                weight = signal.get('confidence', 0) / total_weight
                weighted_action[action] += weight
            
            # Get dominant action
            best_action = max(weighted_action.items(), key=lambda x: x[1])
            
            if best_action[1] < self.config.get('min_consensus_weight', 0.6):
                return None
            
            # Create combined signal
            combined = {
                'symbol': symbol,
                'action': best_action[0],
                'confidence': best_action[1],
                'quantity': self.calculate_position_size(symbol, best_action[1]),
                'strategy': 'consensus',
                'signals': signals,
                'timestamp': datetime.now()
            }
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate position size based on Kelly Criterion and risk limits"""
        try:
            # Get account equity
            account_equity = self.config.get('account_equity', 100000)
            current_equity = account_equity + self.performance.total_pnl
            
            # Kelly fraction
            win_rate = max(0.01, self.performance.win_rate)
            avg_win = max(0.01, abs(self.performance.avg_win)) if self.performance.avg_win else 1
            avg_loss = max(0.01, abs(self.performance.avg_loss)) if self.performance.avg_loss else 1
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
            
            # Adjust by confidence
            position_fraction = kelly_fraction * confidence
            
            # Apply risk limits
            max_position_size = current_equity * self.config.get('max_position_pct', 0.1)
            position_size = min(position_fraction * current_equity, max_position_size)
            
            # Round to reasonable size
            return round(position_size / 100) * 100
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.config.get('default_position_size', 1000)
    
    async def check_position_exits(self, symbol: str, current_price: float):
        """Check if any positions should be exited"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return
            
            should_exit = False
            exit_reason = ""
            
            # Check stop loss
            if position.stop_loss > 0:
                if (position.quantity > 0 and current_price <= position.stop_loss) or \
                   (position.quantity < 0 and current_price >= position.stop_loss):
                    should_exit = True
                    exit_reason = "stop_loss"
            
            # Check take profit
            if position.take_profit > 0:
                if (position.quantity > 0 and current_price >= position.take_profit) or \
                   (position.quantity < 0 and current_price <= position.take_profit):
                    should_exit = True
                    exit_reason = "take_profit"
            
            # Check time-based exit
            hold_time = (datetime.now() - position.entry_time).total_seconds() / 3600
            max_hold_time = self.config.get('max_hold_hours', 24)
            
            if hold_time > max_hold_time:
                should_exit = True
                exit_reason = "max_hold_time"
            
            if should_exit:
                await self.close_position(symbol, current_price, exit_reason)
                
        except Exception as e:
            logger.error(f"Error checking position exits: {e}")
    
    async def close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a position and record P&L"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return
            
            # Calculate P&L
            pnl = position.close_position(exit_price)
            
            # Update performance metrics
            self.performance.update(pnl)
            
            # Store closed position
            self.closed_positions.append(position)
            
            # Remove from open positions
            del self.positions[symbol]
            
            # Store in database
            await self.db_manager.close_position(
                symbol=symbol,
                exit_price=exit_price,
                pnl=pnl,
                reason=reason
            )
            
            # Log
            logger.info(f"Closed position {symbol}: P&L=${pnl:.2f}, Reason: {reason}")
            
            # Update monitoring
            self.monitoring.record_trade({
                'symbol': symbol,
                'pnl': pnl,
                'reason': reason,
                'exit_price': exit_price
            })
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    async def update_trade_status(self, trade_result: Dict):
        """Update trade status and create/update positions"""
        try:
            symbol = trade_result['signal']['symbol']
            action = trade_result['signal']['action']
            quantity = trade_result['signal']['quantity']
            price = trade_result['execution_price']
            
            if action == 'buy':
                # Create or add to position
                if symbol in self.positions:
                    # Average into existing position
                    pos = self.positions[symbol]
                    total_cost = pos.entry_price * pos.quantity + price * quantity
                    pos.quantity += quantity
                    pos.entry_price = total_cost / pos.quantity
                else:
                    # Create new position
                    position = Position(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=price,
                        entry_time=datetime.now(),
                        strategy=trade_result['signal'].get('strategy', ''),
                        stop_loss=price * (1 - self.config.get('stop_loss_pct', 0.02)),
                        take_profit=price * (1 + self.config.get('take_profit_pct', 0.05))
                    )
                    self.positions[symbol] = position
                    
            elif action == 'sell' and symbol in self.positions:
                # Reduce or close position
                pos = self.positions[symbol]
                if quantity >= abs(pos.quantity):
                    await self.close_position(symbol, price, 'manual_close')
                else:
                    pos.quantity -= quantity
            
            # Store trade in database
            await self.db_manager.store_trade(trade_result)
            
            # Update order history
            self.order_history.append(trade_result)
            
        except Exception as e:
            logger.error(f"Error updating trade status: {e}")
    
    async def store_signal(self, signal: Dict):
        """Store signal in database and history"""
        try:
            # Add to history
            self.signal_history.append(signal)
            
            # Store in database
            await self.db_manager.store_signal(signal)
            
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
    
    async def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        try:
            total_value = 0
            total_pnl = 0
            
            positions_list = []
            for symbol, pos in self.positions.items():
                pos_value = pos.current_price * pos.quantity
                total_value += pos_value
                total_pnl += pos.unrealized_pnl
                
                positions_list.append({
                    'symbol': symbol,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'value': pos_value
                })
            
            return {
                'positions': positions_list,
                'total_value': total_value,
                'total_unrealized_pnl': total_pnl,
                'total_realized_pnl': self.performance.total_pnl,
                'total_pnl': total_pnl + self.performance.total_pnl,
                'performance': {
                    'total_trades': self.performance.total_trades,
                    'win_rate': self.performance.win_rate,
                    'sharpe_ratio': self.performance.sharpe_ratio,
                    'max_drawdown': self.performance.max_drawdown
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("Cleaning up orchestrator...")
            
            # Save current state
            await self.save_state()
            
            # Cleanup engines
            if self.momentum_engine:
                await self.momentum_engine.cleanup()
            
            if self.phoenix_engine:
                await self.phoenix_engine.cleanup()
            
            if self.risk_manager:
                await self.risk_manager.cleanup()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Clear cache
            self.cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Orchestrator cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def save_state(self):
        """Save current state to database"""
        try:
            # Save performance metrics
            await self.db_manager.save_performance_metrics(self.performance.__dict__)
            
            # Save open positions
            for symbol, position in self.positions.items():
                await self.db_manager.save_position(position.__dict__)
            
            logger.info("State saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")