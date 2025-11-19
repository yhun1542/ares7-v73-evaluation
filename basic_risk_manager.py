"""
Basic Risk Manager for ARES-7 v73
Addresses P0 critical issue: No risk management framework

This implements essential risk controls to prevent catastrophic losses:
- Position limits
- Daily loss limits
- Portfolio-level risk monitoring
- Kill switch mechanism
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_pct: float = 0.10  # 10% of account per position
    max_daily_loss_pct: float = 0.02  # 2% daily loss limit
    max_total_exposure_pct: float = 1.0  # 100% total exposure
    max_leverage: float = 1.0  # No leverage by default
    max_correlation: float = 0.7  # Maximum correlation between positions


class BasicRiskManager:
    """
    Basic risk management system implementing essential controls
    
    Features:
    - Position size limits
    - Daily loss monitoring
    - Portfolio exposure tracking
    - Kill switch for emergency stops
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.daily_pnl = 0.0
        self.positions = {}
        self.kill_switch_active = False
        self.initial_account_value = None
        
        logger.info(f"Risk Manager initialized with limits: {self.limits}")
    
    def set_account_value(self, value: float):
        """Set initial account value for percentage calculations"""
        if self.initial_account_value is None:
            self.initial_account_value = value
            logger.info(f"Initial account value set: ${value:,.2f}")
    
    def check_position_limit(self, symbol: str, position_value: float, 
                            account_value: float) -> tuple[bool, str]:
        """
        Check if position size is within limits
        
        Returns:
            (is_allowed, reason)
        """
        if self.kill_switch_active:
            return False, "Kill switch is active"
        
        max_position_value = account_value * self.limits.max_position_pct
        
        if abs(position_value) > max_position_value:
            return False, (f"Position size ${abs(position_value):,.2f} exceeds "
                          f"limit ${max_position_value:,.2f} "
                          f"({self.limits.max_position_pct*100:.1f}% of account)")
        
        return True, "OK"
    
    def check_daily_loss_limit(self, current_pnl: float, 
                               account_value: float) -> tuple[bool, str]:
        """
        Check if daily loss is within limits
        
        Returns:
            (is_allowed, reason)
        """
        max_daily_loss = account_value * self.limits.max_daily_loss_pct
        
        if current_pnl < -max_daily_loss:
            self.activate_kill_switch(f"Daily loss limit breached: ${abs(current_pnl):,.2f}")
            return False, (f"Daily loss ${abs(current_pnl):,.2f} exceeds "
                          f"limit ${max_daily_loss:,.2f} "
                          f"({self.limits.max_daily_loss_pct*100:.1f}% of account)")
        
        return True, "OK"
    
    def check_total_exposure(self, positions: Dict[str, float], 
                            account_value: float) -> tuple[bool, str]:
        """
        Check if total portfolio exposure is within limits
        
        Returns:
            (is_allowed, reason)
        """
        total_exposure = sum(abs(v) for v in positions.values())
        max_exposure = account_value * self.limits.max_total_exposure_pct
        
        if total_exposure > max_exposure:
            return False, (f"Total exposure ${total_exposure:,.2f} exceeds "
                          f"limit ${max_exposure:,.2f} "
                          f"({self.limits.max_total_exposure_pct*100:.1f}% of account)")
        
        return True, "OK"
    
    def update_position(self, symbol: str, value: float):
        """Update position tracking"""
        self.positions[symbol] = value
        logger.debug(f"Position updated: {symbol} = ${value:,.2f}")
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl = pnl
        logger.debug(f"Daily P&L updated: ${pnl:,.2f}")
    
    def reset_daily_pnl(self):
        """Reset daily P&L (call at start of each trading day)"""
        logger.info(f"Resetting daily P&L. Previous: ${self.daily_pnl:,.2f}")
        self.daily_pnl = 0.0
    
    def activate_kill_switch(self, reason: str):
        """Activate emergency kill switch"""
        self.kill_switch_active = True
        logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
        logger.critical("All trading halted. Manual intervention required.")
    
    def deactivate_kill_switch(self):
        """Deactivate kill switch (requires manual intervention)"""
        self.kill_switch_active = False
        logger.warning("Kill switch deactivated. Trading resumed.")
    
    def validate_trade(self, symbol: str, order_value: float, 
                      current_pnl: float, account_value: float) -> tuple[bool, str]:
        """
        Comprehensive trade validation
        
        Returns:
            (is_allowed, reason)
        """
        # Check kill switch
        if self.kill_switch_active:
            return False, "Kill switch is active - no trading allowed"
        
        # Check position limit
        allowed, reason = self.check_position_limit(symbol, order_value, account_value)
        if not allowed:
            logger.warning(f"Trade rejected for {symbol}: {reason}")
            return False, reason
        
        # Check daily loss limit
        allowed, reason = self.check_daily_loss_limit(current_pnl, account_value)
        if not allowed:
            logger.error(f"Trade rejected for {symbol}: {reason}")
            return False, reason
        
        # Check total exposure
        new_positions = self.positions.copy()
        new_positions[symbol] = new_positions.get(symbol, 0) + order_value
        allowed, reason = self.check_total_exposure(new_positions, account_value)
        if not allowed:
            logger.warning(f"Trade rejected for {symbol}: {reason}")
            return False, reason
        
        return True, "Trade approved"
    
    def get_risk_metrics(self, account_value: float) -> Dict:
        """Get current risk metrics"""
        total_exposure = sum(abs(v) for v in self.positions.values())
        exposure_pct = (total_exposure / account_value * 100) if account_value > 0 else 0
        daily_loss_pct = (self.daily_pnl / account_value * 100) if account_value > 0 else 0
        
        return {
            'account_value': account_value,
            'total_exposure': total_exposure,
            'exposure_pct': exposure_pct,
            'daily_pnl': self.daily_pnl,
            'daily_loss_pct': daily_loss_pct,
            'num_positions': len(self.positions),
            'kill_switch_active': self.kill_switch_active,
            'limits': {
                'max_position_pct': self.limits.max_position_pct * 100,
                'max_daily_loss_pct': self.limits.max_daily_loss_pct * 100,
                'max_exposure_pct': self.limits.max_total_exposure_pct * 100,
            }
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize risk manager
    risk_mgr = BasicRiskManager()
    
    # Set account value
    account_value = 100000.0
    risk_mgr.set_account_value(account_value)
    
    # Test position limit
    print("\n=== Testing Position Limit ===")
    allowed, reason = risk_mgr.validate_trade("AAPL", 15000, 0, account_value)
    print(f"$15k position: {allowed} - {reason}")
    
    # Test daily loss limit
    print("\n=== Testing Daily Loss Limit ===")
    allowed, reason = risk_mgr.validate_trade("MSFT", 5000, -2500, account_value)
    print(f"Trade with $2.5k loss: {allowed} - {reason}")
    
    # Get risk metrics
    print("\n=== Risk Metrics ===")
    metrics = risk_mgr.get_risk_metrics(account_value)
    for key, value in metrics.items():
        print(f"{key}: {value}")
