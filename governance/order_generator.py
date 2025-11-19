"""
ARES-7 v73 Order Generator
시그널과 포지션을 비교하여 실제 주문 생성
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Broker layer imports
try:
    from brokers.execution_model import ExecutionModelV73, ExecutionParams
    from brokers.symbol_mapper import infer_market
except ImportError:
    # Fallback for testing
    ExecutionModelV73 = None
    infer_market = lambda x: "US" if x.isalpha() else "KR"

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """주문 객체"""
    symbol: str
    side: str           # "BUY" or "SELL"
    qty: float
    order_type: str     # "MARKET" or "LIMIT"
    price: float = 0.0  # 0 = market order
    reason: str = ""
    signal: float = 0.0
    confidence: float = 0.0


class OrderGenerator:
    """
    시그널 → 주문 변환 엔진
    
    기능:
    - 시그널과 현재 포지션 비교
    - 목표 포지션 계산
    - 실행 가능한 주문 생성
    - 최소 거래 단위 적용
    """
    
    def __init__(
        self,
        min_order_value: float = 1000.0,     # 최소 주문 금액 ($)
        min_position_change: float = 0.05,    # 최소 포지션 변경 비율 (5%)
        max_orders_per_batch: int = 50,       # 배치당 최대 주문 수
        use_limit_orders: bool = False,       # LIMIT 주문 사용 여부
        limit_offset_pct: float = 0.001       # LIMIT 가격 오프셋 (0.1%)
    ):
        self.min_order_value = min_order_value
        self.min_position_change = min_position_change
        self.max_orders_per_batch = max_orders_per_batch
        self.use_limit_orders = use_limit_orders
        self.limit_offset_pct = limit_offset_pct
    
    def generate_orders(
        self,
        signals: Dict[str, float],          # symbol → signal (-1 ~ 1)
        target_sizes: Dict[str, float],     # symbol → target_notional ($)
        current_positions: Dict[str, float], # symbol → current_qty
        current_prices: Dict[str, float]     # symbol → last_price
    ) -> List[Order]:
        """
        시그널과 목표 포지션을 기반으로 주문 생성
        
        Args:
            signals: 각 심볼의 시그널 (-1 ~ 1)
            target_sizes: 각 심볼의 목표 포지션 크기 (달러 기준)
            current_positions: 현재 보유 수량
            current_prices: 현재 가격
        
        Returns:
            주문 리스트
        """
        orders = []
        
        for symbol in signals.keys():
            signal = signals.get(symbol, 0.0)
            target_notional = target_sizes.get(symbol, 0.0)
            current_qty = current_positions.get(symbol, 0.0)
            price = current_prices.get(symbol, 0.0)
            
            if price <= 0:
                logger.warning(f"[OrderGen] {symbol}: Invalid price {price}, skipping")
                continue
            
            # 현재 포지션 가치
            current_notional = current_qty * price
            
            # 목표 수량 계산
            target_qty = target_notional / price if target_notional != 0 else 0.0
            
            # 변경 수량
            delta_qty = target_qty - current_qty
            delta_notional = abs(delta_qty * price)
            
            # 최소 주문 금액 체크
            if delta_notional < self.min_order_value:
                logger.debug(
                    f"[OrderGen] {symbol}: Delta ${delta_notional:.2f} < "
                    f"min ${self.min_order_value}, skipping"
                )
                continue
            
            # 최소 변경 비율 체크 (기존 포지션이 있는 경우)
            if abs(current_notional) > 0:
                change_pct = abs(delta_notional / abs(current_notional))
                if change_pct < self.min_position_change:
                    logger.debug(
                        f"[OrderGen] {symbol}: Change {change_pct:.2%} < "
                        f"min {self.min_position_change:.2%}, skipping"
                    )
                    continue
            
            # 주문 생성
            order = self._create_order(
                symbol=symbol,
                delta_qty=delta_qty,
                price=price,
                signal=signal,
                reason=self._get_order_reason(delta_qty, current_qty, target_qty)
            )
            
            if order:
                orders.append(order)
        
        # 최대 주문 수 제한
        if len(orders) > self.max_orders_per_batch:
            logger.warning(
                f"[OrderGen] Generated {len(orders)} orders, "
                f"limiting to {self.max_orders_per_batch}"
            )
            # 신호 강도 기준으로 정렬하여 상위 N개만 선택
            orders = sorted(orders, key=lambda x: abs(x.signal), reverse=True)
            orders = orders[:self.max_orders_per_batch]
        
        logger.info(f"[OrderGen] Generated {len(orders)} orders")
        return orders
    
    def _create_order(
        self,
        symbol: str,
        delta_qty: float,
        price: float,
        signal: float,
        reason: str
    ) -> Optional[Order]:
        """개별 주문 생성"""
        
        if abs(delta_qty) < 1e-6:
            return None
        
        # 매수/매도 결정
        side = "BUY" if delta_qty > 0 else "SELL"
        qty = abs(delta_qty)
        
        # 주문 타입 및 가격
        if self.use_limit_orders:
            order_type = "LIMIT"
            # 매수는 현재가보다 약간 낮게, 매도는 약간 높게
            if side == "BUY":
                order_price = price * (1 - self.limit_offset_pct)
            else:
                order_price = price * (1 + self.limit_offset_pct)
        else:
            order_type = "MARKET"
            order_price = 0.0  # Market order
        
        return Order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            price=order_price,
            reason=reason,
            signal=signal
        )
    
    def _get_order_reason(
        self,
        delta_qty: float,
        current_qty: float,
        target_qty: float
    ) -> str:
        """주문 사유 생성"""
        
        if current_qty == 0:
            return "NEW_ENTRY"
        elif target_qty == 0:
            return "FULL_EXIT"
        elif abs(delta_qty) > abs(current_qty):
            return "SCALE_IN" if np.sign(delta_qty) == np.sign(current_qty) else "REVERSE"
        else:
            return "SCALE_OUT" if abs(target_qty) < abs(current_qty) else "ADJUST"
    
    def orders_to_dataframe(self, orders: List[Order]) -> pd.DataFrame:
        """주문 리스트를 DataFrame으로 변환"""
        if not orders:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "symbol": o.symbol,
                "side": o.side,
                "qty": o.qty,
                "order_type": o.order_type,
                "price": o.price,
                "reason": o.reason,
                "signal": o.signal,
                "confidence": o.confidence
            }
            for o in orders
        ])
    
    def generate_flatten_orders(
        self,
        current_positions: Dict[str, float],  # symbol → qty
        current_prices: Dict[str, float]      # symbol → price
    ) -> List[Order]:
        """
        모든 포지션을 청산하는 주문 생성
        
        Args:
            current_positions: 현재 포지션 (symbol → qty)
            current_prices: 현재 가격 (symbol → price)
        
        Returns:
            청산 주문 리스트
        """
        orders = []
        
        for symbol, qty in current_positions.items():
            if abs(qty) < 1e-6:
                continue
            
            price = current_prices.get(symbol, 0.0)
            if price <= 0:
                logger.warning(f"[OrderGen] {symbol}: Invalid price for flatten, skipping")
                continue
            
            # 포지션 반대 방향으로 주문
            side = "SELL" if qty > 0 else "BUY"
            
            order = Order(
                symbol=symbol,
                side=side,
                qty=abs(qty),
                order_type="MARKET",
                price=0.0,
                reason="FLATTEN_ALL",
                signal=0.0
            )
            
            orders.append(order)
        
        logger.info(f"[OrderGen] Generated {len(orders)} flatten orders")
        return orders
