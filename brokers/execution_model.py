"""
ARES-7 v73 Execution Model
거래 비용 추정 및 실행 가능성 판단
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionParams:
    """
    거래 실행 파라미터
    """
    
    # 기본 스프레드 (bps)
    base_spread_bps: float = 5.0
    
    # 변동성 비용 계수 (bps per 1% vol)
    vol_sigma_bps: float = 3.0
    
    # 환전 비용 (미국 주식)
    fx_cost_bps: float = 2.0
    
    # 시장 충격 계수
    impact_coef: float = 0.5
    
    # 최대 거래 비용 (bps)
    max_tc_bps: float = 30.0
    
    # 최소 거래 비용 (bps)
    min_tc_bps: float = 1.0


class ExecutionModelV73:
    """
    거래 실행 모델
    
    기능:
    - 거래 비용 추정 (스프레드 + 변동성 + 시장 충격)
    - 거래 가치 판단 (alpha vs transaction cost)
    - 슬리피지 추정
    """
    
    def __init__(self, params: Optional[ExecutionParams] = None):
        """
        Args:
            params: 실행 파라미터
        """
        self.params = params or ExecutionParams()
    
    def estimate_cost_bps(
        self,
        symbol: str,
        recent_ret: pd.Series,
        size_ratio: float,
        market: Optional[str] = None
    ) -> float:
        """
        거래 비용 추정 (basis points)
        
        Args:
            symbol: 종목 심볼
            recent_ret: 최근 수익률 시계열
            size_ratio: 포트폴리오 대비 거래 크기 비율
            market: 시장 ("KR" or "US")
        
        Returns:
            거래 비용 (bps)
        """
        
        # 기본 스프레드
        tc = self.params.base_spread_bps
        
        # 변동성 비용
        if len(recent_ret) > 10:
            vol = recent_ret.std()
            tc += self.params.vol_sigma_bps * vol * 100  # % to bps
        
        # 시장별 추가 비용
        if market is None:
            # 심볼로 시장 추론
            if symbol.isdigit() and len(symbol) == 6:
                market = "KR"
            else:
                market = "US"
        
        if market == "US":
            # 미국 주식: 환전 비용
            tc += self.params.fx_cost_bps
        
        # 시장 충격 (거래 크기에 비례)
        impact = self.params.impact_coef * size_ratio * 10000  # ratio to bps
        tc += impact
        
        # 범위 제한
        tc = max(self.params.min_tc_bps, min(tc, self.params.max_tc_bps))
        
        return float(tc)
    
    def worth_trade(
        self,
        alpha_bps: float,
        tc_bps: float,
        min_edge: float = 1.5
    ) -> bool:
        """
        거래 가치 판단
        
        Args:
            alpha_bps: 예상 알파 (bps)
            tc_bps: 거래 비용 (bps)
            min_edge: 최소 엣지 비율 (alpha / tc)
        
        Returns:
            True if worth trading
        """
        
        if tc_bps <= 0:
            return alpha_bps > 0
        
        edge = alpha_bps / tc_bps
        
        return edge >= min_edge
    
    def estimate_slippage_bps(
        self,
        symbol: str,
        order_size: float,
        avg_volume: float,
        volatility: float
    ) -> float:
        """
        슬리피지 추정
        
        Args:
            symbol: 종목 심볼
            order_size: 주문 크기 ($)
            avg_volume: 평균 거래량 ($)
            volatility: 변동성 (일간 표준편차)
        
        Returns:
            슬리피지 (bps)
        """
        
        if avg_volume <= 0:
            return self.params.max_tc_bps
        
        # 거래량 대비 주문 크기
        volume_ratio = order_size / avg_volume
        
        # 기본 슬리피지
        slippage = self.params.base_spread_bps / 2
        
        # 거래량 충격
        slippage += self.params.impact_coef * volume_ratio * 10000
        
        # 변동성 충격
        slippage += volatility * 100 * self.params.vol_sigma_bps
        
        # 범위 제한
        slippage = max(0, min(slippage, self.params.max_tc_bps))
        
        return float(slippage)
    
    def estimate_fill_probability(
        self,
        order_type: str,
        limit_price: Optional[float],
        current_price: float,
        volatility: float
    ) -> float:
        """
        체결 확률 추정
        
        Args:
            order_type: "MARKET" or "LIMIT"
            limit_price: 지정가 (LIMIT 주문인 경우)
            current_price: 현재가
            volatility: 변동성
        
        Returns:
            체결 확률 (0~1)
        """
        
        if order_type == "MARKET":
            return 1.0
        
        if limit_price is None or current_price <= 0:
            return 0.5
        
        # 가격 차이 (%)
        price_diff_pct = abs(limit_price - current_price) / current_price
        
        # 변동성 대비 가격 차이
        z_score = price_diff_pct / max(volatility, 0.001)
        
        # 체결 확률 (정규분포 가정)
        from scipy.stats import norm
        fill_prob = norm.cdf(z_score)
        
        return float(fill_prob)
    
    def optimize_order_size(
        self,
        target_size: float,
        max_size_per_order: float,
        avg_volume: float,
        max_volume_ratio: float = 0.1
    ) -> list:
        """
        주문 크기 최적화 (분할 주문)
        
        Args:
            target_size: 목표 주문 크기 ($)
            max_size_per_order: 주문당 최대 크기 ($)
            avg_volume: 평균 거래량 ($)
            max_volume_ratio: 거래량 대비 최대 비율
        
        Returns:
            분할 주문 크기 리스트
        """
        
        # 거래량 제약
        max_by_volume = avg_volume * max_volume_ratio
        
        # 실제 최대 크기
        max_size = min(max_size_per_order, max_by_volume)
        
        if max_size <= 0:
            return []
        
        # 분할 주문 수
        num_orders = int(np.ceil(target_size / max_size))
        
        # 균등 분할
        sizes = [target_size / num_orders] * num_orders
        
        return sizes
    
    def estimate_total_cost(
        self,
        symbol: str,
        order_size: float,
        current_price: float,
        recent_ret: pd.Series,
        avg_volume: float
    ) -> dict:
        """
        총 거래 비용 추정
        
        Returns:
            {
                "spread_bps": float,
                "impact_bps": float,
                "slippage_bps": float,
                "total_bps": float,
                "total_dollar": float
            }
        """
        
        # 변동성 계산
        vol = recent_ret.std() if len(recent_ret) > 0 else 0.02
        
        # 포트폴리오 대비 비율 (가정: 1M 포트폴리오)
        portfolio_value = 1_000_000
        size_ratio = order_size / portfolio_value
        
        # 스프레드 + 충격
        tc_bps = self.estimate_cost_bps(symbol, recent_ret, size_ratio)
        
        # 슬리피지
        slippage_bps = self.estimate_slippage_bps(
            symbol, order_size, avg_volume, vol
        )
        
        # 총 비용
        total_bps = tc_bps + slippage_bps
        total_dollar = order_size * total_bps / 10000
        
        return {
            "spread_bps": self.params.base_spread_bps,
            "impact_bps": tc_bps - self.params.base_spread_bps,
            "slippage_bps": slippage_bps,
            "total_bps": total_bps,
            "total_dollar": total_dollar
        }
