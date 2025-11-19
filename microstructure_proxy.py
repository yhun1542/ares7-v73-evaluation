#!/usr/bin/env python3
"""
ARES7 v73 마이크로구조 프록시 구현

Level 2 orderbook 데이터가 없을 때 OHLCV로부터 마이크로구조 지표를 추정합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class MicrostructureProxy:
    """
    OHLCV 데이터로부터 마이크로구조 지표를 추정하는 클래스
    """
    
    def __init__(self, vol_window: int = 20):
        """
        Args:
            vol_window: 변동성 계산 윈도우
        """
        self.vol_window = vol_window
    
    def calculate_spread_proxy(self, df: pd.DataFrame) -> pd.Series:
        """
        스프레드 프록시 계산
        
        Formula: (high - low) / close
        의미: 일중 변동성을 스프레드로 근사
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            spread 프록시 시리즈
        """
        if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
            logger.warning("Missing columns for spread proxy")
            return pd.Series(0.0, index=df.index)
        
        spread = (df["high"] - df["low"]) / df["close"]
        spread = spread.fillna(0.0)
        
        # Normalize to [-1, 1] range
        spread_norm = np.tanh(spread * 10)
        
        return spread_norm
    
    def calculate_depth_imbalance_proxy(self, df: pd.DataFrame) -> pd.Series:
        """
        호가 불균형 프록시 계산
        
        Formula: (close - low) / (high - low)
        의미: 종가가 고가/저가 중 어디에 가까운지 (매수/매도 압력)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            depth_imbalance 프록시 시리즈
        """
        if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
            logger.warning("Missing columns for depth imbalance proxy")
            return pd.Series(0.0, index=df.index)
        
        range_val = df["high"] - df["low"]
        range_val = range_val.replace(0, np.nan)  # Avoid division by zero
        
        imbalance = (df["close"] - df["low"]) / range_val
        imbalance = imbalance.fillna(0.5)  # Neutral if no range
        
        # Convert to [-1, 1] range (0.5 → 0, 1 → 1, 0 → -1)
        imbalance_norm = (imbalance - 0.5) * 2
        
        return imbalance_norm
    
    def calculate_order_flow_imbalance_proxy(self, df: pd.DataFrame) -> pd.Series:
        """
        주문 흐름 불균형 프록시 계산
        
        Formula: (close - open) / (high - low)
        의미: 시가 대비 종가의 상대적 위치 (순 매수/매도 흐름)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            order_flow_imbalance 프록시 시리즈
        """
        if "open" not in df.columns or "close" not in df.columns:
            logger.warning("Missing columns for order flow imbalance proxy")
            return pd.Series(0.0, index=df.index)
        
        if "high" not in df.columns or "low" not in df.columns:
            logger.warning("Missing columns for order flow imbalance proxy")
            return pd.Series(0.0, index=df.index)
        
        range_val = df["high"] - df["low"]
        range_val = range_val.replace(0, np.nan)
        
        flow = (df["close"] - df["open"]) / range_val
        flow = flow.fillna(0.0)
        
        # Already in [-1, 1] range approximately
        flow_norm = np.tanh(flow * 2)
        
        return flow_norm
    
    def calculate_tick_direction_proxy(self, df: pd.DataFrame) -> pd.Series:
        """
        틱 방향 프록시 계산
        
        Formula: sign(close - close_prev)
        의미: 가격 변화 방향
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            tick_direction 프록시 시리즈
        """
        if "close" not in df.columns:
            logger.warning("Missing close column for tick direction proxy")
            return pd.Series(0.0, index=df.index)
        
        close_diff = df["close"].diff()
        tick = np.sign(close_diff)
        tick = tick.fillna(0.0)
        
        return tick
    
    def calculate_volatility_proxy(self, df: pd.DataFrame) -> pd.Series:
        """
        변동성 프록시 계산
        
        Formula: rolling std of returns
        의미: 최근 수익률의 표준편차
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            volatility 프록시 시리즈
        """
        if "close" not in df.columns:
            logger.warning("Missing close column for volatility proxy")
            return pd.Series(0.0, index=df.index)
        
        returns = df["close"].pct_change()
        vol = returns.rolling(window=self.vol_window, min_periods=1).std()
        vol = vol.fillna(0.0)
        
        # Normalize
        vol_norm = np.tanh(vol * 50)
        
        return vol_norm
    
    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame에 모든 마이크로구조 프록시 추가
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            마이크로구조 컬럼이 추가된 DataFrame
        """
        df = df.copy()
        
        logger.info("Calculating microstructure proxies...")
        
        df["spread"] = self.calculate_spread_proxy(df)
        df["depth_imbalance"] = self.calculate_depth_imbalance_proxy(df)
        df["order_flow_imbalance"] = self.calculate_order_flow_imbalance_proxy(df)
        df["tick_direction"] = self.calculate_tick_direction_proxy(df)
        df["volatility"] = self.calculate_volatility_proxy(df)
        
        logger.info("Microstructure proxies added successfully")
        
        return df
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        마이크로구조 피처가 올바르게 추가되었는지 검증
        
        Args:
            df: DataFrame
            
        Returns:
            검증 결과 딕셔너리
        """
        required_cols = [
            "spread",
            "depth_imbalance",
            "order_flow_imbalance",
            "tick_direction",
            "volatility"
        ]
        
        validation = {}
        for col in required_cols:
            exists = col in df.columns
            non_zero = df[col].abs().sum() > 0 if exists else False
            in_range = (df[col].abs() <= 1.5).all() if exists else False
            
            validation[col] = {
                "exists": exists,
                "non_zero": non_zero,
                "in_range": in_range,
                "valid": exists and non_zero and in_range
            }
        
        return validation


def test_microstructure_proxy():
    """
    마이크로구조 프록시 테스트
    """
    print("=" * 80)
    print("마이크로구조 프록시 테스트")
    print("=" * 80)
    print()
    
    # 샘플 데이터 생성
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="1D")
    np.random.seed(42)
    
    base_price = 100
    returns = np.random.randn(100) * 0.02
    close_prices = base_price * (1 + returns).cumprod()
    
    df = pd.DataFrame({
        "date": dates,
        "open": close_prices * (1 + np.random.randn(100) * 0.005),
        "high": close_prices * (1 + np.abs(np.random.randn(100)) * 0.01),
        "low": close_prices * (1 - np.abs(np.random.randn(100)) * 0.01),
        "close": close_prices,
        "volume": np.random.randint(1000000, 10000000, 100)
    })
    
    print("샘플 데이터:")
    print(df.head())
    print()
    
    # 프록시 계산
    proxy = MicrostructureProxy(vol_window=20)
    df_with_features = proxy.add_microstructure_features(df)
    
    print("마이크로구조 피처 추가 완료:")
    print(df_with_features[["date", "close", "spread", "depth_imbalance", 
                            "order_flow_imbalance", "tick_direction", "volatility"]].tail())
    print()
    
    # 검증
    validation = proxy.validate_features(df_with_features)
    
    print("검증 결과:")
    print("-" * 80)
    for feature, result in validation.items():
        status = "✅" if result["valid"] else "❌"
        print(f"{status} {feature}")
        print(f"   존재: {result['exists']}")
        print(f"   Non-zero: {result['non_zero']}")
        print(f"   범위 내: {result['in_range']}")
    print()
    
    # 통계
    print("통계:")
    print("-" * 80)
    for col in ["spread", "depth_imbalance", "order_flow_imbalance", "tick_direction", "volatility"]:
        print(f"{col}:")
        print(f"  평균: {df_with_features[col].mean():.4f}")
        print(f"  표준편차: {df_with_features[col].std():.4f}")
        print(f"  최소: {df_with_features[col].min():.4f}")
        print(f"  최대: {df_with_features[col].max():.4f}")
    print()
    
    print("=" * 80)
    print("테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    test_microstructure_proxy()
