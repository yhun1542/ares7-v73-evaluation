"""
ARES-7 v73 Broker Router
심볼에 따른 브로커 라우팅 로직
"""

import logging

logger = logging.getLogger(__name__)


def choose_broker(symbol: str) -> str:
    """
    심볼에 따라 적절한 브로커(시장) 선택
    
    Args:
        symbol: 종목 심볼
    
    Returns:
        "KR" (한국 시장) 또는 "US" (미국 시장)
    
    규칙:
        - 6자리 숫자: 한국 주식 (예: 005930 = 삼성전자)
        - 알파벳: 미국 주식 (예: AAPL, SPY)
        - 기타: 기본값 US
    """
    
    # 한국 주식: 6자리 숫자
    if symbol.isdigit() and len(symbol) == 6:
        return "KR"
    
    # 미국 주식: 알파벳
    if symbol.isalpha():
        return "US"
    
    # 기본값: 미국
    return "US"


def infer_market(symbol: str) -> str:
    """
    choose_broker의 alias (하위 호환성)
    """
    return choose_broker(symbol)


def get_exchange_for_symbol(symbol: str) -> str:
    """
    심볼에 따른 거래소 코드 반환
    
    Returns:
        "KOSPI", "KOSDAQ", "NASD", "NYSE", "AMEX" 등
    """
    
    market = choose_broker(symbol)
    
    if market == "KR":
        # 한국 주식: 기본적으로 KOSPI
        # 실제로는 API로 조회하거나 매핑 테이블 필요
        return "KOSPI"
    
    else:
        # 미국 주식: 기본적으로 NASDAQ
        # 실제로는 심볼별 거래소 매핑 필요
        return "NASD"


def validate_symbol(symbol: str) -> bool:
    """
    심볼 유효성 검증
    
    Returns:
        True if valid, False otherwise
    """
    
    if not symbol or not isinstance(symbol, str):
        return False
    
    # 공백 제거
    symbol = symbol.strip()
    
    if len(symbol) == 0:
        return False
    
    # 한국 주식: 6자리 숫자
    if symbol.isdigit():
        return len(symbol) == 6
    
    # 미국 주식: 1~5자리 알파벳
    if symbol.isalpha():
        return 1 <= len(symbol) <= 5
    
    # 기타 (예: BRK.B)
    return True


class BrokerRouter:
    """
    고급 브로커 라우팅 클래스
    
    기능:
    - 심볼별 브로커 매핑
    - 사용자 정의 라우팅 규칙
    - 브로커 우선순위 설정
    """
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: 라우팅 설정
                {
                    "default_us": "ibkr",  # 미국 주식 기본 브로커
                    "default_kr": "kis",   # 한국 주식 기본 브로커
                    "overrides": {         # 특정 심볼 오버라이드
                        "AAPL": "kis",
                        "005930": "kis"
                    }
                }
        """
        self.config = config or {}
        self.default_us = self.config.get("default_us", "ibkr")
        self.default_kr = self.config.get("default_kr", "kis")
        self.overrides = self.config.get("overrides", {})
    
    def route(self, symbol: str) -> str:
        """
        심볼에 대한 브로커 결정
        
        Returns:
            브로커 이름 ("kis" 또는 "ibkr")
        """
        
        # 오버라이드 체크
        if symbol in self.overrides:
            return self.overrides[symbol]
        
        # 시장 판단
        market = choose_broker(symbol)
        
        if market == "KR":
            return self.default_kr
        else:
            return self.default_us
    
    def add_override(self, symbol: str, broker: str):
        """특정 심볼에 대한 브로커 오버라이드 추가"""
        self.overrides[symbol] = broker
        logger.info(f"[BrokerRouter] Override added: {symbol} → {broker}")
    
    def remove_override(self, symbol: str):
        """오버라이드 제거"""
        if symbol in self.overrides:
            del self.overrides[symbol]
            logger.info(f"[BrokerRouter] Override removed: {symbol}")
