"""
ARES-7 v73 Symbol Mapper
심볼 변환 및 시장 추론
"""

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# 한국 주요 종목 매핑 (예시)
KR_SYMBOL_MAP = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035720": "카카오",
    "035420": "NAVER",
    "051910": "LG화학",
    "006400": "삼성SDI",
    "207940": "삼성바이오로직스",
    "005380": "현대차",
    "000270": "기아",
    "068270": "셀트리온"
}

# 미국 주요 종목 매핑 (예시)
US_SYMBOL_MAP = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms Inc.",
    "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000 ETF"
}


def infer_market(symbol: str) -> str:
    """
    심볼로부터 시장 추론
    
    Args:
        symbol: 종목 심볼
    
    Returns:
        "KR" (한국) 또는 "US" (미국)
    """
    
    if not symbol:
        return "US"
    
    # 6자리 숫자: 한국 주식
    if symbol.isdigit() and len(symbol) == 6:
        return "KR"
    
    # 알파벳: 미국 주식
    if symbol.isalpha():
        return "US"
    
    # 기본값
    return "US"


def get_symbol_name(symbol: str) -> Optional[str]:
    """
    심볼의 이름 조회
    
    Args:
        symbol: 종목 심볼
    
    Returns:
        종목명 (없으면 None)
    """
    
    market = infer_market(symbol)
    
    if market == "KR":
        return KR_SYMBOL_MAP.get(symbol)
    else:
        return US_SYMBOL_MAP.get(symbol)


def normalize_symbol(symbol: str, market: Optional[str] = None) -> str:
    """
    심볼 정규화
    
    Args:
        symbol: 원본 심볼
        market: 시장 ("KR" or "US", None이면 자동 추론)
    
    Returns:
        정규화된 심볼
    """
    
    if not symbol:
        return ""
    
    # 공백 제거, 대문자 변환
    symbol = symbol.strip().upper()
    
    # 시장 추론
    if market is None:
        market = infer_market(symbol)
    
    # 한국 주식: 6자리 패딩
    if market == "KR":
        if symbol.isdigit():
            return symbol.zfill(6)
    
    # 미국 주식: 대문자 유지
    return symbol


def convert_symbol(symbol: str, from_format: str, to_format: str) -> str:
    """
    심볼 포맷 변환
    
    Args:
        symbol: 원본 심볼
        from_format: 원본 포맷 ("kis", "ibkr", "yahoo" 등)
        to_format: 목표 포맷
    
    Returns:
        변환된 심볼
    """
    
    # 현재는 단순 구현
    # 실제로는 브로커별 심볼 포맷 차이 처리 필요
    
    if from_format == to_format:
        return symbol
    
    # KIS → IBKR
    if from_format == "kis" and to_format == "ibkr":
        # 한국 주식은 변환 불필요 (IBKR에서 한국 주식 지원 안 함)
        return symbol
    
    # IBKR → KIS
    if from_format == "ibkr" and to_format == "kis":
        return symbol
    
    # 기본값: 그대로 반환
    return symbol


def get_exchange(symbol: str) -> str:
    """
    심볼의 거래소 코드 반환
    
    Returns:
        거래소 코드 ("KOSPI", "KOSDAQ", "NASD", "NYSE", "AMEX")
    """
    
    market = infer_market(symbol)
    
    if market == "KR":
        # 한국: 기본 KOSPI
        # 실제로는 API 조회 또는 매핑 테이블 필요
        if symbol in ["035720", "035420"]:  # 카카오, 네이버
            return "KOSDAQ"
        return "KOSPI"
    
    else:
        # 미국: 기본 NASDAQ
        # 실제로는 심볼별 거래소 매핑 필요
        if symbol in ["BRK.A", "BRK.B", "JPM", "BAC"]:
            return "NYSE"
        return "NASD"


class SymbolMapper:
    """
    고급 심볼 매핑 클래스
    
    기능:
    - 심볼 정규화
    - 시장 추론
    - 브로커별 포맷 변환
    - 커스텀 매핑 지원
    """
    
    def __init__(self, custom_map: Optional[Dict[str, str]] = None):
        """
        Args:
            custom_map: 사용자 정의 심볼 매핑
                {
                    "AAPL": "애플",
                    "005930": "삼성전자"
                }
        """
        self.custom_map = custom_map or {}
        self.kr_map = KR_SYMBOL_MAP.copy()
        self.us_map = US_SYMBOL_MAP.copy()
        
        # 커스텀 매핑 병합
        for symbol, name in self.custom_map.items():
            market = infer_market(symbol)
            if market == "KR":
                self.kr_map[symbol] = name
            else:
                self.us_map[symbol] = name
    
    def get_name(self, symbol: str) -> Optional[str]:
        """심볼의 이름 조회"""
        market = infer_market(symbol)
        
        if market == "KR":
            return self.kr_map.get(symbol)
        else:
            return self.us_map.get(symbol)
    
    def normalize(self, symbol: str) -> str:
        """심볼 정규화"""
        return normalize_symbol(symbol)
    
    def add_mapping(self, symbol: str, name: str):
        """매핑 추가"""
        market = infer_market(symbol)
        
        if market == "KR":
            self.kr_map[symbol] = name
        else:
            self.us_map[symbol] = name
        
        logger.info(f"[SymbolMapper] Added mapping: {symbol} → {name}")
    
    def get_market(self, symbol: str) -> str:
        """시장 추론"""
        return infer_market(symbol)
    
    def get_exchange(self, symbol: str) -> str:
        """거래소 코드 반환"""
        return get_exchange(symbol)
