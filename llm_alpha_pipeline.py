#!/usr/bin/env python3
"""
ARES7 v73 LLM 알파 파이프라인

Anthropic Claude를 사용하여 뉴스, 공시, 경제 지표를 분석하고 알파 신호를 생성합니다.
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


class LLMAlphaPipeline:
    """
    LLM 기반 알파 생성 파이프라인
    """
    
    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        news_api_key: Optional[str] = None,
        sec_api_key: Optional[str] = None
    ):
        """
        Args:
            anthropic_api_key: Anthropic API 키
            news_api_key: News API 키
            sec_api_key: SEC API 키
        """
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.news_api_key = news_api_key or os.getenv("NEWS_API_KEY")
        self.sec_api_key = sec_api_key or os.getenv("SEC_API_KEY")
        
        if not self.anthropic_api_key:
            logger.warning("Anthropic API key not found")
        if not self.news_api_key:
            logger.warning("News API key not found")
        if not self.sec_api_key:
            logger.warning("SEC API key not found")
    
    def fetch_news(self, symbol: str, lookback_days: int = 7) -> List[Dict[str, Any]]:
        """
        뉴스 데이터 가져오기
        
        Args:
            symbol: 종목 코드
            lookback_days: 조회 기간 (일)
            
        Returns:
            뉴스 리스트
        """
        if not self.news_api_key or self.news_api_key == "your_news_api_key":
            logger.warning("News API key not configured")
            return []
        
        try:
            # Get company name from symbol (simplified)
            company_map = {
                "AAPL": "Apple",
                "MSFT": "Microsoft",
                "GOOGL": "Google",
                "AMZN": "Amazon",
                "TSLA": "Tesla",
                "NVDA": "NVIDIA",
                "META": "Meta"
            }
            company_name = company_map.get(symbol, symbol)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": company_name,
                "apiKey": self.news_api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "from": (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                logger.info(f"Fetched {len(articles)} news articles for {symbol}")
                return articles
        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {e}")
        
        return []
    
    def fetch_sec_filings(self, symbol: str, lookback_days: int = 30) -> List[Dict[str, Any]]:
        """
        SEC 공시 데이터 가져오기
        
        Args:
            symbol: 종목 코드
            lookback_days: 조회 기간 (일)
            
        Returns:
            공시 리스트
        """
        if not self.sec_api_key or self.sec_api_key == "your_sec_api_key":
            logger.warning("SEC API key not configured")
            return []
        
        try:
            # SEC EDGAR API
            # Note: This is a simplified version. Real implementation needs CIK lookup
            logger.info(f"Fetching SEC filings for {symbol}")
            # TODO: Implement actual SEC API call
            return []
        except Exception as e:
            logger.error(f"Failed to fetch SEC filings for {symbol}: {e}")
        
        return []
    
    def analyze_with_claude(
        self,
        symbol: str,
        news_articles: List[Dict[str, Any]],
        sec_filings: List[Dict[str, Any]],
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Claude를 사용하여 뉴스 및 공시 분석
        
        Args:
            symbol: 종목 코드
            news_articles: 뉴스 리스트
            sec_filings: 공시 리스트
            market_data: 시장 데이터
            
        Returns:
            알파 신호 딕셔너리
        """
        if not self.anthropic_api_key or self.anthropic_api_key == "your_anthropic_api_key":
            logger.warning("Anthropic API key not configured, returning neutral signals")
            return {
                "risk_scalar": 0.0,
                "sentiment_score": 0.0,
                "earnings_factor": 0.0,
                "uncertainty": 0.0
            }
        
        try:
            # Prepare context
            news_summary = self._summarize_news(news_articles)
            filings_summary = self._summarize_filings(sec_filings)
            market_summary = self._summarize_market_data(market_data) if market_data is not None else ""
            
            # Construct prompt
            prompt = f"""You are a quantitative analyst analyzing {symbol} for alpha generation.

News Summary (last 7 days):
{news_summary}

SEC Filings (last 30 days):
{filings_summary}

Market Data:
{market_summary}

Based on this information, provide the following scores in JSON format:

1. risk_scalar: Risk appetite score (-1 to 1)
   - Positive: Low risk, favorable conditions
   - Negative: High risk, unfavorable conditions

2. sentiment_score: Overall sentiment (-1 to 1)
   - Positive: Bullish sentiment
   - Negative: Bearish sentiment

3. earnings_factor: Earnings/fundamentals score (-1 to 1)
   - Positive: Strong fundamentals
   - Negative: Weak fundamentals

4. uncertainty: Market uncertainty (0 to 1)
   - High: Uncertain, volatile conditions
   - Low: Stable, predictable conditions

Respond with ONLY a JSON object in this format:
{{
  "risk_scalar": <float>,
  "sentiment_score": <float>,
  "earnings_factor": <float>,
  "uncertainty": <float>,
  "reasoning": "<brief explanation>"
}}"""
            
            # Call Claude API
            headers = {
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            payload = {
                "model": "claude-opus-4-20250514",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["content"][0]["text"]
                
                # Parse JSON response
                try:
                    alpha_signals = json.loads(content)
                    logger.info(f"LLM alpha signals for {symbol}: {alpha_signals}")
                    
                    # Validate ranges
                    alpha_signals["risk_scalar"] = np.clip(alpha_signals.get("risk_scalar", 0.0), -1, 1)
                    alpha_signals["sentiment_score"] = np.clip(alpha_signals.get("sentiment_score", 0.0), -1, 1)
                    alpha_signals["earnings_factor"] = np.clip(alpha_signals.get("earnings_factor", 0.0), -1, 1)
                    alpha_signals["uncertainty"] = np.clip(alpha_signals.get("uncertainty", 0.0), 0, 1)
                    
                    return alpha_signals
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse Claude response: {content}")
            else:
                logger.error(f"Claude API error: {response.status_code} - {response.text}")
        
        except Exception as e:
            logger.error(f"Failed to analyze with Claude: {e}")
        
        # Return neutral signals on error
        return {
            "risk_scalar": 0.0,
            "sentiment_score": 0.0,
            "earnings_factor": 0.0,
            "uncertainty": 0.0
        }
    
    def _summarize_news(self, articles: List[Dict[str, Any]]) -> str:
        """뉴스 요약"""
        if not articles:
            return "No recent news available."
        
        summary = []
        for i, article in enumerate(articles[:10], 1):
            title = article.get("title", "")
            description = article.get("description", "")
            published = article.get("publishedAt", "")
            summary.append(f"{i}. [{published}] {title}\n   {description}")
        
        return "\n".join(summary)
    
    def _summarize_filings(self, filings: List[Dict[str, Any]]) -> str:
        """공시 요약"""
        if not filings:
            return "No recent SEC filings available."
        
        summary = []
        for i, filing in enumerate(filings[:5], 1):
            form_type = filing.get("form_type", "")
            filed_date = filing.get("filed_date", "")
            summary.append(f"{i}. [{filed_date}] {form_type}")
        
        return "\n".join(summary)
    
    def _summarize_market_data(self, df: pd.DataFrame) -> str:
        """시장 데이터 요약"""
        if df is None or df.empty:
            return "No market data available."
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        price_change = (latest["close"] - prev["close"]) / prev["close"] * 100
        volume_change = (latest["volume"] - prev["volume"]) / prev["volume"] * 100 if prev["volume"] > 0 else 0
        
        # Calculate volatility
        returns = df["close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        
        summary = f"""Latest Price: ${latest['close']:.2f}
Price Change: {price_change:+.2f}%
Volume Change: {volume_change:+.2f}%
Annualized Volatility: {volatility:.2f}%
20-day High: ${df['high'].tail(20).max():.2f}
20-day Low: ${df['low'].tail(20).min():.2f}"""
        
        return summary
    
    def generate_llm_alpha(
        self,
        symbol: str,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        전체 LLM 알파 파이프라인 실행
        
        Args:
            symbol: 종목 코드
            market_data: 시장 데이터
            
        Returns:
            알파 신호 딕셔너리
        """
        logger.info(f"Generating LLM alpha for {symbol}")
        
        # Fetch data
        news_articles = self.fetch_news(symbol, lookback_days=7)
        sec_filings = self.fetch_sec_filings(symbol, lookback_days=30)
        
        # Analyze with Claude
        alpha_signals = self.analyze_with_claude(
            symbol=symbol,
            news_articles=news_articles,
            sec_filings=sec_filings,
            market_data=market_data
        )
        
        logger.info(f"LLM alpha generated for {symbol}: {alpha_signals}")
        
        return alpha_signals


def test_llm_alpha_pipeline():
    """
    LLM 알파 파이프라인 테스트
    """
    print("=" * 80)
    print("LLM 알파 파이프라인 테스트")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    pipeline = LLMAlphaPipeline()
    
    # Test with sample symbol
    symbol = "AAPL"
    
    print(f"종목: {symbol}")
    print()
    
    # Generate sample market data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="1D")
    np.random.seed(42)
    
    base_price = 150
    returns = np.random.randn(100) * 0.02
    close_prices = base_price * (1 + returns).cumprod()
    
    market_data = pd.DataFrame({
        "date": dates,
        "open": close_prices * (1 + np.random.randn(100) * 0.005),
        "high": close_prices * (1 + np.abs(np.random.randn(100)) * 0.01),
        "low": close_prices * (1 - np.abs(np.random.randn(100)) * 0.01),
        "close": close_prices,
        "volume": np.random.randint(50000000, 100000000, 100)
    })
    
    # Generate LLM alpha
    alpha_signals = pipeline.generate_llm_alpha(symbol, market_data)
    
    print("LLM 알파 신호:")
    print("-" * 80)
    for key, value in alpha_signals.items():
        if key != "reasoning":
            print(f"{key}: {value:.4f}")
    
    if "reasoning" in alpha_signals:
        print()
        print("분석 근거:")
        print(alpha_signals["reasoning"])
    
    print()
    print("=" * 80)
    print("테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    test_llm_alpha_pipeline()
