"""
ARES-7 v73 LLM Feature Builder
Unified LLM alpha feature builder for MetaEngine
"""

import numpy as np
from typing import Dict, Any, Optional
from .llm_regime_classifier import LLMRegimeClassifier
from .llm_news_sentiment import LLMNewsSentiment
from .llm_earnings_analysis import LLMEarningsAnalyzer
from .llm_embeddings import LLMEmbeddingEngine


class LLMFeatureBuilder:
    """
    Combines:
    - Macro regime classification
    - News sentiment analysis
    - Earnings analysis
    - Text embeddings
    
    into a unified alpha feature dict for MetaEngine
    """
    
    def __init__(self, provider="openai"):
        self.regime = LLMRegimeClassifier(provider)
        self.news = LLMNewsSentiment(provider)
        self.earn = LLMEarningsAnalyzer(provider)
        self.embed = LLMEmbeddingEngine(provider=provider)
    
    def build(
        self,
        *,
        market_text: Optional[str] = None,
        news_headline: Optional[str] = None,
        news_body: Optional[str] = None,
        earnings_text: Optional[str] = None,
        whisper_number: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Build unified LLM alpha features
        
        Args:
            market_text: Macro market commentary
            news_headline: News headline
            news_body: News body
            earnings_text: Earnings press release or transcript
            whisper_number: Whisper number for earnings
        
        Returns:
            Dictionary of LLM-derived alpha features
        """
        out = {}
        
        # Macro regime
        if market_text:
            reg = self.regime.classify(market_text)
            out.update({
                "llm_regime": reg.get("regime"),
                "risk_scalar": reg.get("risk_score", 0),
                "macro_vol_bias": reg.get("vol_bias", 0),
                "liquidity_tone": reg.get("liquidity", 0),
                "uncertainty": reg.get("uncertainty", 0),
                "regime_conviction": reg.get("conviction", 0),
                "regime_embedding": reg.get("embedding", [])
            })
        
        # News sentiment
        if news_headline:
            ns = self.news.analyze(news_headline, news_body or "")
            out.update({
                "sentiment_score": ns.get("sentiment", 0),
                "impact_score": ns.get("impact", 0),
                "news_vol_shock": ns.get("vol_shock", 0),
                "narrative": ns.get("summary", ""),
                "news_embedding": ns.get("embedding", [])
            })
        
        # Earnings analysis
        if earnings_text:
            ea = self.earn.analyze(earnings_text, whisper_number)
            out.update({
                "earnings_score": ea.get("earnings_score", 0),
                "tone_score": ea.get("tone_score", 0),
                "guidance_score": ea.get("guidance_score", 0),
                "surprise_prob": ea.get("surprise_prob", 0),
                "earnings_vol_shock": ea.get("vol_shock", 0),
                "drift_direction": ea.get("drift_direction", "flat"),
                "earnings_embedding": ea.get("embedding", [])
            })
        
        # Global embedding (all-in-one)
        embed_text = f"{market_text or ''} {news_headline or ''} {news_body or ''} {earnings_text or ''}"
        embed_text = embed_text.strip()
        
        if embed_text:
            out["global_embedding"] = self.embed.embed_text(embed_text)
        else:
            out["global_embedding"] = []
        
        return out
