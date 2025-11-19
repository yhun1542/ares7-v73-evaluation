"""
ARES-7 v73 LLM News Sentiment
Financial news sentiment analysis using LLM
"""

import json
import logging
from typing import Dict, Any
from .llm_client import LLMClient
from .llm_embeddings import LLMEmbeddingEngine


class LLMNewsSentiment:
    """
    Financial news sentiment engine
    """
    
    def __init__(self, provider="openai"):
        self.client = LLMClient(provider=provider)
        self.embed = LLMEmbeddingEngine(provider=provider)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze(self, headline: str, body: str = "") -> Dict[str, Any]:
        """
        Analyze financial news sentiment
        
        Args:
            headline: News headline
            body: News body (optional)
        
        Returns:
            {
                "sentiment": float (-1 to 1),
                "impact": float (0 to 1),
                "vol_shock": float (0 to 1),
                "direction": str ("bullish", "bearish", "neutral"),
                "summary": str,
                "embedding": List[float]
            }
        """
        prompt = f"""
Analyze financial news and produce STRICT JSON.

HEADLINE: {headline}
BODY: {body}

FORMAT:
{{
  "sentiment": <float -1~1>,
  "impact": <float 0~1>,
  "vol_shock": <float 0~1>,
  "direction": "bullish|bearish|neutral",
  "summary": "..."
}}
"""
        
        res = self.client.completion(prompt, temperature=0)
        parsed = self._parse_json(res)
        
        # Add embedding
        parsed["embedding"] = self.embed.embed_text(headline + " " + body)
        
        return parsed
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {
                "sentiment": 0,
                "impact": 0,
                "vol_shock": 0,
                "direction": "neutral",
                "summary": ""
            }
