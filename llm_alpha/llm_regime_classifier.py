"""
ARES-7 v73 LLM Regime Classifier
Macro regime classification using LLM
"""

import json
import logging
from typing import Dict, Any
from .llm_client import LLMClient
from .llm_embeddings import LLMEmbeddingEngine


class LLMRegimeClassifier:
    """
    Classifies macro regime:
    - Tightening / Easing
    - Growth / Slowdown
    - Volatility expansion / crush
    - Risk-on / Risk-off
    """
    
    def __init__(self, provider: str = "openai"):
        self.client = LLMClient(provider=provider)
        self.embed = LLMEmbeddingEngine(provider=provider)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify macro regime from text
        
        Args:
            text: Market commentary or macro narrative
        
        Returns:
            {
                "regime": str,
                "risk_score": float (-1 to 1),
                "vol_bias": float (-1 to 1),
                "liquidity": float (-1 to 1),
                "uncertainty": float (0 to 1),
                "conviction": float (0 to 1),
                "embedding": List[float]
            }
        """
        cleaned = text.replace("\n", " ")
        
        prompt = f"""
Analyze the macro narrative below and output strictly JSON:

TEXT:
{cleaned}

FORMAT:
{{
  "regime": "...",
  "risk_score": <float -1~1>,
  "vol_bias": <float -1~1>,
  "liquidity": <float -1~1>,
  "uncertainty": <float 0~1>,
  "conviction": <float 0~1>
}}
"""
        
        res = self.client.completion(prompt, temperature=0)
        parsed = self._parse_json(res)
        
        # Add embedding
        parsed["embedding"] = self.embed.embed_text(cleaned)
        
        return parsed
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {
                "regime": "UNKNOWN",
                "risk_score": 0,
                "vol_bias": 0,
                "liquidity": 0,
                "uncertainty": 0,
                "conviction": 0
            }
