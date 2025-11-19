"""
ARES-7 v73 LLM Earnings Analysis
Earnings call and press release analysis using LLM
"""

import json
import logging
from typing import Dict, Any
from .llm_client import LLMClient
from .llm_embeddings import LLMEmbeddingEngine


class LLMEarningsAnalyzer:
    """
    LLM-based analysis of earnings press release or call transcript
    """
    
    def __init__(self, provider="openai"):
        self.client = LLMClient(provider=provider)
        self.embed = LLMEmbeddingEngine(provider=provider)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze(self, text: str, whisper_number: float = None) -> Dict[str, Any]:
        """
        Analyze earnings text
        
        Args:
            text: Earnings press release or call transcript
            whisper_number: Whisper number (optional)
        
        Returns:
            {
                "earnings_score": float (-1 to 1),
                "tone_score": float (-1 to 1),
                "guidance_score": float (0 to 1),
                "surprise_prob": float (0 to 1),
                "drift_direction": str ("up", "down", "flat"),
                "vol_shock": float (0 to 1),
                "summary": str,
                "embedding": List[float]
            }
        """
        prompt = f"""
Analyze the following earnings-related text and return STRICT JSON.

TEXT:
{text}

FORMAT:
{{
  "earnings_score": <float -1~1>,
  "tone_score": <float -1~1>,
  "guidance_score": <float 0~1>,
  "surprise_prob": <float 0~1>,
  "drift_direction": "up|down|flat",
  "vol_shock": <float 0~1>,
  "summary": "..."
}}
"""
        
        if whisper_number is not None:
            prompt += f"\nWhisper Number: {whisper_number}\n"
        
        res = self.client.completion(prompt, temperature=0)
        parsed = self._parse_json(res)
        
        # Add embedding
        parsed["embedding"] = self.embed.embed_text(text)
        
        return parsed
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {
                "earnings_score": 0,
                "tone_score": 0,
                "guidance_score": 0,
                "surprise_prob": 0,
                "drift_direction": "flat",
                "vol_shock": 0,
                "summary": ""
            }
