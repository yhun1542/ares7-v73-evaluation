"""
ARES-7 v73 LLM Client
Unified LLM API client for OpenAI and Google Gemini
"""

import os
import json
import logging
import requests
from typing import List, Dict, Optional, Any


class LLMClient:
    """
    Unified LLM Client for ARES-7 v73
    
    Supports:
    - OpenAI API (GPT-4 series)
    - Google Gemini API
    """
    
    def __init__(
        self,
        provider: str = "openai",
        system_prompt: str = None,
        openai_model: str = "gpt-4o-mini",
        gemini_model: str = "gemini-pro"
    ):
        self.provider = provider.lower()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # API Keys from environment
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.gemini_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
        
        self.openai_model = openai_model
        self.gemini_model = gemini_model
        
        self.system_prompt = (
            system_prompt
            or "You are an advanced financial analysis model. Produce factual, structured outputs."
        )
    
    # ========== Completion (text generation) ==========
    
    def completion(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Generate text completion
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0-1)
        
        Returns:
            Generated text
        """
        if self.provider == "openai":
            return self._call_openai(prompt, temperature)
        elif self.provider == "gemini":
            return self._call_gemini(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    # ========== OpenAI ==========
    
    def _call_openai(self, prompt: str, temperature: float) -> str:
        """Call OpenAI API"""
        try:
            import openai
            openai.api_key = self.openai_key
            
            res = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            
            return res["choices"][0]["message"]["content"]
        
        except Exception as e:
            self.logger.error(f"OpenAI API Error: {e}")
            return ""
    
    # ========== Gemini ==========
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API"""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent?key={self.gemini_key}"
        )
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        try:
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()
            
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        
        except Exception as e:
            self.logger.error(f"Gemini API Error: {e}")
            return ""
    
    # ========== Embeddings (OpenAI only) ==========
    
    def embed(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        """
        Generate text embedding
        
        Args:
            text: Input text
            model: Embedding model
        
        Returns:
            Embedding vector
        """
        try:
            import openai
            openai.api_key = self.openai_key
            
            res = openai.Embedding.create(
                model=model,
                input=text
            )
            
            return res["data"][0]["embedding"]
        
        except Exception as e:
            self.logger.error(f"Embedding error: {e}")
            return []
