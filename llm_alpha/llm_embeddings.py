"""
ARES-7 v73 LLM Embeddings
Text embedding engine with caching and chunking
"""

import os
import re
import json
import hashlib
import logging
from typing import List, Dict
from .llm_client import LLMClient


class LLMEmbeddingEngine:
    """
    Embedding engine with:
    - Text cleaning
    - Chunking for long texts
    - Vector normalization
    - Disk-based caching
    """
    
    def __init__(
        self,
        cache_path: str = "/tmp/llm_embedding_cache.json",
        max_chunk_words: int = 160,
        provider: str = "openai"
    ):
        self.cache_path = cache_path
        self.max_chunk_words = max_chunk_words
        self.client = LLMClient(provider=provider)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    self.cache = json.load(f)
            except:
                self.cache = {}
        else:
            self.cache = {}
    
    # ========== Text Cleaning ==========
    
    def _clean(self, text: str) -> str:
        """Clean text for embedding"""
        # Remove zero-width spaces
        text = text.replace("\u200b", "")
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # Remove non-ASCII characters
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        
        return text
    
    # ========== Chunking ==========
    
    def _chunk(self, text: str) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        
        if len(words) <= self.max_chunk_words:
            return [text]
        
        chunks = []
        for i in range(0, len(words), self.max_chunk_words):
            chunks.append(" ".join(words[i : i + self.max_chunk_words]))
        
        return chunks
    
    # ========== Hashing ==========
    
    def _hash(self, text: str) -> str:
        """Generate hash for caching"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    # ========== Normalization ==========
    
    def _normalize(self, vec: List[float]) -> List[float]:
        """Normalize vector to unit length"""
        import math
        
        norm = math.sqrt(sum(v * v for v in vec)) + 1e-9
        return [v / norm for v in vec]
    
    # ========== Main Embedding ==========
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed text with caching and chunking
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector (normalized, averaged over chunks)
        """
        # Clean text
        text = self._clean(text)
        
        # Split into chunks
        chunks = self._chunk(text)
        
        vectors = []
        
        for c in chunks:
            # Check cache
            key = self._hash(c)
            
            if key in self.cache:
                vectors.append(self.cache[key])
                continue
            
            # Generate embedding
            vec = self.client.embed(c)
            
            if vec:
                # Normalize and cache
                nvec = self._normalize(vec)
                vectors.append(nvec)
                self.cache[key] = nvec
                
                # Save cache
                try:
                    with open(self.cache_path, "w") as f:
                        json.dump(self.cache, f)
                except:
                    pass
        
        if not vectors:
            return []
        
        # Average vectors
        dim = len(vectors[0])
        out = [0.0] * dim
        
        for v in vectors:
            for i in range(dim):
                out[i] += v[i]
        
        return [x / len(vectors) for x in out]
