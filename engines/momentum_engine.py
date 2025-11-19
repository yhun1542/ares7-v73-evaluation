"""
Improved Momentum Engine with GPU/CPU Auto-detection and Fallback
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DeviceManager:
    """Manages device selection with automatic GPU/CPU detection"""
    
    def __init__(self):
        self.device = self._detect_device()
        self.use_flash_attention = self._check_flash_attention()
        logger.info(f"Device: {self.device}, FlashAttention: {self.use_flash_attention}")
    
    def _detect_device(self) -> torch.device:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')  # Apple Silicon
            logger.info("✅ Apple MPS detected")
            return device
        else:
            device = torch.device('cpu')
            logger.info("⚠️  No GPU detected, using CPU")
            return device
    
    def _check_flash_attention(self) -> bool:
        """Check if FlashAttention is available"""
        try:
            import flash_attn
            if self.device.type == 'cuda':
                logger.info("✅ FlashAttention available on GPU")
                return True
            else:
                logger.info("⚠️  FlashAttention requires CUDA GPU")
                return False
        except ImportError:
            logger.info("⚠️  FlashAttention not installed, using standard attention")
            return False

class FlashAttentionCPU(nn.Module):
    """CPU-optimized attention fallback"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention (CPU-optimized)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class FlashAttentionGPU(nn.Module):
    """GPU-accelerated FlashAttention"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        try:
            from flash_attn import flash_attn_qkvpacked_func
            self.flash_attn_func = flash_attn_qkvpacked_func
        except ImportError:
            raise ImportError("flash_attn not available")
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        
        # Use FlashAttention
        x = self.flash_attn_func(qkv)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        
        return x

class AdaptiveAttention(nn.Module):
    """Adaptive attention that automatically selects GPU or CPU implementation"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.device_manager = DeviceManager()
        
        # Select appropriate attention implementation
        if self.device_manager.use_flash_attention:
            logger.info("Using FlashAttention (GPU)")
            self.attention = FlashAttentionGPU(dim, num_heads)
        else:
            logger.info("Using CPU-optimized attention")
            self.attention = FlashAttentionCPU(dim, num_heads)
        
        self.attention = self.attention.to(self.device_manager.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)

class TransformerMomentumNet(nn.Module):
    """Transformer-based momentum prediction with adaptive attention"""
    
    def __init__(self, 
                 input_dim: int = 10,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.device_manager = DeviceManager()
        self.device = self.device_manager.device
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers with adaptive attention
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Momentum score between -1 and 1
        )
        
        self.to(self.device)
        logger.info(f"TransformerMomentumNet initialized on {self.device}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            momentum_score: (batch_size, 1)
        """
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Project input
        x = self.input_proj(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output
        momentum_score = self.output_head(x)
        
        return momentum_score

class MomentumEngine:
    """Momentum calculation engine with adaptive GPU/CPU support"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device_manager = DeviceManager()
        self.model = TransformerMomentumNet(
            input_dim=self.config.get('input_dim', 10),
            hidden_dim=self.config.get('hidden_dim', 256),
            num_heads=self.config.get('num_heads', 8),
            num_layers=self.config.get('num_layers', 4)
        )
        
        logger.info("MomentumEngine initialized with adaptive device selection")
    
    def calculate_momentum(self, data: np.ndarray) -> float:
        """
        Calculate momentum score from price data
        
        Args:
            data: numpy array of shape (seq_len, features)
        Returns:
            momentum_score: float between -1 and 1
        """
        # Convert to tensor
        x = torch.from_numpy(data).float().unsqueeze(0)  # (1, seq_len, features)
        
        # Forward pass
        with torch.no_grad():
            score = self.model(x)
        
        return score.item()
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get current device information"""
        return {
            'device': str(self.device_manager.device),
            'device_type': self.device_manager.device.type,
            'flash_attention': self.device_manager.use_flash_attention,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize engine (auto-detects device)
    engine = MomentumEngine()
    
    # Print device info
    info = engine.get_device_info()
    print("\n📊 Device Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with dummy data
    dummy_data = np.random.randn(30, 10)  # 30 timesteps, 10 features
    momentum = engine.calculate_momentum(dummy_data)
    print(f"\n📈 Momentum Score: {momentum:.4f}")
