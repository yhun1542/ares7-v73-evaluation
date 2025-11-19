# File: phoenix_engine.py
# Purpose: Advanced ML-based trading strategy using transformer architecture
# Key features: Transformer model, feature engineering, prediction pipeline, memory optimization

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import asyncio
from collections import deque
import pickle
import json
import gc
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PhoenixConfig:
    """Configuration for Phoenix Engine"""
    model_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    learning_rate: float = 0.0001
    batch_size: int = 32
    sequence_length: int = 100
    prediction_horizon: int = 24
    feature_dim: int = 128
    max_memory_gb: float = 4.0
    confidence_threshold: float = 0.7
    retraining_interval: int = 24  # hours
    anomaly_threshold: float = 0.05
    ensemble_size: int = 3

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class TransformerPredictor(nn.Module):
    """Transformer model for price prediction"""
    
    def __init__(self, config: PhoenixConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.feature_dim, config.model_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.model_dim)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.model_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=config.num_layers
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim // 2, config.model_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim // 4, 3)  # [price_change, volume, volatility]
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(config.model_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
            encoded = encoded.masked_fill(mask_expanded, 0)
            lengths = (~mask).sum(dim=1, keepdim=True).float()
            pooled = encoded.sum(dim=1) / lengths
        else:
            pooled = encoded.mean(dim=1)
        
        # Predictions
        predictions = self.output_projection(pooled)
        confidence = self.confidence_head(pooled)
        
        return predictions, confidence

class FeatureEngineer:
    """Advanced feature engineering for Phoenix Engine"""
    
    def __init__(self, config: PhoenixConfig):
        self.config = config
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=min(50, config.feature_dim // 2))
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.feature_cache = {}
        self.is_fitted = False
        
    def create_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create advanced features from market data"""
        try:
            features = []
            
            # Price features
            features.extend(self._create_price_features(df))
            
            # Volume features
            features.extend(self._create_volume_features(df))
            
            # Technical indicators
            features.extend(self._create_technical_features(df))
            
            # Market microstructure
            features.extend(self._create_microstructure_features(df))
            
            # Sentiment features (if available)
            if 'sentiment' in df.columns:
                features.extend(self._create_sentiment_features(df))
            
            # Time-based features
            features.extend(self._create_time_features(df))
            
            # Combine all features
            feature_matrix = np.column_stack(features)
            
            # Handle NaN values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            if self.is_fitted:
                feature_matrix = self.scaler.transform(feature_matrix)
            else:
                feature_matrix = self.scaler.fit_transform(feature_matrix)
                self.is_fitted = True
            
            # Apply PCA for dimensionality reduction
            if feature_matrix.shape[1] > self.config.feature_dim:
                if self.is_fitted:
                    feature_matrix = self.pca.transform(feature_matrix)
                else:
                    feature_matrix = self.pca.fit_transform(feature_matrix)
            
            # Pad or truncate to match feature_dim
            if feature_matrix.shape[1] < self.config.feature_dim:
                padding = np.zeros((feature_matrix.shape[0], 
                                   self.config.feature_dim - feature_matrix.shape[1]))
                feature_matrix = np.hstack([feature_matrix, padding])
            elif feature_matrix.shape[1] > self.config.feature_dim:
                feature_matrix = feature_matrix[:, :self.config.feature_dim]
            
            return feature_matrix
            
        except Exception as e:
            logger.error(f"Feature creation error: {e}")
            return np.zeros((len(df), self.config.feature_dim))
    
    def _create_price_features(self, df: pd.DataFrame) -> List[np.ndarray]:
        """Create price-based features"""
        features = []
        
        # Returns at different scales
        for period in [1, 5, 10, 20, 50]:
            if len(df) > period:
                returns = df['close'].pct_change(period).fillna(0).values
                features.append(returns)
        
        # Log returns
        log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
        features.append(log_returns)
        
        # Price momentum
        for period in [10, 20, 50]:
            if len(df) > period:
                momentum = (df['close'] - df['close'].shift(period)).fillna(0).values
                features.append(momentum)
        
        # Price acceleration
        returns = df['close'].pct_change().fillna(0)
        acceleration = returns.diff().fillna(0).values
        features.append(acceleration)
        
        # High-low spread
        hl_spread = ((df['high'] - df['low']) / df['close']).fillna(0).values
        features.append(hl_spread)
        
        # Close position in range
        close_position = ((df['close'] - df['low']) / 
                         (df['high'] - df['low'] + 1e-10)).fillna(0.5).values
        features.append(close_position)
        
        return features
    
    def _create_volume_features(self, df: pd.DataFrame) -> List[np.ndarray]:
        """Create volume-based features"""
        features = []
        
        # Volume moving averages
        for period in [5, 10, 20]:
            if len(df) > period:
                vol_ma = df['volume'].rolling(period).mean().fillna(0).values
                features.append(vol_ma)
        
        # Volume rate of change
        vol_roc = df['volume'].pct_change(5).fillna(0).values
        features.append(vol_roc)
        
        # Price-volume correlation
        if len(df) > 20:
            pv_corr = df['close'].rolling(20).corr(df['volume']).fillna(0).values
            features.append(pv_corr)
        
        # Volume weighted average price (VWAP)
        vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        vwap_ratio = (df['close'] / vwap).fillna(1).values
        features.append(vwap_ratio)
        
        # On-balance volume
        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum().fillna(0).values
        features.append(obv / (obv.max() + 1e-10))
        
        return features
    
    def _create_technical_features(self, df: pd.DataFrame) -> List[np.ndarray]:
        """Create technical indicator features"""
        features = []
        
        # RSI
        for period in [14, 28]:
            rsi = self._calculate_rsi(df['close'], period)
            features.append(rsi)
        
        # MACD
        macd, signal, histogram = self._calculate_macd(df['close'])
        features.extend([macd, signal, histogram])
        
        # Bollinger Bands
        for period in [20, 50]:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'], period)
            bb_position = ((df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)).fillna(0.5).values
            features.append(bb_position)
        
        # ATR (Average True Range)
        atr = self._calculate_atr(df, 14)
        features.append(atr)
        
        # Stochastic Oscillator
        k_percent, d_percent = self._calculate_stochastic(df, 14)
        features.extend([k_percent, d_percent])
        
        return features
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> List[np.ndarray]:
        """Create market microstructure features"""
        features = []
        
        # Bid-ask spread proxy
        spread_proxy = 2 * np.sqrt(np.abs(np.log(df['high'] / df['close']) * 
                                         np.log(df['high'] / df['open'])))
        features.append(spread_proxy.fillna(0).values)
        
        # Kyle's lambda (price impact)
        if len(df) > 20:
            price_changes = df['close'].diff().abs()
            volume_sqrt = np.sqrt(df['volume'])
            kyle_lambda = price_changes.rolling(20).mean() / volume_sqrt.rolling(20).mean()
            features.append(kyle_lambda.fillna(0).values)
        
        # Amihud illiquidity
        amihud = np.abs(df['close'].pct_change()) / (df['volume'] + 1e-10)
        features.append(amihud.fillna(0).values)
        
        # Roll's measure
        if len(df) > 2:
            price_changes = df['close'].diff()
            roll_measure = 2 * np.sqrt(-price_changes.cov(price_changes.shift(1)))
            roll_array = np.full(len(df), roll_measure if not np.isnan(roll_measure) else 0)
            features.append(roll_array)
        
        return features
    
    def _create_sentiment_features(self, df: pd.DataFrame) -> List[np.ndarray]:
        """Create sentiment-based features"""
        features = []
        
        # Raw sentiment
        features.append(df['sentiment'].fillna(0).values)
        
        # Sentiment moving averages
        for period in [5, 10, 20]:
            if len(df) > period:
                sent_ma = df['sentiment'].rolling(period).mean().fillna(0).values
                features.append(sent_ma)
        
        # Sentiment momentum
        sent_momentum = df['sentiment'].diff(5).fillna(0).values
        features.append(sent_momentum)
        
        return features
    
    def _create_time_features(self, df: pd.DataFrame) -> List[np.ndarray]:
        """Create time-based features"""
        features = []
        
        if 'timestamp' in df.columns:
            # Hour of day (cyclical encoding)
            hour = pd.to_datetime(df['timestamp']).dt.hour
            hour_sin = np.sin(2 * np.pi * hour / 24).values
            hour_cos = np.cos(2 * np.pi * hour / 24).values
            features.extend([hour_sin, hour_cos])
            
            # Day of week (cyclical encoding)
            dow = pd.to_datetime(df['timestamp']).dt.dayofweek
            dow_sin = np.sin(2 * np.pi * dow / 7).values
            dow_cos = np.cos(2 * np.pi * dow / 7).values
            features.extend([dow_sin, dow_cos])
            
            # Month (cyclical encoding)
            month = pd.to_datetime(df['timestamp']).dt.month
            month_sin = np.sin(2 * np.pi * month / 12).values
            month_cos = np.cos(2 * np.pi * month / 12).values
            features.extend([month_sin, month_cos])
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values / 100
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        # Normalize
        macd_norm = macd / (prices.rolling(50).std() + 1e-10)
        signal_norm = signal / (prices.rolling(50).std() + 1e-10)
        histogram_norm = histogram / (prices.rolling(50).std() + 1e-10)
        
        return (macd_norm.fillna(0).values, 
                signal_norm.fillna(0).values, 
                histogram_norm.fillna(0).values)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * 2)
        lower = middle - (std * 2)
        return upper, middle, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        # Normalize by price
        atr_norm = atr / df['close']
        return atr_norm.fillna(0).values
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-10))
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent.fillna(50).values / 100, d_percent.fillna(50).values / 100

class PhoenixEngine:
    """Advanced ML-based trading engine with transformer architecture"""
    
    def __init__(self, config: Optional[PhoenixConfig] = None):
        self.config = config or PhoenixConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(self.config)
        self.models = self._initialize_ensemble()
        self.optimizers = self._initialize_optimizers()
        
        # Training state
        self.training_history = deque(maxlen=10000)
        self.prediction_cache = {}
        self.last_training = datetime.now()
        self.training_lock = asyncio.Lock()
        
        # Performance tracking
        self.performance_metrics = {
            'predictions': deque(maxlen=1000),
            'accuracy': deque(maxlen=100),
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        
        # Memory management
        self.memory_monitor = MemoryMonitor(self.config.max_memory_gb)
        
        logger.info(f"Phoenix Engine initialized on {self.device}")
    
    def _initialize_ensemble(self) -> List[TransformerPredictor]:
        """Initialize ensemble of models"""
        models = []
        for i in range(self.config.ensemble_size):
            model = TransformerPredictor(self.config).to(self.device)
            models.append(model)
        return models
    
    def _initialize_optimizers(self) -> List[optim.Optimizer]:
        """Initialize optimizers for ensemble"""
        optimizers = []
        for model in self.models:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01
            )
            optimizers.append(optimizer)
        return optimizers
    
    async def predict(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions from market data"""
        try:
            # Check memory
            self.memory_monitor.check_memory()
            
            # Feature engineering
            features = self.feature_engineer.create_features(market_data)
            
            # Prepare sequences
            sequences = self._prepare_sequences(features)
            if sequences is None:
                return self._empty_prediction()
            
            # Convert to tensor
            x = torch.FloatTensor(sequences).to(self.device)
            
            # Ensemble predictions
            all_predictions = []
            all_confidences = []
            
            with torch.no_grad():
                for model in self.models:
                    model.eval()
                    predictions, confidence = model(x)
                    all_predictions.append(predictions.cpu().numpy())
                    all_confidences.append(confidence.cpu().numpy())
            
            # Aggregate ensemble predictions
            predictions = np.mean(all_predictions, axis=0)
            confidence = np.mean(all_confidences, axis=0)
            
            # Interpret predictions
            result = self._interpret_predictions(predictions, confidence, market_data)
            
            # Cache prediction
            self.prediction_cache[datetime.now()] = result
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._empty_prediction()
    
    def _prepare_sequences(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Prepare sequences for transformer input"""
        if len(features) < self.config.sequence_length:
            return None
        
        # Take the most recent sequence
        sequence = features[-self.config.sequence_length:]
        
        # Add batch dimension
        sequence = np.expand_dims(sequence, axis=0)
        
        return sequence
    
    def _interpret_predictions(self, predictions: np.ndarray, 
                              confidence: np.ndarray, 
                              market_data: pd.DataFrame) -> Dict[str, Any]:
        """Interpret model predictions"""
        
        # Extract prediction components
        price_change = float(predictions[0, 0])
        volume_prediction = float(predictions[0, 1])
        volatility_prediction = float(predictions[0, 2])
        confidence_score = float(confidence[0, 0])
        
        # Current price
        current_price = float(market_data['close'].iloc[-1])
        
        # Calculate target price
        target_price = current_price * (1 + price_change)
        
        # Determine signal
        if confidence_score < self.config.confidence_threshold:
            signal = 'HOLD'
            position_size = 0.0
        elif price_change > 0.02:  # 2% threshold
            signal = 'BUY'
            position_size = min(confidence_score, 1.0)
        elif price_change < -0.02:
            signal = 'SELL'
            position_size = min(confidence_score, 1.0)
        else:
            signal = 'HOLD'
            position_size = 0.0
        
        # Risk metrics
        expected_return = price_change
        expected_volatility = abs(volatility_prediction)
        risk_reward_ratio = abs(expected_return) / (expected_volatility + 1e-10)
        
        return {
            'timestamp': datetime.now(),
            'signal': signal,
            'confidence': confidence_score,
            'position_size': position_size,
            'current_price': current_price,
            'target_price': target_price,
            'expected_return': expected_return,
            'expected_volatility': expected_volatility,
            'risk_reward_ratio': risk_reward_ratio,
            'volume_prediction': volume_prediction,
            'model_version': 'phoenix_v2.0',
            'features_used': self.config.feature_dim
        }
    
    async def train(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """Train the Phoenix Engine models"""
        async with self.training_lock:
            try:
                logger.info("Starting Phoenix Engine training...")
                
                # Prepare training data
                features = self.feature_engineer.create_features(training_data)
                x_train, y_train = self._prepare_training_data(features, training_data)
                
                if x_train is None:
                    return {'status': 'failed', 'reason': 'insufficient_data'}
                
                # Create data loader
                dataset = TensorDataset(
                    torch.FloatTensor(x_train),
                    torch.FloatTensor(y_train)
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True
                )
                
                # Training metrics
                ensemble_losses = []
                
                # Train each model in ensemble
                for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                    model.train()
                    epoch_losses = []
                    
                    for epoch in range(10):  # Quick training
                        batch_losses = []
                        
                        for batch_x, batch_y in dataloader:
                            batch_x = batch_x.to(self.device)
                            batch_y = batch_y.to(self.device)
                            
                            # Forward pass
                            predictions, confidence = model(batch_x)
                            
                            # Calculate loss
                            prediction_loss = nn.MSELoss()(predictions, batch_y)
                            confidence_loss = self._calculate_confidence_loss(
                                predictions, batch_y, confidence
                            )
                            total_loss = prediction_loss + 0.1 * confidence_loss
                            
                            # Backward pass
                            optimizer.zero_grad()
                            total_loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            
                            batch_losses.append(total_loss.item())
                        
                        epoch_loss = np.mean(batch_losses)
                        epoch_losses.append(epoch_loss)
                    
                    ensemble_losses.append(np.mean(epoch_losses))
                    logger.info(f"Model {model_idx} trained, loss: {ensemble_losses[-1]:.4f}")
                
                # Update training state
                self.last_training = datetime.now()
                
                # Clear memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return {
                    'status': 'success',
                    'ensemble_loss': float(np.mean(ensemble_losses)),
                    'training_samples': len(x_train),
                    'timestamp': self.last_training.isoformat()
                }
                
            except Exception as e:
                logger.error(f"Training error: {e}")
                return {'status': 'failed', 'error': str(e)}
    
    def _prepare_training_data(self, features: np.ndarray, 
                               market_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare sequences and targets for training"""
        if len(features) < self.config.sequence_length + self.config.prediction_horizon:
            return None, None
        
        x_sequences = []
        y_targets = []
        
        for i in range(len(features) - self.config.sequence_length - self.config.prediction_horizon):
            # Input sequence
            x_seq = features[i:i + self.config.sequence_length]
            x_sequences.append(x_seq)
            
            # Target values
            future_idx = i + self.config.sequence_length + self.config.prediction_horizon
            current_price = market_data['close'].iloc[i + self.config.sequence_length]
            future_price = market_data['close'].iloc[future_idx]
            
            price_change = (future_price - current_price) / current_price
            
            # Volume change
            current_volume = market_data['volume'].iloc[i + self.config.sequence_length]
            future_volume = market_data['volume'].iloc[future_idx]
            volume_change = (future_volume - current_volume) / (current_volume + 1e-10)
            
            # Volatility (using high-low range as proxy)
            future_volatility = (market_data['high'].iloc[future_idx] - 
                               market_data['low'].iloc[future_idx]) / current_price
            
            y_targets.append([price_change, volume_change, future_volatility])
        
        return np.array(x_sequences), np.array(y_targets)
    
    def _calculate_confidence_loss(self, predictions: torch.Tensor, 
                                  targets: torch.Tensor, 
                                  confidence: torch.Tensor) -> torch.Tensor:
        """Calculate confidence calibration loss"""
        # Prediction error
        error = torch.abs(predictions - targets).mean(dim=1, keepdim=True)
        
        # Confidence should be high when error is low
        target_confidence = torch.exp(-error * 10)  # Exponential decay
        
        # MSE loss for confidence
        confidence_loss = nn.MSELoss()(confidence, target_confidence)
        
        return confidence_loss
    
    def _update_performance_metrics(self, prediction: Dict[str, Any]):
        """Update performance tracking metrics"""
        self.performance_metrics['predictions'].append(prediction)
        
        # Calculate rolling accuracy
        if len(self.performance_metrics['predictions']) > 10:
            recent_predictions = list(self.performance_metrics['predictions'])[-10:]
            correct = sum(1 for p in recent_predictions 
                         if p.get('actual_return', 0) * p['expected_return'] > 0)
            accuracy = correct / len(recent_predictions)
            self.performance_metrics['accuracy'].append(accuracy)
    
    def _empty_prediction(self) -> Dict[str, Any]:
        """Return empty prediction when unable to generate signal"""
        return {
            'timestamp': datetime.now(),
            'signal': 'HOLD',
            'confidence': 0.0,
            'position_size': 0.0,
            'current_price': 0.0,
            'target_price': 0.0,
            'expected_return': 0.0,
            'expected_volatility': 0.0,
            'risk_reward_ratio': 0.0,
            'volume_prediction': 0.0,
            'model_version': 'phoenix_v2.0',
            'features_used': 0
        }
    
    async def backtest(self, historical_data: pd.DataFrame, 
                      initial_capital: float = 100000) -> Dict[str, Any]:
        """Backtest Phoenix Engine on historical data"""
        try:
            logger.info("Starting Phoenix Engine backtest...")
            
            # Initialize backtest state
            capital = initial_capital
            position = 0
            trades = []
            equity_curve = [initial_capital]
            
            # Split data for training and testing
            train_size = int(len(historical_data) * 0.7)
            train_data = historical_data[:train_size]
            test_data = historical_data[train_size:]
            
            # Initial training
            await self.train(train_data)
            
            # Run backtest
            for i in range(self.config.sequence_length, len(test_data)):
                # Get current window
                window = test_data.iloc[:i+1]
                
                # Generate prediction
                prediction = await self.predict(window)
                
                current_price = float(window['close'].iloc[-1])
                
                # Execute trades based on signal
                if prediction['signal'] == 'BUY' and position <= 0:
                    # Close short and go long
                    if position < 0:
                        capital += position * current_price  # Close short
                        trades.append({
                            'type': 'close_short',
                            'price': current_price,
                            'size': abs(position),
                            'capital': capital
                        })
                    
                    # Open long position
                    position_value = capital * prediction['position_size'] * 0.95  # 95% to leave some cash
                    position = position_value / current_price
                    
                    trades.append({
                        'type': 'buy',
                        'price': current_price,
                        'size': position,
                        'confidence': prediction['confidence'],
                        'capital': capital
                    })
                    
                elif prediction['signal'] == 'SELL' and position >= 0:
                    # Close long and go short
                    if position > 0:
                        capital = position * current_price  # Close long
                        trades.append({
                            'type': 'close_long',
                            'price': current_price,
                            'size': position,
                            'capital': capital
                        })
                        position = 0
                    
                    # Open short position (simplified)
                    position_value = capital * prediction['position_size'] * 0.95
                    position = -position_value / current_price
                    
                    trades.append({
                        'type': 'sell',
                        'price': current_price,
                        'size': abs(position),
                        'confidence': prediction['confidence'],
                        'capital': capital
                    })
                
                # Update equity
                if position > 0:
                    current_equity = position * current_price
                elif position < 0:
                    current_equity = capital + position * current_price
                else:
                    current_equity = capital
                
                equity_curve.append(current_equity)
                
                # Periodic retraining
                if i % 100 == 0 and i > train_size + 100:
                    retrain_data = test_data.iloc[:i]
                    await self.train(retrain_data)
            
            # Calculate metrics
            equity_curve = np.array(equity_curve)
            returns = np.diff(equity_curve) / equity_curve[:-1]
            
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            
            # Maximum drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Win rate
            winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            win_rate = winning_trades / len(trades) if trades else 0
            
            return {
                'initial_capital': initial_capital,
                'final_capital': float(equity_curve[-1]),
                'total_return': float((equity_curve[-1] - initial_capital) / initial_capital),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'total_trades': len(trades),
                'equity_curve': equity_curve.tolist()[-100:],  # Last 100 points
                'trades': trades[-20:]  # Last 20 trades
            }
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return {
                'error': str(e),
                'initial_capital': initial_capital,
                'final_capital': initial_capital,
                'total_return': 0.0
            }
    
    def save_model(self, filepath: str):
        """Save model state"""
        try:
            state = {
                'config': self.config,
                'models': [model.state_dict() for model in self.models],
                'feature_scaler': self.feature_engineer.scaler,
                'performance_metrics': dict(self.performance_metrics),
                'training_history': list(self.training_history)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load model state"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.config = state['config']
            
            # Reinitialize models with loaded weights
            self.models = self._initialize_ensemble()
            for model, state_dict in zip(self.models, state['models']):
                model.load_state_dict(state_dict)
            
            self.feature_engineer.scaler = state['feature_scaler']
            self.performance_metrics.update(state['performance_metrics'])
            self.training_history = deque(state['training_history'], 
                                        maxlen=10000)
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")

class MemoryMonitor:
    """Monitor and manage memory usage"""
    
    def __init__(self, max_memory_gb: float):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
    
    def check_memory(self):
        """Check current memory usage and clean if necessary"""
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss
            
            if memory_usage > self.max_memory_bytes * 0.9:  # 90% threshold
                logger.warning(f"High memory usage: {memory_usage / 1024 / 1024 / 1024:.2f} GB")
                self.cleanup_memory()
            
        except ImportError:
            pass  # psutil not available
    
    def cleanup_memory(self):
        """Force garbage collection and clear caches"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Memory cleanup completed")