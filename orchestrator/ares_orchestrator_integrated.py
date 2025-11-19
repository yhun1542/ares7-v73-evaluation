"""
ARES-7 v73 FULL Orchestrator
백테스트 + 실거래 모드 통합
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

import numpy as np
import pandas as pd

# v73 Engine Imports
from engines.phoenix.phoenix_engine import PhoenixEngineV73, PhoenixConfig
from engines.momentum.momentum_engine import MomentumEngineV73, MomentumConfig
from engines.mean_reversion.meanrev_engine import MeanReversionEngineV73, MeanRevConfig
from meta.ensemble_meta_engine import EnsembleMetaEngineV73, MetaConfig
from risk.risk_manager import RiskManagerV73
from engines.execution.execution_engine import FullExecutionPipelineV73, ExecutionConfigV73
from data.pipelines.alpha_pipeline import AlphaPipelineV73, AlphaPipelineConfig
from monitoring.monitoring_engine import MonitoringEngineV73

# v64 Broker Layer Imports (실거래 모드용)
from brokers.unified_broker import UnifiedBrokerV2
from governance.order_generator import OrderGenerator
from governance.kill_switch import KillSwitch, get_kill_switch

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """거래 모드"""
    BACKTEST = "backtest"      # 백테스트 (가상 실행)
    PAPER = "paper"            # 페이퍼 트레이딩 (모의투자)
    LIVE = "live"              # 실거래


class OrchestratorConfig:
    """
    Main system configuration object.
    """
    
    def __init__(
        self,
        # Trading mode
        mode: TradingMode = TradingMode.BACKTEST,
        
        # v73 Engine configs
        phoenix_cfg: PhoenixConfig = None,
        momentum_cfg: MomentumConfig = None,
        meanrev_cfg: MeanRevConfig = None,
        meta_cfg: MetaConfig = None,
        exec_cfg: ExecutionConfigV73 = None,
        alpha_cfg: AlphaPipelineConfig = None,
        
        # Capital
        capital: float = 1_000_000,
        
        # Broker config (실거래 모드용)
        broker_config: Optional[Dict[str, Any]] = None
    ):
        self.mode = mode
        self.phoenix_cfg = phoenix_cfg or PhoenixConfig()
        self.momentum_cfg = momentum_cfg or MomentumConfig()
        self.meanrev_cfg = meanrev_cfg or MeanRevConfig()
        self.meta_cfg = meta_cfg or MetaConfig()
        self.exec_cfg = exec_cfg or ExecutionConfigV73()
        self.alpha_cfg = alpha_cfg or AlphaPipelineConfig()
        self.capital = capital
        self.broker_config = broker_config


class AresOrchestratorV73Full:
    """
    ARES-7 v73 통합 Orchestrator
    
    백테스트 모드:
        - 기존 v73 ExecutionEngine 사용 (가상 실행)
        - 브로커 연결 없음
    
    실거래 모드 (Paper/Live):
        - UnifiedBroker 연결 (KIS/IBKR)
        - OrderGenerator로 시그널 → 주문 변환
        - 실제 브로커 API 호출
        - KillSwitch 활성화
    """
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.mode = config.mode
        
        # v73 Engine instances (모든 모드 공통)
        self.phoenix = PhoenixEngineV73(config.phoenix_cfg)
        self.momentum = MomentumEngineV73(config.momentum_cfg)
        self.meanrev = MeanReversionEngineV73(config.meanrev_cfg)
        self.meta = EnsembleMetaEngineV73(config.meta_cfg)
        self.risk = RiskManagerV73()
        self.alpha = AlphaPipelineV73(config.alpha_cfg)
        self.monitor = MonitoringEngineV73()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 포지션 및 자본
        self.positions: Dict[str, float] = {}   # symbol → qty
        self.equity = config.capital
        
        # 실거래 모드 전용 컴포넌트
        self.broker = None
        self.order_gen = None
        self.kill_switch = None
        
        # 백테스트 모드 전용 컴포넌트
        self.exec_engine = None
        
        # 모드별 초기화
        self._initialize_mode()
    
    def _initialize_mode(self):
        """모드별 컴포넌트 초기화"""
        
        if self.mode == TradingMode.BACKTEST:
            # 백테스트 모드: ExecutionEngine 사용
            self.exec_engine = FullExecutionPipelineV73(self.config.exec_cfg)
            self.logger.info("[Orchestrator] Initialized in BACKTEST mode")
        
        else:
            # 실거래 모드: Broker + OrderGenerator + KillSwitch
            if not self.config.broker_config:
                raise ValueError("broker_config required for live/paper mode")
            
            self.broker = UnifiedBrokerV2(self.config.broker_config)
            self.order_gen = OrderGenerator(
                min_order_value=1000.0,
                min_position_change=0.05,
                max_orders_per_batch=50,
                use_limit_orders=False
            )
            self.kill_switch = get_kill_switch()
            
            self.logger.info(f"[Orchestrator] Initialized in {self.mode.value.upper()} mode")
    
    async def connect(self):
        """브로커 연결 (실거래 모드만)"""
        if self.broker:
            await self.broker.connect()
            self.logger.info("[Orchestrator] Broker connected")
    
    async def disconnect(self):
        """브로커 연결 해제 (실거래 모드만)"""
        if self.broker:
            await self.broker.disconnect()
            self.logger.info("[Orchestrator] Broker disconnected")
    
    # ========================================================
    #   ALPHA PIPELINE (ASYNC)
    # ========================================================
    async def fetch_alpha(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """알파 데이터 수집 (GEX/DIX/WhisperZ/LLM)"""
        await self.alpha.initialize()
        alpha = await self.alpha.fetch_all_alpha(symbol)
        return alpha
    
    # ========================================================
    #   MAIN SIGNAL GENERATION
    # ========================================================
    async def generate_signals(
        self,
        symbol_df_map: Dict[str, pd.DataFrame],
        llm_texts: Optional[Dict[str, Dict[str, Optional[str]]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        모든 전략 엔진에서 시그널 생성
        
        Returns:
            {
                "alpha": {...},
                "phoenix": {...},
                "momentum": {...},
                "meanrev": {...},
                "meta": {...}
            }
        """
        
        all_alpha = {}
        all_phoenix = {}
        all_momentum = {}
        all_mr = {}
        all_meta = {}
        
        # Alpha 데이터 수집 (병렬)
        tasks = {
            sym: asyncio.create_task(self.fetch_alpha(sym, df))
            for sym, df in symbol_df_map.items()
        }
        for sym in tasks:
            all_alpha[sym] = await tasks[sym]
        
        # 각 엔진별 시그널 생성
        for sym, df in symbol_df_map.items():
            alpha_extra = all_alpha[sym]
            
            # Phoenix
            p = self.phoenix.generate_signal(
                df,
                gex=alpha_extra.get("gex", 0),
                dix=alpha_extra.get("dix", 0),
                whisper_z=alpha_extra.get("whisper_z", 0),
                llm_alpha=None
            )
            all_phoenix[sym] = p
            
            # Momentum
            m = self.momentum.generate_signal(
                df,
                alpha_extra=alpha_extra
            )
            all_momentum[sym] = m
            
            # Mean Reversion
            mr = self.meanrev.generate_signal({sym: df["close"]})
            all_mr[sym] = mr
            
            # Meta Ensemble
            meta_out = self.meta.predict(
                phoenix_out=p,
                momentum_out=m.get("momentum", 0),
                meanrev_out=mr.get("signal", 0),
                alpha_features=alpha_extra
            )
            all_meta[sym] = meta_out
        
        return {
            "alpha": all_alpha,
            "phoenix": all_phoenix,
            "momentum": all_momentum,
            "meanrev": all_mr,
            "meta": all_meta
        }
    
    # ========================================================
    #   POSITION SIZING (Risk Manager)
    # ========================================================
    async def size_positions(
        self,
        signals: Dict[str, Dict[str, Any]],
        alpha_map: Dict[str, Dict[str, Any]],
        current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        포지션 사이징 (목표 포지션 크기 계산)
        
        Returns:
            symbol → target_notional ($)
        """
        
        sizes = {}
        
        for sym, meta_out in signals["meta"].items():
            final_sig = meta_out.get("signal", 0)
            conf = meta_out.get("confidence", 0.5)
            risk_s = meta_out.get("risk_score", 0.5)
            
            # 간단한 volatility proxy
            vol = 0.02  # 기본값
            
            # Volume series (VPIN용)
            vs = alpha_map[sym].get("volume_series", pd.Series([100] * 200))
            
            # Regime
            regime = {"type": "normal"}
            
            # GEX
            gex = alpha_map[sym].get("gex", 0)
            
            # Risk Manager로 포지션 크기 계산
            size = self.risk.calculate_position_size(
                signal=final_sig,
                confidence=conf,
                risk_score=risk_s,
                volatility=vol,
                equity=self.equity,
                volume_series=vs,
                regime=regime,
                gex=gex
            )
            
            sizes[sym] = size
        
        return sizes
    
    # ========================================================
    #   EXECUTION (모드별 분기)
    # ========================================================
    async def execute_trades(
        self,
        signals: Dict[str, Dict[str, Any]],
        target_sizes: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        거래 실행 (모드별 분기)
        
        백테스트 모드: ExecutionEngine (가상)
        실거래 모드: UnifiedBroker (실제)
        """
        
        if self.mode == TradingMode.BACKTEST:
            return await self._execute_backtest(signals, target_sizes, current_prices)
        else:
            return await self._execute_live(signals, target_sizes, current_prices)
    
    async def _execute_backtest(
        self,
        signals: Dict[str, Dict[str, Any]],
        target_sizes: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """백테스트 모드 실행 (가상)"""
        
        # ExecutionEngine 사용
        results = {}
        
        for sym, size in target_sizes.items():
            price = current_prices.get(sym, 0)
            if price <= 0:
                continue
            
            qty = size / price
            
            # 가상 실행
            results[sym] = {
                "symbol": sym,
                "qty": qty,
                "price": price,
                "status": "FILLED",
                "mode": "backtest"
            }
        
        self.logger.info(f"[Orchestrator] Backtest execution: {len(results)} orders")
        return {"orders": results, "mode": "backtest"}
    
    async def _execute_live(
        self,
        signals: Dict[str, Dict[str, Any]],
        target_sizes: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """실거래 모드 실행 (실제 브로커)"""
        
        # KillSwitch 체크
        if self.kill_switch.is_tripped():
            self.logger.critical("[Orchestrator] KillSwitch TRIPPED! Flattening positions...")
            await self._flatten_all_positions()
            return {"status": "KILL_SWITCH_TRIPPED", "mode": self.mode.value}
        
        # 현재 포지션 조회
        positions_df = await self.broker.get_positions()
        current_positions = {}
        if not positions_df.empty:
            current_positions = positions_df["quantity"].to_dict()
        
        # 시그널 추출
        signal_values = {
            sym: signals["meta"][sym].get("signal", 0)
            for sym in target_sizes.keys()
        }
        
        # OrderGenerator로 주문 생성
        orders = self.order_gen.generate_orders(
            signals=signal_values,
            target_sizes=target_sizes,
            current_positions=current_positions,
            current_prices=current_prices
        )
        
        if not orders:
            self.logger.info("[Orchestrator] No orders to execute")
            return {"status": "NO_ORDERS", "mode": self.mode.value}
        
        # DataFrame 변환
        orders_df = self.order_gen.orders_to_dataframe(orders)
        
        # 실제 주문 실행
        self.logger.info(f"[Orchestrator] Executing {len(orders)} orders via broker...")
        results_df = await self.broker.place_orders(orders_df)
        
        # 결과 로깅
        if not results_df.empty:
            success_count = len(results_df[results_df["status"] == "SUCCESS"])
            self.logger.info(
                f"[Orchestrator] Execution complete: "
                f"{success_count}/{len(results_df)} successful"
            )
        
        return {
            "orders": orders_df.to_dict("records"),
            "results": results_df.to_dict("records") if not results_df.empty else [],
            "mode": self.mode.value
        }
    
    async def _flatten_all_positions(self):
        """모든 포지션 긴급 청산"""
        try:
            await self.broker.flatten_all_positions()
            self.logger.critical("[Orchestrator] All positions flattened")
        except Exception as e:
            self.logger.error(f"[Orchestrator] Failed to flatten positions: {e}")
    
    # ========================================================
    #   MAIN LOOP
    # ========================================================
    async def run_step(
        self,
        symbol_df_map: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        단일 스텝 실행 (백테스트/실거래 공통)
        
        1. 시그널 생성
        2. 포지션 사이징
        3. 주문 실행
        4. 모니터링
        """
        
        step_start = time.time()
        
        # 1. 시그널 생성
        signals = await self.generate_signals(symbol_df_map)
        
        # 2. 포지션 사이징
        target_sizes = await self.size_positions(
            signals=signals,
            alpha_map=signals["alpha"],
            current_prices=current_prices
        )
        
        # 3. 주문 실행
        execution_results = await self.execute_trades(
            signals=signals,
            target_sizes=target_sizes,
            current_prices=current_prices
        )
        
        # 4. 모니터링
        step_time = time.time() - step_start
        
        return {
            "signals": signals,
            "target_sizes": target_sizes,
            "execution": execution_results,
            "step_time": step_time,
            "timestamp": datetime.now().isoformat()
        }
