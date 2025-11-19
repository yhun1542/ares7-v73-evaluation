# src/connectors/unified_broker_v2.py
"""
Unified Broker v2 - Complete Implementation
Integrates KIS and IBKR brokers with intelligent routing
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd

from .kis_broker import KoreaBrokerClient, KoreaBrokerConfig
try:
    from .ibkr_broker import IBKRBrokerClient
except ImportError:
    IBKRBrokerClient = None
    
logger = logging.getLogger(__name__)


class UnifiedBrokerV2:
    """
    통합 브로커 인터페이스 v2
    - KIS (한국투자증권) 완전 통합
    - IBKR (Interactive Brokers) 옵션
    - 시장별 지능형 라우팅
    - 통합 포지션/잔고 관리
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Config example:
        {
            "kis": {
                "enabled": True,
                "svr": "vps",  # "prod" or "vps"
                "market": "US",  # "US" or "KR"
                "exchange": "NASD"  # NASD, NYSE, AMEX, KOSPI, KOSDAQ
            },
            "ibkr": {
                "enabled": False,
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 1
            },
            "routing": {
                "us": "kis",  # "kis" or "ibkr"
                "kr": "kis",  # "kis" only
                "default": "kis"
            }
        }
        """
        self.config = config
        self.clients = {}
        
        # KIS 클라이언트 초기화
        kis_cfg = config.get("kis", {})
        if kis_cfg.get("enabled", True):
            self.clients["kis"] = KoreaBrokerClient(
                KoreaBrokerConfig(
                    svr=kis_cfg.get("svr", "vps"),
                    market=kis_cfg.get("market", "US"),
                    exchange=kis_cfg.get("exchange", "NASD")
                )
            )
            logger.info("[UnifiedBroker] KIS client initialized")
        
        # IBKR 클라이언트 초기화 (옵션)
        ibkr_cfg = config.get("ibkr", {})
        if ibkr_cfg.get("enabled", False) and IBKRBrokerClient:
            try:
                self.clients["ibkr"] = IBKRBrokerClient(ibkr_cfg)
                logger.info("[UnifiedBroker] IBKR client initialized")
            except Exception as e:
                logger.warning(f"[UnifiedBroker] IBKR client init failed: {e}")
        
        # 라우팅 설정
        routing = config.get("routing", {})
        self.route_us = routing.get("us", "kis")
        self.route_kr = routing.get("kr", "kis")
        self.default_broker = routing.get("default", "kis")
        
        self.connected = False

    # ========== 연결 관리 ==========

    async def connect(self):
        """모든 브로커 연결"""
        tasks = []
        for name, client in self.clients.items():
            logger.info(f"[UnifiedBroker] Connecting {name}...")
            tasks.append(client.connect())
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(self.clients.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"[UnifiedBroker] {name} connection failed: {result}")
                else:
                    logger.info(f"[UnifiedBroker] {name} connected successfully")
        
        self.connected = True
        logger.info(f"[UnifiedBroker] All brokers connected. Active: {list(self.clients.keys())}")

    async def disconnect(self):
        """모든 브로커 연결 해제"""
        tasks = []
        for name, client in self.clients.items():
            logger.info(f"[UnifiedBroker] Disconnecting {name}...")
            tasks.append(client.disconnect())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.connected = False
        logger.info("[UnifiedBroker] All brokers disconnected")

    # ========== 라우팅 헬퍼 ==========

    def _get_broker_for_symbol(self, symbol: str) -> str:
        """심볼에 따른 브로커 선택"""
        # 한국 주식: 6자리 숫자
        if symbol.isdigit() and len(symbol) == 6:
            return self.route_kr
        # 미국 주식: 알파벳
        elif symbol.isalpha():
            return self.route_us
        # 기본값
        return self.default_broker

    def _get_client(self, broker_name: str):
        """브로커 클라이언트 가져오기"""
        client = self.clients.get(broker_name)
        if not client:
            # Fallback to default
            client = self.clients.get(self.default_broker)
            if not client:
                # Get first available
                if self.clients:
                    client = list(self.clients.values())[0]
        return client

    # ========== 포지션/잔고 조회 ==========

    async def get_positions(self) -> pd.DataFrame:
        """모든 브로커의 포지션 통합 조회"""
        all_positions = []
        
        for name, client in self.clients.items():
            try:
                positions = await client.get_positions()
                if not positions.empty:
                    positions["broker"] = name
                    all_positions.append(positions)
                    logger.info(f"[UnifiedBroker] Got {len(positions)} positions from {name}")
            except Exception as e:
                logger.error(f"[UnifiedBroker] Failed to get positions from {name}: {e}")
        
        if not all_positions:
            return pd.DataFrame()
        
        # 통합 DataFrame
        combined = pd.concat(all_positions, ignore_index=False)
        
        # 중복 제거 (같은 symbol이 여러 브로커에 있는 경우)
        if not combined.empty:
            combined = combined.reset_index()
            combined = combined.groupby("symbol").agg({
                "quantity": "sum",
                "market_value": "sum",
                "avg_price": "mean",
                "last_price": "last",
                "pnl": "sum",
                "broker": "first"
            })
        
        return combined

    async def get_balance(self) -> Dict[str, Any]:
        """모든 브로커의 잔고 통합"""
        total_value = 0
        total_cash = 0
        total_pnl = 0
        broker_balances = {}
        
        for name, client in self.clients.items():
            try:
                balance = await client.get_balance()
                broker_balances[name] = balance
                
                total_value += balance.get("total_value", 0)
                total_cash += balance.get("cash", 0)
                total_pnl += balance.get("pnl", 0)
                
                logger.info(f"[UnifiedBroker] {name} balance: ${balance.get('total_value', 0):,.2f}")
                
            except Exception as e:
                logger.error(f"[UnifiedBroker] Failed to get balance from {name}: {e}")
                broker_balances[name] = {"error": str(e)}
        
        return {
            "total_value": total_value,
            "total_cash": total_cash,
            "total_pnl": total_pnl,
            "brokers": broker_balances,
            "timestamp": datetime.now().isoformat()
        }

    # ========== 주문 실행 ==========

    async def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """
        주문 실행 (브로커별 라우팅)
        
        orders DataFrame columns:
        - symbol: 종목코드
        - side: "BUY" or "SELL"
        - qty: 수량
        - price: 가격 (optional, 0 = market order)
        - order_type: "MARKET" or "LIMIT" (optional)
        - broker: 특정 브로커 지정 (optional)
        """
        if orders.empty:
            return pd.DataFrame()
        
        # 브로커별로 주문 분류
        broker_orders = {}
        
        for _, order in orders.iterrows():
            # 브로커 결정
            if "broker" in order and order["broker"]:
                broker_name = order["broker"]
            else:
                broker_name = self._get_broker_for_symbol(order["symbol"])
            
            if broker_name not in broker_orders:
                broker_orders[broker_name] = []
            
            broker_orders[broker_name].append(order.to_dict())
        
        # 브로커별 주문 실행
        all_results = []
        
        for broker_name, order_list in broker_orders.items():
            client = self._get_client(broker_name)
            if not client:
                logger.error(f"[UnifiedBroker] No client for broker: {broker_name}")
                continue
            
            try:
                orders_df = pd.DataFrame(order_list)
                results = await client.place_orders(orders_df)
                results["broker"] = broker_name
                all_results.append(results)
                
                logger.info(f"[UnifiedBroker] Placed {len(order_list)} orders via {broker_name}")
                
            except Exception as e:
                logger.error(f"[UnifiedBroker] Order placement failed on {broker_name}: {e}")
                
                # 실패한 주문들 기록
                for order in order_list:
                    all_results.append(pd.DataFrame([{
                        "symbol": order["symbol"],
                        "side": order["side"],
                        "qty": order["qty"],
                        "status": "ERROR",
                        "message": str(e),
                        "broker": broker_name,
                        "timestamp": datetime.now().isoformat()
                    }]))
        
        if not all_results:
            return pd.DataFrame()
        
        return pd.concat(all_results, ignore_index=True)

    # ========== 주문 취소 ==========

    async def cancel_all_orders(self) -> Dict[str, int]:
        """
        모든 브로커의 미체결 주문 취소
        
        Returns:
            브로커별 취소된 주문 개수
        """
        logger.info("[UnifiedBroker] Cancelling all orders across all brokers...")
        
        cancel_counts = {}
        total_cancelled = 0
        
        for name, client in self.clients.items():
            try:
                # 각 브로커의 cancel_all 메서드 호출
                if hasattr(client, "cancel_all_orders"):
                    count = await client.cancel_all_orders()
                else:
                    # Legacy method
                    await client.cancel_all()
                    count = 0  # Unknown count
                
                cancel_counts[name] = count
                total_cancelled += count
                
                logger.info(f"[UnifiedBroker] {name}: {count} orders cancelled")
                
            except Exception as e:
                logger.error(f"[UnifiedBroker] Failed to cancel orders on {name}: {e}")
                cancel_counts[name] = -1  # Error indicator
        
        logger.info(f"[UnifiedBroker] Total {total_cancelled} orders cancelled across all brokers")
        
        return cancel_counts

    async def get_open_orders(self) -> pd.DataFrame:
        """모든 브로커의 미체결 주문 조회"""
        all_orders = []
        
        for name, client in self.clients.items():
            try:
                if hasattr(client, "get_open_orders"):
                    orders = await client.get_open_orders()
                    if not orders.empty:
                        orders["broker"] = name
                        all_orders.append(orders)
                        logger.info(f"[UnifiedBroker] {name}: {len(orders)} open orders")
            except Exception as e:
                logger.error(f"[UnifiedBroker] Failed to get open orders from {name}: {e}")
        
        if not all_orders:
            return pd.DataFrame()
        
        return pd.concat(all_orders, ignore_index=True)

    # ========== 포지션 관리 ==========

    async def flatten_all_positions(self) -> pd.DataFrame:
        """모든 포지션 청산"""
        logger.info("[UnifiedBroker] Flattening all positions...")
        
        all_results = []
        
        for name, client in self.clients.items():
            try:
                results = await client.flatten_all_positions()
                if not results.empty:
                    results["broker"] = name
                    all_results.append(results)
                    logger.info(f"[UnifiedBroker] {name}: {len(results)} positions flattened")
            except Exception as e:
                logger.error(f"[UnifiedBroker] Failed to flatten positions on {name}: {e}")
        
        if not all_results:
            logger.info("[UnifiedBroker] No positions to flatten")
            return pd.DataFrame()
        
        combined = pd.concat(all_results, ignore_index=True)
        logger.info(f"[UnifiedBroker] Total {len(combined)} positions flattened")
        
        return combined

    async def reduce_positions(self, ratio: float = 0.5) -> pd.DataFrame:
        """포지션 축소"""
        logger.info(f"[UnifiedBroker] Reducing all positions by {ratio:.0%}...")
        
        all_results = []
        
        for name, client in self.clients.items():
            try:
                if hasattr(client, "reduce_positions"):
                    results = await client.reduce_positions(ratio)
                    if not results.empty:
                        results["broker"] = name
                        all_results.append(results)
                        logger.info(f"[UnifiedBroker] {name}: {len(results)} positions reduced")
            except Exception as e:
                logger.error(f"[UnifiedBroker] Failed to reduce positions on {name}: {e}")
        
        if not all_results:
            return pd.DataFrame()
        
        return pd.concat(all_results, ignore_index=True)

    # ========== 시세 조회 ==========

    async def get_price(self, symbol: str) -> Dict[str, float]:
        """현재가 조회"""
        broker_name = self._get_broker_for_symbol(symbol)
        client = self._get_client(broker_name)
        
        if not client:
            logger.error(f"[UnifiedBroker] No client available for {symbol}")
            return {"symbol": symbol, "last": 0}
        
        try:
            if hasattr(client, "get_price"):
                return await client.get_price(symbol)
            else:
                logger.warning(f"[UnifiedBroker] {broker_name} doesn't support get_price")
                return {"symbol": symbol, "last": 0}
        except Exception as e:
            logger.error(f"[UnifiedBroker] Failed to get price for {symbol}: {e}")
            return {"symbol": symbol, "last": 0}

    # ========== 헬스 체크 ==========

    async def health_check(self) -> Dict[str, Any]:
        """브로커 상태 체크"""
        health = {
            "status": "healthy" if self.connected else "disconnected",
            "brokers": {},
            "timestamp": datetime.now().isoformat()
        }
        
        for name, client in self.clients.items():
            try:
                # Simple connectivity check
                balance = await client.get_balance()
                health["brokers"][name] = {
                    "status": "connected",
                    "total_value": balance.get("total_value", 0)
                }
            except Exception as e:
                health["brokers"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                health["status"] = "degraded"
        
        return health
