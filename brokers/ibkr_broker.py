#!/usr/bin/env python3
"""
IBKR (Interactive Brokers) Broker v2 - Direct Trading Implementation
직접 거래 기능이 포함된 완전 구현
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from ib_insync import *

logger = logging.getLogger(__name__)


class IBKRBroker:
    """IBKR 브로커 - 직접 거래 구현"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.host = config.get("host", "127.0.0.1")
        self.port = int(config.get("port", 7497))  # 7497 for paper, 7496 for live
        self.client_id = int(config.get("client_id", 1))
        self.account = config.get("account", "")
        self.mode = config.get("mode", "paper")  # paper or live
        
        self.ib = IB()
        self.connected = False
        self.positions_cache = {}
        self.orders_cache = {}
        
    async def connect(self) -> bool:
        """브로커 연결"""
        try:
            # TWS/Gateway 연결
            await self.ib.connectAsync(
                host=self.host, 
                port=self.port, 
                clientId=self.client_id
            )
            
            self.connected = True
            
            # 계좌 정보 요청
            self.ib.reqAccountSummary()
            
            # 실시간 포지션 업데이트
            self.ib.reqPositions()
            
            logger.info(f"[IBKR] 브로커 연결 성공 ({self.mode} mode)")
            logger.info(f"[IBKR] 계좌: {self.account if self.account else 'Auto-detected'}")
            
            return True
            
        except Exception as e:
            logger.error(f"[IBKR] 연결 실패: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """연결 해제"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("[IBKR] 브로커 연결 해제")
    
    async def get_balance(self) -> Dict[str, Any]:
        """계좌 잔고 조회"""
        if not self.connected:
            await self.connect()
        
        try:
            # 계좌 요약 정보 가져오기
            summary = self.ib.accountSummary(account=self.account)
            
            balance = {
                "cash": 0,
                "total_value": 0,
                "buying_power": 0,
                "positions": pd.DataFrame()
            }
            
            for item in summary:
                if item.tag == "CashBalance":
                    balance["cash"] = float(item.value)
                elif item.tag == "NetLiquidation":
                    balance["total_value"] = float(item.value)
                elif item.tag == "BuyingPower":
                    balance["buying_power"] = float(item.value)
            
            # 포지션 정보 추가
            balance["positions"] = await self.get_positions()
            
            return balance
            
        except Exception as e:
            logger.error(f"[IBKR] 잔고 조회 실패: {e}")
            return {"cash": 0, "total_value": 0, "buying_power": 0, "positions": pd.DataFrame()}
    
    async def get_positions(self) -> pd.DataFrame:
        """보유 포지션 조회"""
        if not self.connected:
            await self.connect()
        
        try:
            positions_list = []
            positions = self.ib.positions(account=self.account)
            
            for position in positions:
                contract = position.contract
                
                # 현재가 조회
                ticker = self.ib.reqTickers(contract)[0] if self.ib.reqTickers(contract) else None
                current_price = ticker.last if ticker and ticker.last else position.avgCost
                
                pos_dict = {
                    "symbol": contract.symbol,
                    "exchange": contract.exchange,
                    "quantity": position.position,
                    "avg_price": position.avgCost,
                    "current_price": current_price,
                    "pnl": position.unrealizedPNL,
                    "pnl_pct": (position.unrealizedPNL / (position.avgCost * abs(position.position)) * 100) 
                              if position.avgCost and position.position else 0
                }
                
                positions_list.append(pos_dict)
                self.positions_cache[contract.symbol] = pos_dict
            
            df = pd.DataFrame(positions_list)
            logger.info(f"[IBKR] {len(positions_list)}개 포지션 조회")
            
            return df
            
        except Exception as e:
            logger.error(f"[IBKR] 포지션 조회 실패: {e}")
            return pd.DataFrame()
    
    async def place_order(self, symbol: str, quantity: int, order_type: str = "MARKET",
                          price: Optional[float] = None, side: str = "BUY",
                          exchange: str = "SMART") -> Dict[str, Any]:
        """주문 실행 - 직접 거래"""
        if not self.connected:
            await self.connect()
        
        try:
            # Contract 생성
            contract = Stock(symbol, exchange, 'USD')
            
            # Order 생성
            if order_type.upper() == "MARKET":
                order = MarketOrder(
                    action=side.upper(),
                    totalQuantity=abs(quantity)
                )
            elif order_type.upper() == "LIMIT":
                order = LimitOrder(
                    action=side.upper(),
                    totalQuantity=abs(quantity),
                    lmtPrice=price or 0
                )
            elif order_type.upper() == "STOP":
                order = StopOrder(
                    action=side.upper(),
                    totalQuantity=abs(quantity),
                    stopPrice=price or 0
                )
            else:
                logger.error(f"[IBKR] 지원하지 않는 주문 타입: {order_type}")
                return {"success": False, "message": "Unsupported order type"}
            
            # 주문 전송
            trade = self.ib.placeOrder(contract, order)
            
            # 주문 상태 대기 (최대 5초)
            for _ in range(50):
                await asyncio.sleep(0.1)
                if trade.orderStatus.status in ['Filled', 'Cancelled', 'Error']:
                    break
            
            # 결과 반환
            result = {
                "success": trade.orderStatus.status == 'Filled',
                "order_id": trade.order.orderId,
                "symbol": symbol,
                "quantity": quantity,
                "side": side,
                "order_type": order_type,
                "status": trade.orderStatus.status,
                "filled_qty": trade.orderStatus.filled,
                "avg_fill_price": trade.orderStatus.avgFillPrice,
                "message": f"Order {trade.orderStatus.status}"
            }
            
            # 주문 캐시 저장
            self.orders_cache[trade.order.orderId] = result
            
            if result["success"]:
                logger.info(f"[IBKR] 주문 체결: {symbol} {side} {quantity}주 @ {result['avg_fill_price']}")
            else:
                logger.warning(f"[IBKR] 주문 상태: {result['status']} - {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"[IBKR] 주문 실행 실패: {e}")
            return {
                "success": False,
                "message": str(e)
            }
    
    async def cancel_order(self, order_id: int) -> bool:
        """주문 취소"""
        if not self.connected:
            await self.connect()
        
        try:
            # 열린 주문 찾기
            for trade in self.ib.openTrades():
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"[IBKR] 주문 취소: {order_id}")
                    return True
            
            logger.warning(f"[IBKR] 취소할 주문을 찾을 수 없음: {order_id}")
            return False
            
        except Exception as e:
            logger.error(f"[IBKR] 주문 취소 실패: {e}")
            return False
    
    async def get_open_orders(self) -> pd.DataFrame:
        """미체결 주문 조회"""
        if not self.connected:
            await self.connect()
        
        try:
            orders_list = []
            
            for trade in self.ib.openTrades():
                order = trade.order
                contract = trade.contract
                status = trade.orderStatus
                
                orders_list.append({
                    "order_id": order.orderId,
                    "symbol": contract.symbol,
                    "side": order.action,
                    "quantity": order.totalQuantity,
                    "filled_qty": status.filled,
                    "remaining_qty": status.remaining,
                    "order_type": order.orderType,
                    "limit_price": order.lmtPrice if hasattr(order, 'lmtPrice') else None,
                    "stop_price": order.auxPrice if hasattr(order, 'auxPrice') else None,
                    "status": status.status,
                    "time": trade.log[0].time if trade.log else None
                })
            
            df = pd.DataFrame(orders_list)
            logger.info(f"[IBKR] {len(orders_list)}개 미체결 주문 조회")
            
            return df
            
        except Exception as e:
            logger.error(f"[IBKR] 미체결 주문 조회 실패: {e}")
            return pd.DataFrame()
    
    async def cancel_all_orders(self) -> int:
        """모든 미체결 주문 취소"""
        if not self.connected:
            await self.connect()
        
        try:
            open_trades = self.ib.openTrades()
            cancelled_count = 0
            
            for trade in open_trades:
                try:
                    self.ib.cancelOrder(trade.order)
                    cancelled_count += 1
                    logger.info(f"[IBKR] 주문 취소: {trade.contract.symbol} - Order ID: {trade.order.orderId}")
                except Exception as e:
                    logger.error(f"[IBKR] 주문 취소 실패: {trade.order.orderId} - {e}")
            
            logger.info(f"[IBKR] 전체 주문 취소 완료: {cancelled_count}건")
            return cancelled_count
            
        except Exception as e:
            logger.error(f"[IBKR] 전체 주문 취소 실패: {e}")
            return 0
    
    async def flatten_all_positions(self) -> int:
        """모든 포지션 청산"""
        if not self.connected:
            await self.connect()
        
        try:
            positions = self.ib.positions(account=self.account)
            closed_count = 0
            
            for position in positions:
                if position.position != 0:  # 포지션이 있는 경우만
                    contract = position.contract
                    quantity = abs(position.position)
                    side = "SELL" if position.position > 0 else "BUY"
                    
                    # 시장가로 청산
                    result = await self.place_order(
                        symbol=contract.symbol,
                        quantity=quantity,
                        order_type="MARKET",
                        side=side,
                        exchange=contract.exchange
                    )
                    
                    if result.get("success"):
                        closed_count += 1
                        logger.info(f"[IBKR] 포지션 청산: {contract.symbol} {quantity}주")
                    else:
                        logger.error(f"[IBKR] 포지션 청산 실패: {contract.symbol}")
            
            logger.info(f"[IBKR] 전체 포지션 청산 완료: {closed_count}건")
            return closed_count
            
        except Exception as e:
            logger.error(f"[IBKR] 포지션 청산 실패: {e}")
            return 0
    
    async def get_price(self, symbol: str, exchange: str = "SMART") -> Dict[str, float]:
        """실시간 가격 조회"""
        if not self.connected:
            await self.connect()
        
        try:
            contract = Stock(symbol, exchange, 'USD')
            ticker = self.ib.reqMktData(contract)
            
            # 데이터 수신 대기 (최대 2초)
            for _ in range(20):
                await asyncio.sleep(0.1)
                if ticker.last or ticker.bid or ticker.ask:
                    break
            
            # 가격 정보 반환
            return {
                "current": ticker.last or 0,
                "bid": ticker.bid or 0,
                "ask": ticker.ask or 0,
                "open": ticker.open or 0,
                "high": ticker.high or 0,
                "low": ticker.low or 0,
                "close": ticker.close or ticker.last or 0,
                "volume": ticker.volume or 0,
                "bid_size": ticker.bidSize or 0,
                "ask_size": ticker.askSize or 0
            }
            
        except Exception as e:
            logger.error(f"[IBKR] 가격 조회 실패: {symbol} - {e}")
            return {"current": 0, "bid": 0, "ask": 0, "open": 0, 
                   "high": 0, "low": 0, "close": 0, "volume": 0}
    
    async def get_historical_data(self, symbol: str, duration: str = "1 M",
                                 bar_size: str = "1 day", exchange: str = "SMART") -> pd.DataFrame:
        """과거 데이터 조회"""
        if not self.connected:
            await self.connect()
        
        try:
            contract = Stock(symbol, exchange, 'USD')
            
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if bars:
                df = util.df(bars)
                df['symbol'] = symbol
                logger.info(f"[IBKR] {symbol} 과거 데이터 {len(df)}개 조회")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"[IBKR] 과거 데이터 조회 실패: {symbol} - {e}")
            return pd.DataFrame()


# 선택적 거래 모드 관리자
class IBKRTradingMode:
    """IBKR 거래 모드 선택 관리"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.broker = IBKRBroker(config)
        self.trading_enabled = False  # 기본값: 조회만 가능
        
    def enable_trading(self, confirmation: str = "") -> bool:
        """거래 기능 활성화 (사용자 확인 필요)"""
        if confirmation.upper() == "ENABLE_TRADING_CONFIRMED":
            self.trading_enabled = True
            logger.warning("[IBKR] 직접 거래 기능이 활성화되었습니다. 주의하세요!")
            return True
        else:
            logger.info("[IBKR] 거래 기능 활성화 실패. 확인 문자열이 일치하지 않습니다.")
            return False
    
    def disable_trading(self) -> None:
        """거래 기능 비활성화"""
        self.trading_enabled = False
        logger.info("[IBKR] 직접 거래 기능이 비활성화되었습니다.")
    
    async def place_order_safe(self, *args, **kwargs) -> Dict[str, Any]:
        """안전한 주문 실행 (거래 활성화 필요)"""
        if not self.trading_enabled:
            return {
                "success": False,
                "message": "Trading is disabled. Enable trading first with confirmation."
            }
        return await self.broker.place_order(*args, **kwargs)


# 테스트 함수
async def test_ibkr_broker():
    """IBKR 브로커 테스트"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    config = {
        "host": os.getenv("IBKR_HOST", "127.0.0.1"),
        "port": os.getenv("IBKR_PORT", "7497"),
        "client_id": os.getenv("IBKR_CLIENT_ID", "1"),
        "account": os.getenv("IBKR_ACCOUNT", ""),
        "mode": os.getenv("IBKR_MODE", "paper")
    }
    
    # 일반 브로커 (조회 기능만)
    broker = IBKRBroker(config)
    
    # 연결 테스트
    connected = await broker.connect()
    print(f"[TEST] 연결 상태: {connected}")
    
    if connected:
        # 잔고 조회
        balance = await broker.get_balance()
        print(f"[TEST] 잔고: ${balance.get('total_value', 0):,.2f}")
        
        # 포지션 조회
        positions = await broker.get_positions()
        print(f"[TEST] 포지션 수: {len(positions)}")
        
        # 가격 조회
        price = await broker.get_price("AAPL")
        print(f"[TEST] AAPL 가격: ${price.get('current', 0):.2f}")
        
        # 선택적 거래 모드 테스트
        trading_mode = IBKRTradingMode(config)
        
        # 거래 시도 (비활성화 상태)
        result = await trading_mode.place_order_safe(
            symbol="AAPL",
            quantity=10,
            order_type="MARKET",
            side="BUY"
        )
        print(f"[TEST] 거래 시도 결과: {result.get('message')}")
        
        # 거래 활성화
        if trading_mode.enable_trading("ENABLE_TRADING_CONFIRMED"):
            print("[TEST] 거래 기능 활성화됨")
            # 이제 실제 거래 가능
            # result = await trading_mode.place_order_safe(...)
        
        await broker.disconnect()
    
    print("[TEST] 테스트 완료")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트 실행
    asyncio.run(test_ibkr_broker())
