#!/usr/bin/env python3
"""
ARES-7 v73 FULL - Main Entry Point
백테스트 / 실거래 통합 실행
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.ares_orchestrator_integrated import (
    AresOrchestratorV73Full,
    OrchestratorConfig,
    TradingMode
)
from governance.kill_switch import get_kill_switch


# ============================================================
#   LOGGING SETUP
# ============================================================

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """로깅 설정"""
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        handlers.append(
            logging.FileHandler(log_dir / log_file)
        )
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


# ============================================================
#   BROKER CONFIG LOADER
# ============================================================

def load_broker_config(env: str = "production") -> Dict:
    """
    환경 변수에서 브로커 설정 로드
    
    Args:
        env: "production", "vps", "paper"
    """
    
    config = {
        "kis": {
            "enabled": True,
            "svr": "prod" if env == "production" else "vps",
            "market": os.getenv("KIS_MARKET", "US"),
            "exchange": os.getenv("KIS_EXCHANGE", "NASD")
        },
        "ibkr": {
            "enabled": os.getenv("IBKR_ENABLED", "false").lower() == "true",
            "host": os.getenv("IBKR_HOST", "127.0.0.1"),
            "port": int(os.getenv("IBKR_PORT", 7497)),
            "client_id": int(os.getenv("IBKR_CLIENT_ID", 1))
        },
        "routing": {
            "us": os.getenv("BROKER_ROUTE_US", "kis"),
            "kr": os.getenv("BROKER_ROUTE_KR", "kis"),
            "default": os.getenv("BROKER_ROUTE_DEFAULT", "kis")
        }
    }
    
    return config


# ============================================================
#   BACKTEST MODE
# ============================================================

async def run_backtest(
    symbols: List[str],
    start_date: str,
    end_date: str,
    capital: float
):
    """
    백테스트 모드 실행
    
    Args:
        symbols: 거래 심볼 리스트
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        capital: 초기 자본
    """
    
    logger = logging.getLogger("BACKTEST")
    logger.info("=" * 80)
    logger.info("ARES-7 v73 BACKTEST MODE")
    logger.info("=" * 80)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Period: {start_date} ~ {end_date}")
    logger.info(f"Capital: ${capital:,.2f}")
    logger.info("=" * 80)
    
    # Orchestrator 초기화
    config = OrchestratorConfig(
        mode=TradingMode.BACKTEST,
        capital=capital
    )
    
    orchestrator = AresOrchestratorV73Full(config)
    
    # 백테스트 데이터 로드 (여기서는 더미 데이터)
    # 실제로는 데이터 소스에서 로드
    logger.info("Loading market data...")
    
    # TODO: 실제 데이터 로드 로직
    # symbol_df_map = load_market_data(symbols, start_date, end_date)
    
    # 더미 데이터 생성
    dates = pd.date_range(start_date, end_date, freq="D")
    symbol_df_map = {}
    
    for symbol in symbols:
        df = pd.DataFrame({
            "date": dates,
            "open": 100 + np.random.randn(len(dates)).cumsum(),
            "high": 102 + np.random.randn(len(dates)).cumsum(),
            "low": 98 + np.random.randn(len(dates)).cumsum(),
            "close": 100 + np.random.randn(len(dates)).cumsum(),
            "volume": np.random.randint(1000000, 10000000, len(dates))
        })
        symbol_df_map[symbol] = df
    
    logger.info(f"Loaded data for {len(symbol_df_map)} symbols")
    
    # 백테스트 실행
    logger.info("Starting backtest...")
    
    results = []
    
    for i, date in enumerate(dates):
        logger.info(f"[{i+1}/{len(dates)}] Processing {date.strftime('%Y-%m-%d')}...")
        
        # 현재까지의 데이터만 사용
        current_data = {
            sym: df.iloc[:i+1]
            for sym, df in symbol_df_map.items()
        }
        
        # 현재 가격
        current_prices = {
            sym: df.iloc[i]["close"]
            for sym, df in symbol_df_map.items()
        }
        
        # 스텝 실행
        step_result = await orchestrator.run_step(current_data, current_prices)
        results.append(step_result)
    
    logger.info("=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info(f"Total steps: {len(results)}")
    logger.info("=" * 80)
    
    # 결과 저장
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    import json
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")


# ============================================================
#   LIVE/PAPER MODE
# ============================================================

async def run_live(
    symbols: List[str],
    capital: float,
    mode: str = "paper"
):
    """
    실거래/페이퍼 모드 실행
    
    Args:
        symbols: 거래 심볼 리스트
        capital: 초기 자본
        mode: "paper" or "live"
    """
    
    logger = logging.getLogger("LIVE")
    logger.info("=" * 80)
    logger.info(f"ARES-7 v73 {mode.upper()} MODE")
    logger.info("=" * 80)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Capital: ${capital:,.2f}")
    logger.info("=" * 80)
    
    # 브로커 설정 로드
    env = "vps" if mode == "paper" else "production"
    broker_config = load_broker_config(env)
    
    logger.info(f"Broker config: {broker_config}")
    
    # Orchestrator 초기화
    trading_mode = TradingMode.PAPER if mode == "paper" else TradingMode.LIVE
    
    config = OrchestratorConfig(
        mode=trading_mode,
        capital=capital,
        broker_config=broker_config
    )
    
    orchestrator = AresOrchestratorV73Full(config)
    
    # 브로커 연결
    logger.info("Connecting to broker...")
    await orchestrator.connect()
    
    # KillSwitch 초기화
    kill_switch = get_kill_switch()
    logger.info(f"KillSwitch status: {kill_switch.get_status()}")
    
    try:
        # 메인 루프
        logger.info("Starting trading loop...")
        
        step = 0
        while True:
            step += 1
            
            # KillSwitch 체크
            if kill_switch.is_halted():
                logger.critical("KillSwitch HALTED! Stopping...")
                break
            
            logger.info(f"[Step {step}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 시장 데이터 수집
            # TODO: 실시간 데이터 소스 연결
            # symbol_df_map = fetch_realtime_data(symbols)
            
            # 더미 데이터 (실제로는 실시간 데이터)
            symbol_df_map = {}
            current_prices = {}
            
            for symbol in symbols:
                # 더미 데이터 생성
                df = pd.DataFrame({
                    "date": pd.date_range(end=datetime.now(), periods=200, freq="1min"),
                    "open": 100 + np.random.randn(200).cumsum(),
                    "high": 102 + np.random.randn(200).cumsum(),
                    "low": 98 + np.random.randn(200).cumsum(),
                    "close": 100 + np.random.randn(200).cumsum(),
                    "volume": np.random.randint(10000, 100000, 200)
                })
                symbol_df_map[symbol] = df
                current_prices[symbol] = df["close"].iloc[-1]
            
            # 스텝 실행
            step_result = await orchestrator.run_step(symbol_df_map, current_prices)
            
            logger.info(f"Step {step} complete: {step_result['execution']['status']}")
            
            # 대기 (예: 1분)
            await asyncio.sleep(60)
    
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Error in trading loop: {e}", exc_info=True)
    
    finally:
        # 브로커 연결 해제
        logger.info("Disconnecting from broker...")
        await orchestrator.disconnect()
        
        logger.info("=" * 80)
        logger.info("TRADING SESSION ENDED")
        logger.info("=" * 80)


# ============================================================
#   MAIN
# ============================================================

def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(
        description="ARES-7 v73 FULL - Trading System"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "paper", "live"],
        default="backtest",
        help="Trading mode"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ,IWM",
        help="Comma-separated list of symbols"
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=1000000.0,
        help="Initial capital"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date for backtest (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-01-01",
        help="End date for backtest (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env file"
    )
    
    args = parser.parse_args()
    
    # .env 파일 로드
    if Path(args.env_file).exists():
        load_dotenv(args.env_file)
        print(f"✓ Loaded environment from: {args.env_file}")
    else:
        print(f"⚠ Warning: {args.env_file} not found")
    
    # 로깅 설정
    log_file = f"ares7_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(args.log_level, log_file)
    
    # 심볼 파싱
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    # 모드별 실행
    if args.mode == "backtest":
        asyncio.run(run_backtest(
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            capital=args.capital
        ))
    else:
        asyncio.run(run_live(
            symbols=symbols,
            capital=args.capital,
            mode=args.mode
        ))


if __name__ == "__main__":
    main()
