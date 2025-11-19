#!/usr/bin/env python3
"""
ARES-7 v64 Emergency Stop Script - Production Ready
GPT 피드백 완전 반영 + 실제 브로커 연결 지원
"""

import sys
import os
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# 프로젝트 경로 추가
sys.path.insert(0, '/opt/ares7')

# 로깅 설정
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - EMERGENCY - %(message)s'
)
logger = logging.getLogger(__name__)

class EmergencyStop:
    """긴급 정지 시스템"""
    
    def __init__(self):
        self.log_dir = Path('/opt/ares7/logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.log_dir / 'emergency_stop.json'
        
    async def execute_emergency_stop(self, reason: str = "Manual emergency stop"):
        """긴급 정지 실행"""
        
        print("=" * 60)
        print("ARES-7 v64 EMERGENCY STOP INITIATED")
        print(f"Time: {datetime.now()}")
        print(f"Reason: {reason}")
        print("=" * 60)
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "operator": os.getenv("USER", "unknown"),
            "actions": []
        }
        
        try:
            # 1. Kill Switch 작동
            print("[1/5] Activating Kill Switch...")
            from governance.kill_switch import KillSwitch
            
            kill_switch = KillSwitch()
            kill_switch.trip("EMERGENCY", reason, "operator")
            print("✓ Kill Switch activated (EMERGENCY → FLATTEN mode)")
            status["actions"].append("kill_switch_activated")
            
            # 2. 브로커 연결
            print("[2/5] Connecting to brokers...")
            broker = await self._connect_broker()
            
            if broker:
                # 3. 미체결 주문 취소
                print("[3/5] Cancelling all pending orders...")
                await broker.cancel_all_orders()
                print("✓ All pending orders cancelled")
                status["actions"].append("orders_cancelled")
                
                # 포지션 정보 수집
                positions = await broker.get_positions()
                position_count = len(positions) if not positions.empty else 0
                status["positions_before"] = position_count
                
                # 4. 모든 포지션 청산
                print(f"[4/5] Flattening {position_count} positions...")
                await broker.flatten_all_positions()
                print("✓ All positions flattened")
                status["actions"].append("positions_flattened")
                
                # 최종 잔고 확인
                balance = await broker.get_balance()
                status["final_balance"] = balance
                
                # 브로커 연결 해제
                await broker.disconnect()
            else:
                print("⚠ Warning: Broker connection failed")
                status["error"] = "Broker connection failed"
            
            # 5. 시스템 프로세스 종료
            print("[5/5] Stopping system processes...")
            os.system("sudo supervisorctl stop ares7")
            print("✓ System processes stopped")
            status["actions"].append("processes_stopped")
            
            # 상태 저장
            self._save_status(status)
            
            # Telegram 알림 전송
            await self._send_alert(status)
            
            print("=" * 60)
            print("EMERGENCY STOP COMPLETE")
            print(f"Status saved to: {self.status_file}")
            print("System is now HALTED")
            print("Manual restart required to resume operations")
            print("=" * 60)
            
            return status
            
        except Exception as e:
            logger.critical(f"Emergency stop error: {e}", exc_info=True)
            print(f"✗ Error during emergency stop: {e}")
            status["error"] = str(e)
            self._save_status(status)
            raise
    
    async def _connect_broker(self) -> Optional[object]:
        """브로커 연결"""
        try:
            from brokers.unified_broker import UnifiedBrokerV2 as UnifiedBroker
            from dotenv import load_dotenv
            
            load_dotenv('/opt/ares7/.env')
            
            env = os.getenv("ENV", "production")
            
            # 브로커 설정
            config = {
                "kis": {
                    "env": env,
                    "app_key": os.getenv(f"KIS_APP_KEY_{env.upper()}", 
                                        os.getenv("KIS_APP_KEY_REAL")),
                    "app_secret": os.getenv(f"KIS_APP_SECRET_{env.upper()}", 
                                           os.getenv("KIS_APP_SECRET_REAL")),
                    "account": os.getenv(f"KIS_ACCOUNT_{env.upper()}", 
                                        os.getenv("KIS_ACCOUNT_REAL")),
                }
            }
            
            # IBKR 설정 (옵션)
            if os.getenv("IBKR_HOST"):
                config["ibkr"] = {
                    "host": os.getenv("IBKR_HOST", "127.0.0.1"),
                    "port": int(os.getenv("IBKR_PORT", 7497)),
                    "client_id": int(os.getenv("IBKR_CLIENT_ID", 1)),
                }
            
            broker = UnifiedBroker(config)
            await broker.connect()
            print("✓ Broker connected successfully")
            return broker
            
        except Exception as e:
            print(f"✗ Broker connection failed: {e}")
            return None
    
    async def _send_alert(self, status: Dict):
        """Telegram 알림 전송"""
        try:
            import aiohttp
            
            token = os.getenv("TELEGRAM_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
            
            if not token or not chat_id:
                return
            
            final_value = status.get("final_balance", {}).get("total_value", 0)
            positions = status.get("positions_before", 0)
            
            message = f"""
🚨 <b>ARES-7 v64 EMERGENCY STOP</b>

📅 Time: {status['timestamp']}
📝 Reason: {status['reason']}
👤 Operator: {status['operator']}

✅ Actions Completed:
• Orders cancelled
• {positions} positions flattened
• System halted

💰 Final NAV: ${final_value:,.2f}

⚠️ Manual restart required
            """
            
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                }) as response:
                    if response.status == 200:
                        print("✓ Emergency notification sent")
                    else:
                        print("✗ Failed to send notification")
                        
        except Exception as e:
            print(f"✗ Notification error: {e}")
    
    def _save_status(self, status: Dict):
        """상태 저장"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
            print(f"✓ Status saved to {self.status_file}")
        except Exception as e:
            print(f"✗ Failed to save status: {e}")

async def verify_and_execute():
    """확인 후 실행"""
    print("\n" + "="*60)
    print("⚠️  EMERGENCY STOP WARNING ⚠️")
    print("="*60)
    print("\nThis will:")
    print("1. Activate Kill Switch (FLATTEN mode)")
    print("2. Cancel ALL pending orders")
    print("3. Close ALL positions immediately")
    print("4. Stop the trading system")
    print("5. Require manual restart to resume")
    print("\nThis action CANNOT be undone!")
    print("="*60)
    
    confirm = input("\nType 'CONFIRM' to proceed (or press Ctrl+C to cancel): ")
    
    if confirm.upper() == "CONFIRM":
        print("\nSecond confirmation required.")
        reason = input("Enter reason for emergency stop: ").strip()
        
        if not reason:
            reason = "Manual emergency stop - no reason provided"
        
        confirm2 = input(f"\nReason: '{reason}'\nType 'EXECUTE' to proceed: ")
        
        if confirm2.upper() == "EXECUTE":
            emergency = EmergencyStop()
            status = await emergency.execute_emergency_stop(reason)
            return status
        else:
            print("\nEmergency stop cancelled")
            return None
    else:
        print("\nEmergency stop cancelled")
        return None

def check_system_status():
    """시스템 상태 확인"""
    print("\n" + "="*60)
    print("SYSTEM STATUS CHECK")
    print("="*60)
    
    # Kill Switch 상태
    try:
        kill_switch_file = Path("/opt/ares7/runtime/kill_switch.json")
        if kill_switch_file.exists():
            with open(kill_switch_file, 'r') as f:
                ks_state = json.load(f)
                if ks_state.get("tripped"):
                    print(f"⚠ Kill Switch: TRIPPED ({ks_state.get('mode', 'UNKNOWN')})")
                    print(f"  Reason: {ks_state.get('reason', 'N/A')}")
                    print(f"  Time: {ks_state.get('timestamp', 'N/A')}")
                else:
                    print("✓ Kill Switch: Normal")
        else:
            print("✓ Kill Switch: Normal (no state file)")
    except Exception as e:
        print(f"✗ Kill Switch: Unable to check ({e})")
    
    # Process 상태
    try:
        result = os.popen("supervisorctl status ares7 2>/dev/null").read()
        if "RUNNING" in result:
            print("✓ Process: Running")
        elif "STOPPED" in result:
            print("⚠ Process: Stopped")
        else:
            print("✗ Process: Unknown")
    except:
        print("✗ Process: Unable to check")
    
    # 최근 emergency stop
    try:
        status_file = Path("/opt/ares7/logs/emergency_stop.json")
        if status_file.exists():
            with open(status_file, 'r') as f:
                last_stop = json.load(f)
                print(f"\n📍 Last Emergency Stop:")
                print(f"  Time: {last_stop.get('timestamp', 'N/A')}")
                print(f"  Reason: {last_stop.get('reason', 'N/A')}")
    except:
        pass
    
    print("="*60)

if __name__ == "__main__":
    import sys
    
    # 명령줄 인자 처리
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            check_system_status()
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  emergency_stop.py         # Execute emergency stop")
            print("  emergency_stop.py --status # Check system status")
            print("  emergency_stop.py --help   # Show this help")
            sys.exit(0)
    
    # 긴급 정지 실행
    try:
        status = asyncio.run(verify_and_execute())
        if status:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nEmergency stop cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(2)
