"""
ARES-7 v73 Kill Switch
Singleton pattern으로 전역 긴급 정지 스위치 구현
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class KillSwitchState(Enum):
    """Kill Switch 상태"""
    NORMAL = "NORMAL"           # 정상 운영
    WARNING = "WARNING"         # 경고 (계속 운영하되 주의)
    EMERGENCY = "EMERGENCY"     # 긴급 (신규 진입 중지)
    FLATTEN = "FLATTEN"         # 청산 모드 (모든 포지션 청산)
    HALT = "HALT"              # 완전 정지 (모든 활동 중단)


class KillSwitch:
    """
    전역 Kill Switch (Singleton)
    
    사용법:
        kill_switch = KillSwitch()
        
        # 상태 확인
        if kill_switch.is_tripped():
            # 긴급 모드 처리
            pass
        
        # 스위치 작동
        kill_switch.trip("EMERGENCY", "Drawdown exceeded", "risk_manager")
        
        # 스위치 해제
        kill_switch.reset("operator")
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, state_file: Optional[Path] = None):
        # Singleton이므로 한 번만 초기화
        if self._initialized:
            return
        
        self.state_file = state_file or Path("/tmp/ares7_kill_switch.json")
        self.state = KillSwitchState.NORMAL
        self.reason = ""
        self.triggered_by = ""
        self.triggered_at = None
        
        # 상태 파일에서 복원
        self._load_state()
        
        self._initialized = True
        logger.info(f"[KillSwitch] Initialized (state={self.state.value})")
    
    def _load_state(self):
        """상태 파일에서 복원"""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            
            self.state = KillSwitchState(data.get("state", "NORMAL"))
            self.reason = data.get("reason", "")
            self.triggered_by = data.get("triggered_by", "")
            self.triggered_at = data.get("triggered_at")
            
            logger.info(f"[KillSwitch] State loaded: {self.state.value}")
            
        except Exception as e:
            logger.error(f"[KillSwitch] Failed to load state: {e}")
    
    def _save_state(self):
        """상태 파일에 저장"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "state": self.state.value,
                "reason": self.reason,
                "triggered_by": self.triggered_by,
                "triggered_at": self.triggered_at,
                "updated_at": datetime.now().isoformat()
            }
            
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"[KillSwitch] State saved: {self.state.value}")
            
        except Exception as e:
            logger.error(f"[KillSwitch] Failed to save state: {e}")
    
    def trip(self, state: str, reason: str, triggered_by: str):
        """
        Kill Switch 작동
        
        Args:
            state: 상태 ("WARNING", "EMERGENCY", "FLATTEN", "HALT")
            reason: 작동 사유
            triggered_by: 작동 주체 (예: "risk_manager", "operator")
        """
        try:
            new_state = KillSwitchState(state.upper())
        except ValueError:
            logger.error(f"[KillSwitch] Invalid state: {state}")
            return
        
        self.state = new_state
        self.reason = reason
        self.triggered_by = triggered_by
        self.triggered_at = datetime.now().isoformat()
        
        self._save_state()
        
        logger.critical(
            f"[KillSwitch] TRIPPED → {self.state.value} | "
            f"Reason: {reason} | By: {triggered_by}"
        )
    
    def reset(self, operator: str):
        """
        Kill Switch 해제 (NORMAL 상태로 복귀)
        
        Args:
            operator: 해제 주체
        """
        prev_state = self.state
        
        self.state = KillSwitchState.NORMAL
        self.reason = f"Reset by {operator}"
        self.triggered_by = operator
        self.triggered_at = datetime.now().isoformat()
        
        self._save_state()
        
        logger.warning(
            f"[KillSwitch] RESET: {prev_state.value} → NORMAL | By: {operator}"
        )
    
    def is_tripped(self) -> bool:
        """긴급 상태 여부 확인"""
        return self.state in [
            KillSwitchState.EMERGENCY,
            KillSwitchState.FLATTEN,
            KillSwitchState.HALT
        ]
    
    def is_halted(self) -> bool:
        """완전 정지 상태 여부"""
        return self.state == KillSwitchState.HALT
    
    def should_flatten(self) -> bool:
        """청산 모드 여부"""
        return self.state in [KillSwitchState.FLATTEN, KillSwitchState.HALT]
    
    def can_trade(self) -> bool:
        """거래 가능 여부"""
        return self.state in [KillSwitchState.NORMAL, KillSwitchState.WARNING]
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "state": self.state.value,
            "is_tripped": self.is_tripped(),
            "can_trade": self.can_trade(),
            "should_flatten": self.should_flatten(),
            "reason": self.reason,
            "triggered_by": self.triggered_by,
            "triggered_at": self.triggered_at
        }


# 전역 인스턴스
_global_kill_switch = None


def get_kill_switch() -> KillSwitch:
    """전역 Kill Switch 인스턴스 반환"""
    global _global_kill_switch
    if _global_kill_switch is None:
        _global_kill_switch = KillSwitch()
    return _global_kill_switch
