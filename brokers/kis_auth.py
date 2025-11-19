# src/connectors/kis_auth_v2.py
"""
KIS Open API Authentication Module v2
Based on official open-trading-api structure
"""

from __future__ import annotations

import os
import time
import json
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Dict, Optional
from datetime import datetime, timedelta

import requests
import yaml

logger = logging.getLogger(__name__)


@dataclass
class KisEnvConfig:
    app_key: str
    app_secret: str
    acct_no: str          # 앞 8자리
    prod_code: str        # 뒤 2자리 (01 등)
    base_url: str         # openapi.koreainvestment.com or openapivts.koreainvestment.com
    user_agent: str


class KisAuthV2:
    """
    공식 open-trading-api의 kis_devlp.yaml 구조를 따르는 인증 래퍼.
    - 접근 토큰 24h 캐시
    - prod / vps 자동 전환
    - WebSocket 인증키 지원
    - 해시키 생성 지원 (정정/취소 API용)
    """

    def __init__(self, svr: Literal["prod", "vps"] = "vps", config_root: str | None = None):
        # kis_devlp.yaml 위치
        root = Path(config_root) if config_root else Path(__file__).resolve().parents[2]
        self.kis_cfg_path = root / "kis_devlp.yaml"

        # .env 파일에서도 읽을 수 있도록
        if not self.kis_cfg_path.exists():
            logger.info("[KIS] kis_devlp.yaml not found, using ENV variables")
            self.raw_cfg = self._load_from_env()
        else:
            self.raw_cfg = yaml.safe_load(self.kis_cfg_path.read_text(encoding="utf-8"))

        self.svr = svr  # "prod" or "vps"

        if svr == "prod":
            app_key = self.raw_cfg.get("my_app", os.getenv("KIS_APP_KEY_REAL"))
            app_secret = self.raw_cfg.get("my_sec", os.getenv("KIS_APP_SECRET_REAL"))
            acct_no = self.raw_cfg.get("my_acct_stock", os.getenv("KIS_ACCOUNT_REAL", "")[:8])
            base_url = "https://openapi.koreainvestment.com:9443"
        else:
            app_key = self.raw_cfg.get("paper_app", os.getenv("KIS_APP_KEY_PAPER"))
            app_secret = self.raw_cfg.get("paper_sec", os.getenv("KIS_APP_SECRET_PAPER"))
            acct_no = self.raw_cfg.get("my_paper_stock", os.getenv("KIS_ACCOUNT_PAPER", "")[:8])
            base_url = "https://openapivts.koreainvestment.com:29443"

        prod_code = self.raw_cfg.get("my_prod", os.getenv("KIS_ACCOUNT_PROD_CODE", "01"))
        user_agent = self.raw_cfg.get("my_agent", 
                                     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        self.env_cfg = KisEnvConfig(
            app_key=app_key,
            app_secret=app_secret,
            acct_no=acct_no,
            prod_code=prod_code,
            base_url=base_url,
            user_agent=user_agent,
        )

        # 토큰 캐시
        cache_dir = Path("/opt/ares7/runtime/kis_tokens")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.token_cache = cache_dir / f"KIS_TOKEN_{svr}.json"

        self._access_token: Optional[str] = None
        self._token_expire_at: float = 0.0
        self._websocket_key: Optional[str] = None

    # ---------- 내부 헬퍼 ----------

    def _load_from_env(self) -> Dict:
        """환경변수에서 설정 로드"""
        return {
            "my_app": os.getenv("KIS_APP_KEY_REAL", ""),
            "my_sec": os.getenv("KIS_APP_SECRET_REAL", ""),
            "paper_app": os.getenv("KIS_APP_KEY_PAPER", ""),
            "paper_sec": os.getenv("KIS_APP_SECRET_PAPER", ""),
            "my_acct_stock": os.getenv("KIS_ACCOUNT_REAL", "")[:8],
            "my_paper_stock": os.getenv("KIS_ACCOUNT_PAPER", "")[:8],
            "my_prod": os.getenv("KIS_ACCOUNT_PROD_CODE", "01"),
        }

    def _load_token_cache(self) -> None:
        if not self.token_cache.exists():
            return
        try:
            data = json.loads(self.token_cache.read_text(encoding="utf-8"))
            if data.get("svr") != self.svr:
                return
            self._access_token = data["access_token"]
            self._token_expire_at = data["expire_at"]
            self._websocket_key = data.get("websocket_key")
        except Exception as e:
            logger.warning(f"[KIS] token cache load 실패: {e}")

    def _save_token_cache(self) -> None:
        if not self._access_token:
            return
        data = {
            "svr": self.svr,
            "access_token": self._access_token,
            "expire_at": self._token_expire_at,
            "websocket_key": self._websocket_key,
            "updated": datetime.now().isoformat()
        }
        self.token_cache.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ---------- 퍼블릭 메서드 ----------

    def get_access_token(self, force_refresh: bool = False) -> str:
        """접근 토큰 획득"""
        now = time.time()

        if not force_refresh:
            # 캐시 로드
            if not self._access_token:
                self._load_token_cache()
            if self._access_token and now < self._token_expire_at - 60:
                return self._access_token

        # 토큰 갱신
        url = f"{self.env_cfg.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.env_cfg.app_key,
            "appsecret": self.env_cfg.app_secret,
        }
        
        try:
            r = requests.post(url, json=body, headers=headers, timeout=5)
            r.raise_for_status()
            data = r.json()
            
            if data.get("access_token"):
                self._access_token = data["access_token"]
                # 공식 문서상 24시간 → 23시간 여유
                self._token_expire_at = now + 60 * 60 * 23
                self._save_token_cache()
                logger.info(f"[KIS] access_token 발급 성공 ({self.svr})")
            else:
                raise Exception(f"Token response error: {data}")
                
        except Exception as e:
            logger.error(f"[KIS] 토큰 발급 실패: {e}")
            raise
            
        return self._access_token

    def get_websocket_key(self) -> str:
        """WebSocket 인증키 획득"""
        if self._websocket_key:
            return self._websocket_key
            
        url = f"{self.env_cfg.base_url}/oauth2/Approval"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.env_cfg.app_key,
            "secretkey": self.env_cfg.app_secret,
        }
        
        try:
            r = requests.post(url, json=body, headers=headers, timeout=5)
            r.raise_for_status()
            data = r.json()
            self._websocket_key = data["approval_key"]
            self._save_token_cache()
            logger.info("[KIS] WebSocket approval_key 발급 성공")
            return self._websocket_key
        except Exception as e:
            logger.error(f"[KIS] WebSocket key 발급 실패: {e}")
            raise

    def make_headers(self, tr_id: str, custtype: str = "P", hashkey: str = None) -> Dict[str, str]:
        """
        공통 header 생성
        custtype: 개인 P / 법인 B
        hashkey: POST body의 해시값 (정정/취소 API 필요)
        """
        token = self.get_access_token()
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.env_cfg.app_key,
            "appsecret": self.env_cfg.app_secret,
            "tr_id": tr_id,
            "custtype": custtype,
            "User-Agent": self.env_cfg.user_agent,
        }
        
        if hashkey:
            headers["hashkey"] = hashkey
            
        return headers

    def get_hashkey(self, body: dict) -> str:
        """
        POST body의 해시키 생성 (정정/취소 API용)
        """
        url = f"{self.env_cfg.base_url}/uapi/hashkey"
        headers = {
            "content-type": "application/json",
            "appkey": self.env_cfg.app_key,
            "appsecret": self.env_cfg.app_secret,
        }
        
        try:
            r = requests.post(url, json=body, headers=headers, timeout=5)
            r.raise_for_status()
            data = r.json()
            return data["HASH"]
        except Exception as e:
            logger.error(f"[KIS] hashkey 생성 실패: {e}")
            raise

    @property
    def account(self) -> Dict[str, str]:
        """계좌 정보 (CANO, ACNT_PRDT_CD)"""
        return {
            "CANO": self.env_cfg.acct_no,
            "ACNT_PRDT_CD": self.env_cfg.prod_code,
        }

    @property
    def base_url(self) -> str:
        return self.env_cfg.base_url
        
    @property
    def is_production(self) -> bool:
        return self.svr == "prod"
