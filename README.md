# ARES-7 v73 FULL

**Production-Ready Algorithmic Trading System**

ARES-7 v73은 고급 전략 엔진(Phoenix, Momentum, Mean Reversion, Meta Ensemble)과 실거래 브로커 레이어(KIS, IBKR)를 완전히 통합한 알고리즘 트레이딩 시스템입니다.

---

## 🎯 주요 기능

### ✅ v73 전략 엔진 (Part1~9)
- **Phoenix 10D Engine**: Overnight momentum, option flow (Vanna/Charm), liquidity fragmentation
- **Momentum Engine**: Transformer 기반 모멘텀 분석
- **Mean Reversion Engine**: NSR volatility 기반 평균회귀 전략
- **Meta Ensemble**: CatBoost + TabNet 기반 앙상블 의사결정
- **Risk Manager**: VPIN, GEX 필터, 동적 포지션 사이징
- **Alpha Pipeline**: GEX/DIX/WhisperZ/LLM 알파 통합
- **Execution Engine**: IRL 기반 최적 실행
- **Monitoring Engine**: 실시간 PnL/equity 추적

### ✅ v64 브로커 레이어
- **UnifiedBroker**: KIS + IBKR 통합 인터페이스
- **KIS Broker**: 한국투자증권 (mojito2) - 미국/한국 주식
- **IBKR Broker**: Interactive Brokers (ib_insync)
- **OrderGenerator**: 시그널 → 주문 변환
- **KillSwitch**: 긴급 정지 스위치 (Singleton)
- **EmergencyStop**: 전체 시스템 긴급 종료

### ✅ 통합 기능
- **백테스트 모드**: 가상 실행 (ExecutionEngine)
- **페이퍼 트레이딩**: 모의투자 계좌 (KIS VPS)
- **실거래 모드**: 실제 브로커 API 호출
- **통합 Orchestrator**: 모드 자동 전환

---

## 📁 프로젝트 구조

```
ares7_v73_full/
├── engines/              # v73 전략 엔진
│   ├── phoenix/          # Phoenix 10D Engine
│   ├── momentum/         # Momentum Engine
│   ├── mean_reversion/   # Mean Reversion Engine
│   └── execution/        # Execution Engine (IRL)
├── meta/                 # Meta Ensemble Engine
├── risk/                 # Risk Manager
├── data/                 # Alpha Pipeline
│   └── pipelines/
├── monitoring/           # Monitoring Engine
├── orchestrator/         # Main Orchestrator
│   └── ares_orchestrator_integrated.py
├── brokers/              # v64 브로커 레이어
│   ├── unified_broker.py # UnifiedBroker
│   ├── kis_broker.py     # KIS Broker
│   ├── ibkr_broker.py    # IBKR Broker
│   └── kis_auth.py       # KIS 인증
├── governance/           # 거버넌스 레이어
│   ├── kill_switch.py    # KillSwitch
│   ├── order_generator.py # OrderGenerator
│   └── emergency_stop.py # EmergencyStop
├── llm_alpha/            # LLM Alpha 통합
├── utils/                # 유틸리티
├── config/               # 설정 파일
├── logs/                 # 로그 파일
├── tests/                # 테스트
├── main.py               # 메인 진입점
├── requirements.txt      # Python 의존성
├── .env.template         # 환경 변수 템플릿
└── README.md             # 이 파일
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.template .env
# .env 파일을 열어 실제 API 키 입력
nano .env
```

### 2. 백테스트 실행

```bash
python main.py \
  --mode backtest \
  --symbols SPY,QQQ,IWM \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --capital 1000000
```

### 3. 페이퍼 트레이딩 (모의투자)

```bash
python main.py \
  --mode paper \
  --symbols SPY,QQQ \
  --capital 1000000
```

### 4. 실거래 실행

```bash
python main.py \
  --mode live \
  --symbols SPY,QQQ \
  --capital 1000000
```

---

## ⚙️ 환경 변수 설정

`.env` 파일에서 다음 항목을 설정하세요:

### 필수 설정

```bash
# KIS (한국투자증권)
KIS_APP_KEY_REAL=your_real_app_key
KIS_APP_SECRET_REAL=your_real_app_secret
KIS_ACCOUNT_REAL=12345678-01

# KIS 모의투자
KIS_APP_KEY_VPS=your_vps_app_key
KIS_APP_SECRET_VPS=your_vps_app_secret
KIS_ACCOUNT_VPS=12345678-01

# 거래 설정
KIS_MARKET=US           # US or KR
KIS_EXCHANGE=NASD       # NASD, NYSE, AMEX
```

### 선택 설정

```bash
# IBKR (Interactive Brokers)
IBKR_ENABLED=false
IBKR_HOST=127.0.0.1
IBKR_PORT=7497

# 알파 데이터
SQUEEZEMETRICS_API_KEY=your_key  # DIX/GEX
WHISPERZ_API_KEY=your_key

# LLM
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# 알림
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## 🛡️ 안전 기능

### KillSwitch

긴급 상황 시 자동으로 거래를 중단합니다.

```python
from governance.kill_switch import get_kill_switch

kill_switch = get_kill_switch()

# 긴급 정지
kill_switch.trip("EMERGENCY", "Drawdown exceeded 15%", "risk_manager")

# 상태 확인
if kill_switch.is_tripped():
    # 긴급 모드 처리
    pass

# 해제
kill_switch.reset("operator")
```

### EmergencyStop

전체 시스템을 긴급 종료합니다.

```bash
python -m governance.emergency_stop --reason "Market anomaly"
```

---

## 📊 모니터링

### Prometheus

```bash
# Prometheus 메트릭 엔드포인트
http://localhost:9090/metrics
```

### 로그

```bash
# 로그 파일 위치
logs/ares7_backtest_20241118_120000.log
logs/ares7_live_20241118_120000.log
```

---

## 🧪 테스트

```bash
# 단위 테스트
pytest tests/

# 커버리지 포함
pytest --cov=. tests/

# 특정 모듈 테스트
pytest tests/test_orchestrator.py
```

---

## 📈 성능 최적화

### 백테스트 속도 향상

- 병렬 처리: 여러 심볼을 동시에 처리
- 데이터 캐싱: Redis 사용
- 벡터화: NumPy/Pandas 연산 최적화

### 실거래 지연 최소화

- 비동기 I/O: aiohttp, asyncio 사용
- WebSocket: 실시간 시장 데이터
- 로컬 캐싱: 빈번한 API 호출 방지

---

## 🔧 문제 해결

### 브로커 연결 실패

```bash
# KIS 토큰 갱신
# .env 파일의 API 키 확인
# 네트워크 연결 확인
```

### 주문 실행 실패

```bash
# 계좌 잔고 확인
# 주문 가능 시간 확인 (장 시간)
# 최소 주문 금액 확인 (MIN_ORDER_VALUE)
```

### KillSwitch 작동

```bash
# 상태 파일 확인
cat /tmp/ares7_kill_switch.json

# 수동 해제
python -c "from governance.kill_switch import get_kill_switch; get_kill_switch().reset('operator')"
```

---

## 📚 추가 문서

- **API 문서**: `docs/api.md`
- **전략 설명**: `docs/strategies.md`
- **배포 가이드**: `docs/deployment.md`
- **FAQ**: `docs/faq.md`

---

## 🤝 기여

이 프로젝트는 비공개 프로젝트입니다.

---

## 📄 라이선스

Proprietary - All Rights Reserved

---

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로 제공됩니다. 실제 거래에 사용할 경우 발생하는 모든 손실에 대해 개발자는 책임을 지지 않습니다. 자신의 책임 하에 사용하십시오.

---

## 📞 지원

문의사항이 있으시면 프로젝트 관리자에게 연락하십시오.

---

**ARES-7 v73 FULL** - Built with ❤️ for algorithmic traders
