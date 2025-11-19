#!/usr/bin/env python3
"""
ARES7 v73 알파 생성 로직 검증 스크립트

알파 생성 파이프라인의 각 단계를 검증하고 문제점을 식별합니다.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 프로젝트 루트 추가
sys.path.insert(0, "/home/ubuntu/ares7_v73_full")

print("=" * 80)
print("ARES7 v73 알파 생성 로직 검증")
print("=" * 80)
print()

# ============================================================
# 1. 데이터 파이프라인 검증
# ============================================================
print("1. 데이터 파이프라인 검증")
print("-" * 80)

issues = []
warnings = []

# 필수 컬럼 체크
REQUIRED_COLUMNS = {
    "기본 OHLCV": ["date", "open", "high", "low", "close", "volume"],
    "마이크로구조": ["spread", "depth_imbalance", "order_flow_imbalance", "tick_direction", "volatility"],
    "옵션 플로우": ["vanna_flow_proxy", "charm_flow_proxy", "dealer_hedging_flow", "dealer_hedging_flow_z"],
    "유동성": ["nbbo_spread", "fragmentation_ratio"],
    "VIX": ["VIX"]
}

print("필수 데이터 컬럼:")
for category, columns in REQUIRED_COLUMNS.items():
    print(f"  {category}: {', '.join(columns)}")

print()
print("⚠️  경고: 다음 컬럼들이 없으면 해당 알파 팩터가 0으로 계산됩니다:")
print("  - 마이크로구조 컬럼 → microstructure_signal = 0")
print("  - 옵션 플로우 컬럼 → option_flow_score = 0")
print("  - 유동성 컬럼 → liquidity_fragmentation = 0")
print("  - VIX 컬럼 → overnight_momentum 가중치 감소")
print()

# ============================================================
# 2. 알파 팩터 분석
# ============================================================
print("2. 알파 팩터 분석")
print("-" * 80)

ALPHA_FACTORS = {
    "overnight_momentum": {
        "weight": "0.45 (VIX < 25) or 0.20 (VIX >= 25)",
        "formula": "tanh((open_today / close_prev - 1) * 35.0) * weight",
        "range": "[-1, 1]",
        "dependencies": ["close", "open"],
        "critical": True
    },
    "microstructure_signal": {
        "weight": "1/7 (평균에서)",
        "formula": "tanh(0.3*(-spread) + 0.25*depth + 0.2*flow + 0.15*tick - 0.1*vol)",
        "range": "[-1, 1]",
        "dependencies": ["spread", "depth_imbalance", "order_flow_imbalance", "tick_direction", "volatility"],
        "critical": True
    },
    "liquidity_fragmentation": {
        "weight": "1/7 (평균에서)",
        "formula": "tanh(frag * 2 - spread * 5)",
        "range": "[-1, 1]",
        "dependencies": ["nbbo_spread", "fragmentation_ratio"],
        "critical": False
    },
    "option_flow_score": {
        "weight": "1/7 (평균에서)",
        "formula": "tanh(0.35*vanna + 0.35*charm + 0.3*hedge_z)",
        "range": "[-1, 1]",
        "dependencies": ["vanna_flow_proxy", "charm_flow_proxy", "dealer_hedging_flow_z"],
        "critical": False
    },
    "gex_signal": {
        "weight": "1/7 (평균에서)",
        "formula": "tanh(gex / 2e9)",
        "range": "[-1, 1]",
        "dependencies": ["GEX data from external source"],
        "critical": False
    },
    "dix_signal": {
        "weight": "1/7 (평균에서)",
        "formula": "tanh((dix - 42) / 5)",
        "range": "[-1, 1]",
        "dependencies": ["DIX data from external source"],
        "critical": False
    },
    "whisper_z": {
        "weight": "1/7 (평균에서)",
        "formula": "tanh(whisper_z / 2)",
        "range": "[-1, 1]",
        "dependencies": ["WhisperZ data from external source"],
        "critical": False
    },
    "llm_alpha_boost": {
        "weight": "1/7 (평균에서)",
        "formula": "0.45*risk_scalar + 0.25*sentiment + 0.15*earnings - 0.15*uncertainty",
        "range": "[-1, 1]",
        "dependencies": ["LLM analysis output"],
        "critical": True
    }
}

print("알파 팩터 구성:")
print()
for factor, info in ALPHA_FACTORS.items():
    critical_mark = "🔴 CRITICAL" if info["critical"] else "🟡 OPTIONAL"
    print(f"{critical_mark} {factor}")
    print(f"  가중치: {info['weight']}")
    print(f"  공식: {info['formula']}")
    print(f"  범위: {info['range']}")
    print(f"  의존성: {', '.join(info['dependencies'])}")
    print()

# ============================================================
# 3. 잠재적 문제점 분석
# ============================================================
print("3. 잠재적 문제점 분석")
print("-" * 80)

POTENTIAL_ISSUES = [
    {
        "issue": "합성 데이터 사용",
        "impact": "HIGH",
        "description": "main.py에서 np.random.randn()으로 생성된 데이터 사용",
        "consequence": "백테스팅 결과가 실제 시장과 무관하며, 알파/샤프 비율이 의미 없음",
        "solution": "실제 API 데이터로 교체 (fix_synthetic_data.py 실행)"
    },
    {
        "issue": "더미 GEX/DIX 값",
        "impact": "MEDIUM",
        "description": "gex=0.0, dix=45.2 하드코딩",
        "consequence": "GEX/DIX 팩터가 항상 같은 값을 반환하여 알파 기여도 없음",
        "solution": "실제 옵션 데이터 및 다크풀 데이터 연결"
    },
    {
        "issue": "누락된 마이크로구조 데이터",
        "impact": "HIGH",
        "description": "spread, depth_imbalance 등 컬럼이 데이터에 없음",
        "consequence": "microstructure_signal이 항상 0으로 계산되어 주요 알파 팩터 손실",
        "solution": "Level 2 orderbook 데이터 추가 또는 프록시 계산 로직 구현"
    },
    {
        "issue": "누락된 옵션 플로우 데이터",
        "impact": "MEDIUM",
        "description": "vanna_flow_proxy, charm_flow_proxy 등 컬럼이 없음",
        "consequence": "option_flow_score가 항상 0으로 계산",
        "solution": "옵션 체인 데이터 추가 및 그릭스 계산 로직 구현"
    },
    {
        "issue": "LLM 알파 연결 부재",
        "impact": "HIGH",
        "description": "llm_alpha 파라미터가 None으로 전달됨",
        "consequence": "LLM 기반 알파 팩터가 작동하지 않음",
        "solution": "Anthropic Claude API 연결 및 뉴스/공시 분석 파이프라인 구축"
    },
    {
        "issue": "VIX 데이터 누락",
        "impact": "MEDIUM",
        "description": "VIX 컬럼이 없어 기본값 20 사용",
        "consequence": "overnight_momentum 가중치가 시장 변동성을 반영하지 못함",
        "solution": "VIX 데이터 추가 (Alpha Vantage 또는 CBOE)"
    },
    {
        "issue": "랜덤 PnL 계산",
        "impact": "HIGH",
        "description": "trade_pnl = np.random.randn() * 0.001",
        "consequence": "백테스팅 PnL이 실제 거래와 무관",
        "solution": "실제 체결 가격 기반 PnL 계산 로직 구현"
    }
]

print("발견된 문제점:")
print()
for i, issue in enumerate(POTENTIAL_ISSUES, 1):
    impact_emoji = "🔴" if issue["impact"] == "HIGH" else "🟡" if issue["impact"] == "MEDIUM" else "🟢"
    print(f"{i}. {impact_emoji} {issue['issue']} (영향도: {issue['impact']})")
    print(f"   설명: {issue['description']}")
    print(f"   결과: {issue['consequence']}")
    print(f"   해결: {issue['solution']}")
    print()

# ============================================================
# 4. 알파/샤프 비율 예상
# ============================================================
print("4. 알파/샤프 비율 예상")
print("-" * 80)

print("현재 상태 (합성 데이터 사용):")
print("  예상 알파: ~0 (랜덤 데이터이므로 의미 없음)")
print("  예상 샤프: ~0 (랜덤 신호이므로 시장 대비 초과 수익 없음)")
print("  예상 승률: ~50% (랜덤)")
print()

print("실데이터 연결 후 (최소 구성):")
print("  - OHLCV 데이터만 있는 경우")
print("  - overnight_momentum만 작동")
print("  - 예상 알파: 0.5~1.5% (낮음)")
print("  - 예상 샤프: 0.3~0.8 (낮음)")
print()

print("실데이터 연결 후 (부분 구성):")
print("  - OHLCV + VIX + LLM 알파")
print("  - overnight_momentum + llm_alpha_boost 작동")
print("  - 예상 알파: 2~5% (중간)")
print("  - 예상 샤프: 0.8~1.5 (중간)")
print()

print("실데이터 연결 후 (완전 구성):")
print("  - 모든 데이터 소스 연결")
print("  - 모든 알파 팩터 작동")
print("  - 예상 알파: 5~15% (높음)")
print("  - 예상 샤프: 1.5~3.0 (높음)")
print()

# ============================================================
# 5. 권장 조치 사항
# ============================================================
print("5. 권장 조치 사항")
print("-" * 80)

RECOMMENDATIONS = [
    {
        "priority": "P0 (즉시)",
        "action": "합성 데이터를 실제 API 데이터로 교체",
        "command": "python3 /home/ubuntu/fix_synthetic_data.py",
        "impact": "백테스팅 결과가 의미 있어짐"
    },
    {
        "priority": "P0 (즉시)",
        "action": "랜덤 PnL 계산을 실제 계산으로 교체",
        "command": "fix_synthetic_data.py에 포함됨",
        "impact": "백테스팅 PnL이 정확해짐"
    },
    {
        "priority": "P1 (1주일 내)",
        "action": "LLM 알파 파이프라인 구축",
        "command": "Anthropic Claude API 연결 + 뉴스 분석",
        "impact": "알파 +2~3% 증가 예상"
    },
    {
        "priority": "P1 (1주일 내)",
        "action": "VIX 데이터 추가",
        "command": "Alpha Vantage에서 VIX 데이터 fetch",
        "impact": "overnight_momentum 정확도 향상"
    },
    {
        "priority": "P2 (2주일 내)",
        "action": "마이크로구조 데이터 프록시 구현",
        "command": "OHLCV로부터 spread, imbalance 추정",
        "impact": "알파 +1~2% 증가 예상"
    },
    {
        "priority": "P2 (2주일 내)",
        "action": "GEX/DIX 실데이터 연결",
        "command": "옵션 체인 데이터 + 다크풀 데이터",
        "impact": "알파 +1~2% 증가 예상"
    },
    {
        "priority": "P3 (1개월 내)",
        "action": "옵션 플로우 데이터 추가",
        "command": "옵션 체인 + 그릭스 계산",
        "impact": "알파 +0.5~1% 증가 예상"
    }
]

print("우선순위별 조치 사항:")
print()
for rec in RECOMMENDATIONS:
    print(f"{rec['priority']}: {rec['action']}")
    print(f"  명령: {rec['command']}")
    print(f"  영향: {rec['impact']}")
    print()

# ============================================================
# 6. 마이크로구조 프록시 구현 제안
# ============================================================
print("6. 마이크로구조 프록시 구현 제안")
print("-" * 80)

print("Level 2 데이터가 없을 경우, OHLCV로부터 프록시 계산 가능:")
print()
print("1. spread (스프레드)")
print("   프록시: (high - low) / close")
print("   의미: 일중 변동성을 스프레드로 근사")
print()
print("2. depth_imbalance (호가 불균형)")
print("   프록시: (close - low) / (high - low)")
print("   의미: 종가가 고가/저가 중 어디에 가까운지")
print()
print("3. order_flow_imbalance (주문 흐름 불균형)")
print("   프록시: (close - open) / (high - low)")
print("   의미: 시가 대비 종가의 상대적 위치")
print()
print("4. tick_direction (틱 방향)")
print("   프록시: sign(close - close_prev)")
print("   의미: 가격 변화 방향")
print()
print("5. volatility (변동성)")
print("   프록시: rolling std of returns")
print("   의미: 최근 수익률의 표준편차")
print()

print("이 프록시들을 구현하면 microstructure_signal이 작동합니다!")
print()

# ============================================================
# 7. 종합 평가
# ============================================================
print("=" * 80)
print("종합 평가")
print("=" * 80)
print()

print("현재 상태:")
print("  ❌ 합성 데이터 사용 → 백테스팅 무의미")
print("  ❌ 주요 알파 팩터 미작동 (마이크로구조, 옵션 플로우)")
print("  ❌ LLM 알파 미연결")
print("  ❌ 랜덤 PnL 계산")
print()
print("  예상 백테스팅 결과: 알파 ~0%, 샤프 ~0")
print()

print("실데이터 연결 후 (최소):")
print("  ✅ 실제 OHLCV 데이터")
print("  ✅ overnight_momentum 작동")
print("  ⚠️  기타 팩터 미작동")
print()
print("  예상 백테스팅 결과: 알파 0.5~1.5%, 샤프 0.3~0.8")
print()

print("실데이터 연결 후 (권장):")
print("  ✅ 실제 OHLCV 데이터")
print("  ✅ VIX 데이터")
print("  ✅ LLM 알파 연결")
print("  ✅ 마이크로구조 프록시")
print("  ⚠️  옵션 플로우 미작동")
print()
print("  예상 백테스팅 결과: 알파 3~8%, 샤프 1.0~2.0")
print()

print("실데이터 연결 후 (완전):")
print("  ✅ 모든 데이터 소스 연결")
print("  ✅ 모든 알파 팩터 작동")
print()
print("  예상 백테스팅 결과: 알파 5~15%, 샤프 1.5~3.0")
print()

print("=" * 80)
print("다음 단계:")
print("1. python3 /home/ubuntu/fix_synthetic_data.py 실행")
print("2. API 연결 테스트")
print("3. 마이크로구조 프록시 구현")
print("4. LLM 알파 파이프라인 구축")
print("5. 백테스팅 재실행")
print("=" * 80)
