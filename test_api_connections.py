#!/usr/bin/env python3
"""
ARES7 v73 API 연결 테스트 스크립트

모든 데이터 소스 API의 연결 상태를 확인합니다.
"""

import os
import sys
import requests
from datetime import datetime

# .env 파일 로드
def load_env():
    env_path = "/home/ubuntu/ares7_v73_full/.env"
    env_vars = {}
    
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value.strip('"').strip("'")
    
    return env_vars

ENV = load_env()

# API 테스트 함수들
def test_sec_api():
    """SEC API 테스트"""
    api_key = ENV.get('SEC_API_KEY')
    if not api_key or api_key == 'your_sec_api_key':
        return {"status": "SKIP", "reason": "API key not configured"}
    
    try:
        url = f"https://api.sec-api.io/mapping/ticker/AAPL"
        headers = {"Authorization": api_key}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return {"status": "SUCCESS", "data": response.json()}
        else:
            return {"status": "ERROR", "code": response.status_code, "message": response.text}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def test_fred_api():
    """FRED API 테스트"""
    api_key = ENV.get('FRED_API_KEY')
    if not api_key or api_key == 'your_fred_api_key':
        return {"status": "SKIP", "reason": "API key not configured"}
    
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "GDP",
            "api_key": api_key,
            "file_type": "json",
            "limit": 1
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            return {"status": "SUCCESS", "data": response.json()}
        else:
            return {"status": "ERROR", "code": response.status_code, "message": response.text}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def test_alpha_vantage_api():
    """Alpha Vantage API 테스트"""
    api_key = ENV.get('ALPHA_VANTAGE_API_KEY')
    if not api_key or api_key == 'your_alpha_vantage_key':
        return {"status": "SKIP", "reason": "API key not configured"}
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": "AAPL",
            "apikey": api_key
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "Global Quote" in data:
                return {"status": "SUCCESS", "data": data}
            else:
                return {"status": "ERROR", "message": "Invalid response format"}
        else:
            return {"status": "ERROR", "code": response.status_code, "message": response.text}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def test_news_api():
    """News API 테스트"""
    api_key = ENV.get('NEWS_API_KEY')
    if not api_key or api_key == 'your_news_api_key':
        return {"status": "SKIP", "reason": "API key not configured"}
    
    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "country": "us",
            "category": "business",
            "apiKey": api_key,
            "pageSize": 1
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            return {"status": "SUCCESS", "data": response.json()}
        else:
            return {"status": "ERROR", "code": response.status_code, "message": response.text}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def test_polygon_api():
    """Polygon API 테스트"""
    api_key = ENV.get('POLYGON_API_KEY')
    if not api_key or api_key == 'your_polygon_api_key':
        return {"status": "SKIP", "reason": "API key not configured"}
    
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev"
        params = {"apiKey": api_key}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            return {"status": "SUCCESS", "data": response.json()}
        else:
            return {"status": "ERROR", "code": response.status_code, "message": response.text}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def test_anthropic_api():
    """Anthropic Claude API 테스트"""
    api_key = ENV.get('ANTHROPIC_API_KEY')
    if not api_key or api_key == 'your_anthropic_key':
        return {"status": "SKIP", "reason": "API key not configured"}
    
    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hi"}]
        }
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            return {"status": "SUCCESS", "data": response.json()}
        else:
            return {"status": "ERROR", "code": response.status_code, "message": response.text}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

# API 테스트 목록
API_TESTS = [
    ("SEC API", test_sec_api),
    ("FRED API", test_fred_api),
    ("Alpha Vantage API", test_alpha_vantage_api),
    ("News API", test_news_api),
    ("Polygon API", test_polygon_api),
    ("Anthropic Claude API", test_anthropic_api),
]

def run_all_tests():
    """모든 API 테스트 실행"""
    print("=" * 80)
    print("ARES7 v73 API 연결 테스트")
    print("=" * 80)
    print(f"테스트 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    for name, test_func in API_TESTS:
        print(f"테스트 중: {name}...", end=" ")
        sys.stdout.flush()
        
        result = test_func()
        result["name"] = name
        results.append(result)
        
        if result["status"] == "SUCCESS":
            print("✅ 성공")
        elif result["status"] == "SKIP":
            print(f"⏭️  건너뜀 ({result['reason']})")
        else:
            print(f"❌ 실패")
            if "message" in result:
                print(f"   오류: {result['message']}")
    
    print()
    print("=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    skip_count = sum(1 for r in results if r["status"] == "SKIP")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    
    print(f"총 테스트: {len(results)}")
    print(f"성공: {success_count}")
    print(f"건너뜀: {skip_count}")
    print(f"실패: {error_count}")
    print()
    
    if error_count > 0:
        print("실패한 API:")
        for result in results:
            if result["status"] == "ERROR":
                print(f"  - {result['name']}")
                if "message" in result:
                    print(f"    {result['message']}")
        print()
    
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    results = run_all_tests()
    
    # 종료 코드 설정
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    sys.exit(0 if error_count == 0 else 1)
