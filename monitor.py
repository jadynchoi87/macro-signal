#!/usr/bin/env python3
"""
매크로 신호등 모니터
매일 실행하여 역발상 매수 기회 감지 시 텔레그램 알림 발송.

설정: macro-signal/.env 파일에 TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID 필요
"""

import json
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

# macro_signal.py의 분석 로직 재사용
from macro_signal import (
    fetch_all, analyze, macro_score,
    get_backtest_zone, contrarian_opportunity, opportunity_signal,
    BACKTEST_STATS, CATEGORIES,
)

# ============================================================
# 설정
# ============================================================

STATE_FILE = Path(__file__).parent / ".last_state.json"
ENV_FILE = Path(__file__).parent / ".env"

# 알림 기준
ALERT_THRESHOLD = 70      # 매수 기회 점수 이상이면 알림
EXTREME_THRESHOLD = 85     # 극매수 구간
OVERHEAT_THRESHOLD = 20    # 과열 경고 (기회 점수 이하)


def load_env():
    """간단한 .env 파서"""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def send_telegram(token, chat_id, message):
    """텔레그램 메시지 발송"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
    }).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"텔레그램 발송 실패: {e}")
        return False


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def format_alert(macro, opp, opp_sig, bt_zone, results):
    """텔레그램 알림 메시지 포맷"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    if opp >= EXTREME_THRESHOLD:
        header = "🔴🔴🔴 극매수 신호 🔴🔴🔴"
    elif opp >= ALERT_THRESHOLD:
        header = "🟢 매수 기회 감지"
    elif opp <= OVERHEAT_THRESHOLD:
        header = "⚠️ 과열 경고"
    else:
        header = "📊 매크로 신호등"

    # 주요 지표 요약
    indicators = ""
    for r in results:
        icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}[r["signal"]]
        indicators += f"  {icon} {r['label']}: {r['value']} ({r['desc']})\n"

    # RED 지표 = 역발상 매수 기회
    red_items = [r for r in results if r["signal"] == "RED"]
    red_summary = ""
    if red_items:
        red_summary = "\n<b>역발상 매수 기회 요인:</b>\n"
        for r in red_items:
            red_summary += f"  📍 {r['label']}: {r['value']}\n"

    msg = f"""<b>{header}</b>
{now}

<b>역발상 매수 기회: {opp:.0f}/100</b>
매크로 환경: {macro:.0f}/100 ({bt_zone['label']})

<b>백테스트 근거 (3개월 후):</b>
  평균 수익률: {bt_zone['avg']:+.1f}%
  승률: {bt_zone['win']:.0f}% (표본 {bt_zone['n']}회)
{red_summary}
<b>전체 지표:</b>
{indicators}
---
매크로 신호등 v3 (역발상)"""

    return msg


def format_weekly_summary(macro, opp, bt_zone, results):
    """주간 요약 (매주 월요일)"""
    now = datetime.now().strftime("%Y-%m-%d")

    green_n = sum(1 for r in results if r["signal"] == "GREEN")
    yellow_n = sum(1 for r in results if r["signal"] == "YELLOW")
    red_n = sum(1 for r in results if r["signal"] == "RED")

    msg = f"""📊 <b>주간 매크로 신호등 요약</b>
{now}

매수 기회 점수: <b>{opp:.0f}/100</b>
매크로 환경: {macro:.0f}/100

신호 분포: 🟢{green_n} 🟡{yellow_n} 🔴{red_n}

현재 구간: {bt_zone['label']}
  → 3개월 후 기대: {bt_zone['avg']:+.1f}% (승률 {bt_zone['win']:.0f}%)"""

    return msg


def main():
    env = load_env()
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = env.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print("오류: .env 파일에 TELEGRAM_BOT_TOKEN과 TELEGRAM_CHAT_ID를 설정하세요.")
        print(f"파일 위치: {ENV_FILE}")
        return

    # 분석 실행
    data = fetch_all()
    if not data:
        print("데이터 수집 실패")
        return

    results = analyze(data)
    macro = macro_score(results)
    bt_zone = get_backtest_zone(macro)
    opp = contrarian_opportunity(macro)
    opp_sig = opportunity_signal(opp)

    state = load_state()
    prev_sig = state.get("last_signal", "")
    today = datetime.now().strftime("%Y-%m-%d")
    is_monday = datetime.now().weekday() == 0

    should_alert = False
    alert_reason = ""

    # 1) 매수 기회 구간 진입 (이전에 GREEN이 아니었는데 GREEN 됨)
    if opp_sig == "GREEN" and prev_sig != "GREEN":
        should_alert = True
        alert_reason = "매수 기회 구간 진입"

    # 2) 극매수 구간 (항상 알림)
    if opp >= EXTREME_THRESHOLD:
        should_alert = True
        alert_reason = "극매수 구간"

    # 3) 과열 경고 진입
    if opp_sig == "RED" and prev_sig != "RED":
        should_alert = True
        alert_reason = "과열 구간 진입"

    # 4) 주간 요약 (매주 월요일)
    if is_monday and state.get("last_weekly") != today:
        msg = format_weekly_summary(macro, opp, bt_zone, results)
        print(f"주간 요약 발송 중...")
        if send_telegram(token, chat_id, msg):
            print("주간 요약 발송 완료")
            state["last_weekly"] = today

    # 알림 발송
    if should_alert:
        msg = format_alert(macro, opp, opp_sig, bt_zone, results)
        print(f"알림 발송: {alert_reason}")
        if send_telegram(token, chat_id, msg):
            print("발송 완료")
        else:
            print("발송 실패")
    else:
        print(f"[{today}] 매수기회: {opp:.0f}/100 ({opp_sig}) | 매크로: {macro:.0f}/100 - 알림 조건 미충족")

    # 상태 저장
    state["last_signal"] = opp_sig
    state["last_score"] = opp
    state["last_macro"] = macro
    state["last_check"] = today
    save_state(state)


if __name__ == "__main__":
    main()
