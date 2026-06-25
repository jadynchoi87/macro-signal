#!/usr/bin/env python3
"""
GitHub Actions용 매크로 신호등 모니터
환경변수에서 텔레그램 설정을 읽음 (GitHub Secrets)
"""

import os
import json
import urllib.request
import urllib.parse
from datetime import datetime

from macro_signal import (
    fetch_all, analyze, macro_score,
    get_backtest_zone, contrarian_opportunity, opportunity_signal,
)

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

ALERT_THRESHOLD = 70
EXTREME_THRESHOLD = 85
OVERHEAT_THRESHOLD = 20


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": CHAT_ID, "text": message, "parse_mode": "HTML",
    }).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"텔레그램 실패: {e}")
        return False


def format_alert(macro, opp, bt_zone, results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    if opp >= EXTREME_THRESHOLD:
        header = "🔴🔴🔴 극매수 신호 🔴🔴🔴"
    elif opp >= ALERT_THRESHOLD:
        header = "🟢 매수 기회 감지"
    elif opp <= OVERHEAT_THRESHOLD:
        header = "⚠️ 과열 경고"
    else:
        header = "📊 매크로 신호등"

    indicators = ""
    for r in results:
        icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}[r["signal"]]
        indicators += f"  {icon} {r['label']}: {r['value']} ({r['desc']})\n"

    red_items = [r for r in results if r["signal"] == "RED"]
    red_summary = ""
    if red_items:
        red_summary = "\n<b>역발상 매수 기회 요인:</b>\n"
        for r in red_items:
            red_summary += f"  📍 {r['label']}: {r['value']}\n"

    return f"""<b>{header}</b>
{now} UTC

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


def format_weekly(macro, opp, bt_zone, results):
    now = datetime.now().strftime("%Y-%m-%d")
    g = sum(1 for r in results if r["signal"] == "GREEN")
    y = sum(1 for r in results if r["signal"] == "YELLOW")
    rd = sum(1 for r in results if r["signal"] == "RED")
    return f"""📊 <b>주간 매크로 신호등</b>
{now}

매수 기회: <b>{opp:.0f}/100</b>
매크로 환경: {macro:.0f}/100

신호: 🟢{g} 🟡{y} 🔴{rd}

구간: {bt_zone['label']}
  → 3개월 기대: {bt_zone['avg']:+.1f}% (승률 {bt_zone['win']:.0f}%)"""


def main():
    if not TOKEN or not CHAT_ID:
        print("오류: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 환경변수 없음")
        return

    data = fetch_all()
    if not data:
        print("데이터 수집 실패")
        return

    results = analyze(data)
    macro = macro_score(results)
    bt = get_backtest_zone(macro)
    opp = contrarian_opportunity(macro)
    opp_sig = opportunity_signal(opp)

    is_monday = datetime.now().weekday() == 0
    should_alert = False

    if opp >= ALERT_THRESHOLD:
        should_alert = True
    if opp <= OVERHEAT_THRESHOLD:
        should_alert = True

    # 월요일: 무조건 주간 요약
    if is_monday:
        msg = format_weekly(macro, opp, bt, results)
        send_telegram(msg)
        print("주간 요약 발송")

    # 알림 조건 충족 시
    if should_alert:
        msg = format_alert(macro, opp, bt, results)
        send_telegram(msg)
        print(f"알림 발송: 매수기회 {opp:.0f}/100")
    else:
        print(f"매수기회: {opp:.0f}/100 ({opp_sig}) | 매크로: {macro:.0f}/100 - 알림조건 미충족")


if __name__ == "__main__":
    main()
