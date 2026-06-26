#!/usr/bin/env python3
"""
매크로 신호등 v3 (역발상 Contrarian)
백테스트 검증 결과를 반영: 공포 = 매수 기회, 과열 = 경고

사용법: python macro_signal.py
결과물: 터미널 출력 + dashboard.html 생성
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# 설정
# ============================================================

LOOKBACK_DAYS = 250

TICKERS = {
    "VIX": "^VIX",
    "SKEW": "^SKEW",
    "HYG": "HYG",
    "LQD": "LQD",
    "USDKRW": "KRW=X",
    "US10Y": "^TNX",
    "GOLD": "GC=F",
    "EEM": "EEM",
    "NASDAQ": "^IXIC",
    "KOSPI": "^KS11",
    "WTI": "CL=F",
    "SOXX": "SOXX",
    "DXY": "DX-Y.NYB",
}

LABELS = {
    "VIX": "VIX 공포지수",
    "VIX_ACCEL": "VIX 가속도",
    "SKEW": "SKEW 꼬리 리스크",
    "HYG": "HYG 정크본드",
    "CREDIT_SPREAD": "신용스프레드 (HYG/LQD)",
    "USDKRW": "USD/KRW 환율",
    "US10Y": "미국 10년물 금리",
    "GOLD": "금 (안전자산)",
    "EEM": "신흥국 (EEM)",
    "NASDAQ": "NASDAQ",
    "KOSPI": "KOSPI",
    "WTI": "WTI 유가",
    "SOXX": "반도체 (SOXX)",
    "DXY": "달러 인덱스",
}

DISPLAY_ORDER = [
    "VIX", "VIX_ACCEL", "SKEW", "HYG", "CREDIT_SPREAD",
    "USDKRW", "US10Y", "GOLD", "EEM", "WTI", "DXY",
    "NASDAQ", "KOSPI", "SOXX",
]

WEIGHTS = {
    "VIX": 0.10, "VIX_ACCEL": 0.08, "SKEW": 0.05, "HYG": 0.05,
    "CREDIT_SPREAD": 0.08,
    "USDKRW": 0.12, "US10Y": 0.08, "GOLD": 0.05, "EEM": 0.06,
    "WTI": 0.04, "DXY": 0.03,
    "NASDAQ": 0.10, "KOSPI": 0.06, "SOXX": 0.10,
}

CATEGORIES = {
    "VIX": "leverage", "VIX_ACCEL": "leverage",
    "SKEW": "leverage", "HYG": "leverage", "CREDIT_SPREAD": "leverage",
    "USDKRW": "macro", "US10Y": "macro", "WTI": "macro", "DXY": "macro",
    "GOLD": "macro", "EEM": "macro",
    "NASDAQ": "market", "KOSPI": "market", "SOXX": "market",
}

CATEGORY_LABELS = {
    "leverage": "레버리지 / 투기 지표",
    "macro": "매크로 환경",
    "market": "시장 추세",
}

# 백테스트 검증된 기대 수익률 (3개월, 2015~2026)
BACKTEST_STATS = {
    "0-25":  {"label": "극공포",   "avg": 9.8,  "win": 95.3, "n": 43},
    "25-40": {"label": "공포",     "avg": 9.0,  "win": 71.3, "n": 174},
    "40-55": {"label": "혼조-",    "avg": 3.8,  "win": 56.5, "n": 354},
    "55-65": {"label": "혼조+",    "avg": 5.3,  "win": 60.4, "n": 225},
    "65-80": {"label": "낙관",     "avg": 6.4,  "win": 65.7, "n": 686},
    "80-100":{"label": "극낙관",   "avg": 0.3,  "win": 52.9, "n": 1070},
}

SIGNAL_SCORE = {"GREEN": 2, "YELLOW": 1, "RED": 0}
SIGNAL_EMOJI = {"GREEN": "[+]", "YELLOW": "[~]", "RED": "[-]"}

# ============================================================
# 유틸리티
# ============================================================


def trend_vs_ma(series, period=50):
    if len(series) < period:
        return series.iloc[-1], series.iloc[-1], 0.0
    ma = series.rolling(period).mean().iloc[-1]
    current = series.iloc[-1]
    pct = (current - ma) / ma * 100
    return current, ma, pct


def pct_change_n(series, n):
    if len(series) <= n:
        return 0.0
    return (series.iloc[-1] - series.iloc[-1 - n]) / series.iloc[-1 - n] * 100


# ============================================================
# 평가 함수 (매크로 환경 기준 - 높을수록 환경 좋음)
# ============================================================


def evaluate_vix(close):
    v = close.iloc[-1]
    chg5 = pct_change_n(close, 5)
    if v < 14:
        signal, desc = "GREEN", "시장 안정"
    elif v < 20:
        signal, desc = "YELLOW", "경계"
    else:
        signal, desc = "RED", "공포"
    return signal, f"{v:.1f}", desc, f"5일 {chg5:+.1f}%"


def evaluate_vix_accel(close):
    chg5 = pct_change_n(close, 5)
    chg1 = pct_change_n(close, 1)
    v = close.iloc[-1]
    if len(close) >= 20:
        avg20 = close.rolling(20).mean().iloc[-1]
        std20 = close.rolling(20).std().iloc[-1]
        z = (v - avg20) / std20 if std20 > 0 else 0
    else:
        z = 0
    if chg5 > 30 or z > 2:
        signal, desc = "RED", "VIX 급등"
    elif chg5 > 15 or z > 1:
        signal, desc = "YELLOW", "VIX 상승 가속"
    elif chg5 < -10:
        signal, desc = "GREEN", "VIX 하락 (안정화)"
    else:
        signal, desc = "GREEN", "VIX 안정"
    return signal, f"{chg5:+.1f}%", desc, f"1일 {chg1:+.1f}% | Z {z:+.1f}"


def evaluate_skew(close):
    v = close.iloc[-1]
    chg5 = pct_change_n(close, 5)
    if v < 135:
        signal, desc = "GREEN", "꼬리 리스크 낮음"
    elif v < 150:
        signal, desc = "YELLOW", "꼬리 리스크 주의"
    else:
        signal, desc = "RED", "블랙스완 경고"
    return signal, f"{v:.0f}", desc, f"5일 {chg5:+.1f}%"


def evaluate_hyg(close):
    cur, ma50, pct50 = trend_vs_ma(close, 50)
    _, ma200, pct200 = trend_vs_ma(close, 200)
    chg20 = pct_change_n(close, 20)
    if cur > ma50 and cur > ma200:
        signal, desc = "GREEN", "신용 안정"
    elif cur > ma200:
        signal, desc = "YELLOW", "신용 경계"
    else:
        signal, desc = "RED", "신용 경색"
    return signal, f"${cur:.1f}", desc, f"50MA {pct50:+.1f}% | 20일 {chg20:+.1f}%"


def evaluate_usdkrw(close):
    v = close.iloc[-1]
    _, _, pct50 = trend_vs_ma(close, 50)
    trend = "상승" if pct50 > 0 else "하락"
    if v < 1320:
        signal, desc = "GREEN", "원화 강세"
    elif v < 1400:
        signal, desc = "YELLOW", "보통"
    else:
        signal, desc = "RED", "원화 약세"
    return signal, f"{v:,.0f}원", desc, f"50MA {pct50:+.1f}% ({trend})"


def evaluate_us10y(close):
    v = close.iloc[-1]
    chg20 = v - close.iloc[-21] if len(close) > 20 else 0
    direction = "상승" if chg20 > 0 else "하락"
    if v < 3.8:
        signal, desc = "GREEN", "저금리"
    elif v < 4.5:
        signal, desc = "YELLOW", "중립"
    else:
        signal, desc = "RED", "고금리"
    return signal, f"{v:.2f}%", desc, f"20일 {chg20:+.2f}%p ({direction})"


def evaluate_index(close, name):
    cur, ma50, pct50 = trend_vs_ma(close, 50)
    _, ma200, pct200 = trend_vs_ma(close, 200)
    chg20 = pct_change_n(close, 20)
    if cur > ma50 and cur > ma200:
        signal, desc = "GREEN", "상승 추세"
    elif cur > ma200:
        signal, desc = "YELLOW", "단기 조정"
    else:
        signal, desc = "RED", "하락 추세"
    fmt = f"{cur:,.0f}" if cur > 100 else f"{cur:.2f}"
    return signal, fmt, desc, f"50MA {pct50:+.1f}% | 200MA {pct200:+.1f}% | 20일 {chg20:+.1f}%"


def evaluate_wti(close):
    v = close.iloc[-1]
    _, _, pct50 = trend_vs_ma(close, 50)
    if 55 <= v <= 80:
        signal, desc = "GREEN", "적정"
    elif 40 <= v <= 95:
        signal, desc = "YELLOW", "주의"
    else:
        signal, desc = "RED", "극단"
    return signal, f"${v:.1f}", desc, f"50MA {pct50:+.1f}%"


def evaluate_dxy(close):
    v = close.iloc[-1]
    _, _, pct50 = trend_vs_ma(close, 50)
    if v < 100:
        signal, desc = "GREEN", "달러 약세"
    elif v < 105:
        signal, desc = "YELLOW", "중립"
    else:
        signal, desc = "RED", "달러 강세"
    return signal, f"{v:.1f}", desc, f"50MA {pct50:+.1f}%"


def evaluate_credit_spread(hyg_close, lqd_close):
    """신용스프레드 (HYG/LQD 비율). 하락 = 스프레드 확대 = 위험"""
    ratio = hyg_close / lqd_close
    v = ratio.iloc[-1]
    chg20 = pct_change_n(ratio, 20)
    _, ma50, pct50 = trend_vs_ma(ratio, 50)
    if v > ma50 and chg20 > -1:
        signal, desc = "GREEN", "신용 안정"
    elif chg20 > -3:
        signal, desc = "YELLOW", "스프레드 확대 주의"
    else:
        signal, desc = "RED", "스프레드 급확대 (위험)"
    return signal, f"{v:.3f}", desc, f"50MA {pct50:+.1f}% | 20일 {chg20:+.1f}%"


def evaluate_gold(close):
    """금 가격. 급등 = 안전자산 수요 폭증 = risk-off"""
    v = close.iloc[-1]
    chg5 = pct_change_n(close, 5)
    chg20 = pct_change_n(close, 20)
    _, _, pct50 = trend_vs_ma(close, 50)
    if chg5 > 5 or chg20 > 10:
        signal, desc = "RED", "금 급등 (패닉 매수)"
    elif pct50 > 3:
        signal, desc = "YELLOW", "금 상승 추세"
    else:
        signal, desc = "GREEN", "안정"
    return signal, f"${v:,.0f}", desc, f"5일 {chg5:+.1f}% | 20일 {chg20:+.1f}%"


def evaluate_eem(close):
    """신흥국 ETF. 하락 = EM 자금 이탈 = KOSPI 선행"""
    cur, ma50, pct50 = trend_vs_ma(close, 50)
    _, ma200, pct200 = trend_vs_ma(close, 200)
    chg20 = pct_change_n(close, 20)
    if cur > ma50 and cur > ma200:
        signal, desc = "GREEN", "자금 유입"
    elif cur > ma200:
        signal, desc = "YELLOW", "단기 이탈"
    else:
        signal, desc = "RED", "자금 이탈 (EM 위험)"
    return signal, f"${cur:.1f}", desc, f"50MA {pct50:+.1f}% | 200MA {pct200:+.1f}% | 20일 {chg20:+.1f}%"


EVALUATORS = {
    "VIX": evaluate_vix,
    "VIX_ACCEL": evaluate_vix_accel,
    "SKEW": evaluate_skew,
    "HYG": evaluate_hyg,
    "CREDIT_SPREAD": None,  # 특수 처리 (HYG+LQD 조합)
    "USDKRW": evaluate_usdkrw,
    "US10Y": evaluate_us10y,
    "GOLD": evaluate_gold,
    "EEM": evaluate_eem,
    "NASDAQ": lambda c: evaluate_index(c, "NASDAQ"),
    "KOSPI": lambda c: evaluate_index(c, "KOSPI"),
    "WTI": evaluate_wti,
    "SOXX": lambda c: evaluate_index(c, "SOXX"),
    "DXY": evaluate_dxy,
}


# ============================================================
# 데이터 수집 & 분석
# ============================================================


def fetch_all():
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS + 50)
    ticker_str = " ".join(TICKERS.values())
    print("데이터 수집 중...")
    raw = yf.download(ticker_str, start=start, end=end, progress=False)
    data = {}
    for name, ticker in TICKERS.items():
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                col = raw["Close"][ticker].dropna()
            else:
                col = raw["Close"].dropna()
            if len(col) > 50:
                data[name] = col
        except (KeyError, Exception) as e:
            print(f"  ! {name}: {e}")
    if "VIX" in data:
        data["VIX_ACCEL"] = data["VIX"]
    # 신용스프레드는 HYG/LQD 비율로 계산
    if "HYG" in data and "LQD" in data:
        data["CREDIT_SPREAD"] = (data["HYG"], data["LQD"])
    return data


def analyze(data):
    results = []
    for name in DISPLAY_ORDER:
        if name not in data:
            continue
        if name == "CREDIT_SPREAD":
            hyg_close, lqd_close = data[name]
            signal, value, desc, detail = evaluate_credit_spread(hyg_close, lqd_close)
        else:
            close = data[name]
            signal, value, desc, detail = EVALUATORS[name](close)
        results.append({
            "key": name, "label": LABELS[name], "signal": signal,
            "value": value, "desc": desc, "detail": detail,
            "weight": WEIGHTS[name], "category": CATEGORIES[name],
        })
    return results


def macro_score(results):
    """매크로 환경 점수 (높을수록 환경 좋음)"""
    total_w, score = 0, 0
    for r in results:
        w = r["weight"]
        score += w * SIGNAL_SCORE[r["signal"]]
        total_w += w
    if total_w == 0:
        return 50
    return score / (total_w * 2) * 100


def get_backtest_zone(macro):
    """매크로 점수 -> 백테스트 검증된 존"""
    if macro < 25:
        return BACKTEST_STATS["0-25"]
    elif macro < 40:
        return BACKTEST_STATS["25-40"]
    elif macro < 55:
        return BACKTEST_STATS["40-55"]
    elif macro < 65:
        return BACKTEST_STATS["55-65"]
    elif macro < 80:
        return BACKTEST_STATS["65-80"]
    else:
        return BACKTEST_STATS["80-100"]


def contrarian_opportunity(macro):
    """
    역발상 매수 기회 점수 (0~100)
    백테스트 결과 기반:
      극공포(0-25) -> 기회 95  |  공포(25-40) -> 기회 85
      혼조-(40-55) -> 기회 45  |  혼조+(55-65) -> 기회 50
      낙관(65-80) -> 기회 40   |  극낙관(80-100) -> 기회 10
    """
    # 구간별 기대수익률을 기회 점수로 매핑
    mapping = [
        (0,   25,  95),
        (25,  40,  85),
        (40,  55,  45),
        (55,  65,  50),
        (65,  80,  40),
        (80,  100, 10),
    ]
    for lo, hi, opp in mapping:
        if lo <= macro < hi:
            # 구간 내 선형 보간
            t = (macro - lo) / (hi - lo)
            return opp
    return 10  # 100점일 때


def opportunity_signal(opp_score):
    if opp_score >= 70:
        return "GREEN"
    elif opp_score >= 40:
        return "YELLOW"
    else:
        return "RED"


# ============================================================
# 터미널 출력
# ============================================================


def print_terminal(results, macro, opp_score, opp_signal, bt_zone):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    bar = "=" * 68

    print(f"\n{bar}")
    print(f"  매크로 신호등 v3 (역발상)  |  {now}")
    print(bar)

    current_cat = None
    for r in results:
        if r["category"] != current_cat:
            current_cat = r["category"]
            print(f"\n  --- {CATEGORY_LABELS[current_cat]} ---")
        emoji = SIGNAL_EMOJI[r["signal"]]
        w = int(r["weight"] * 100)
        print(f"  {emoji} {r['label']} ({w}%): {r['value']}  -  {r['desc']}")
        print(f"     {r['detail']}")

    print(f"\n{bar}")
    print(f"  매크로 환경 점수: {macro:.0f}/100 ({bt_zone['label']})")
    print()

    opp_emoji = SIGNAL_EMOJI[opp_signal]
    print(f"  {opp_emoji} 역발상 매수 기회: {opp_score:.0f}/100")
    print()
    print(f"  [백테스트 근거] 이 구간에서 3개월 후:")
    print(f"     평균 수익률: {bt_zone['avg']:+.1f}%")
    print(f"     승률: {bt_zone['win']:.1f}% (표본 {bt_zone['n']}회)")
    print()

    if opp_signal == "GREEN":
        print(f"  >> 역사적으로 지금은 매수 기회 구간")
    elif opp_signal == "YELLOW":
        print(f"  >> 역사적으로 평범한 구간 (선별 투자)")
    else:
        print(f"  >> 역사적으로 과열 구간 (신규 매수 자제)")

    print(bar)
    print()


# ============================================================
# HTML
# ============================================================


def generate_html(results, macro, opp_score, opp_signal, bt_zone, output_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    SC = {
        "GREEN": {"bg": "#0d4d2b", "border": "#22c55e", "text": "#4ade80", "dot": "#22c55e"},
        "YELLOW": {"bg": "#4a3c00", "border": "#eab308", "text": "#facc15", "dot": "#eab308"},
        "RED": {"bg": "#4c0519", "border": "#ef4444", "text": "#f87171", "dot": "#ef4444"},
    }

    opp_c = SC[opp_signal]

    if opp_signal == "GREEN":
        opp_verdict = "매수 기회 구간"
        opp_detail = "역사적으로 이 수준의 공포에서 매수하면 3개월 후 높은 수익률"
    elif opp_signal == "YELLOW":
        opp_verdict = "평범한 구간"
        opp_detail = "특별한 기회도 위험도 없는 구간. 선별적 투자"
    else:
        opp_verdict = "과열 경고"
        opp_detail = "모두가 낙관적일 때. 역사적으로 추가 수익 여력 낮음"

    # 매크로 점수 위치 표시 (게이지)
    macro_pos = max(2, min(98, macro))

    # 백테스트 존별 막대 그래프
    bt_bars = ""
    for key, stats in BACKTEST_STATS.items():
        bar_w = max(2, stats["avg"] * 5)
        bar_color = "#22c55e" if stats["avg"] > 5 else "#eab308" if stats["avg"] > 2 else "#ef4444"
        is_current = " current-zone" if bt_zone["label"] == stats["label"] else ""
        bt_bars += f"""
        <div class="bt-row{is_current}">
          <span class="bt-label">{stats['label']} ({key}점)</span>
          <div class="bt-bar-wrap">
            <div class="bt-bar" style="width:{bar_w}%; background:{bar_color};"></div>
            <span class="bt-val">{stats['avg']:+.1f}% (승률 {stats['win']:.0f}%)</span>
          </div>
        </div>"""

    # 카테고리별 카드
    cat_html = {}
    for cat_key in ["leverage", "macro", "market"]:
        cards = []
        for r in results:
            if r["category"] != cat_key:
                continue
            c = SC[r["signal"]]
            # 역발상: RED 지표 = 매수 기회
            if r["signal"] == "RED":
                opp_badge = '<span class="opp-badge buy">매수 기회</span>'
            elif r["signal"] == "GREEN":
                opp_badge = '<span class="opp-badge caution">이미 반영</span>'
            else:
                opp_badge = ""
            w = int(r["weight"] * 100)
            cards.append(f"""
            <div class="card" style="border-color:{c['border']};">
              <div class="card-header">
                <span class="dot" style="background:{c['dot']};"></span>
                <span class="card-title">{r['label']}</span>
                <span class="weight">{w}%</span>
                {opp_badge}
              </div>
              <div class="card-value" style="color:{c['text']};">{r['value']}</div>
              <div class="card-desc">{r['desc']}</div>
              <div class="card-detail">{r['detail']}</div>
            </div>""")
        cat_html[cat_key] = "".join(cards)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>매크로 신호등 v3</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, 'Segoe UI', sans-serif;
    background: #0a0a0f; color: #e0e0e0;
    padding: 20px; max-width: 1280px; margin: 0 auto;
  }}
  .header {{ text-align: center; margin-bottom: 24px; }}
  .header h1 {{ font-size: 26px; color: #fff; }}
  .header .sub {{ color: #888; font-size: 13px; margin-top: 4px; }}

  /* 메인 패널 */
  .main-panel {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
    margin-bottom: 24px;
  }}
  @media (max-width: 700px) {{ .main-panel {{ grid-template-columns: 1fr; }} }}

  .panel {{
    background: #111118; border-radius: 14px; padding: 24px; text-align: center;
  }}
  .panel .p-label {{ color: #999; font-size: 12px; margin-bottom: 6px; }}
  .panel .p-value {{ font-size: 52px; font-weight: 800; line-height: 1; }}
  .panel .p-max {{ color: #555; font-size: 18px; font-weight: 400; }}
  .panel .p-verdict {{ margin-top: 10px; font-size: 15px; }}
  .panel .p-detail {{ margin-top: 6px; font-size: 12px; color: #888; }}

  /* 매크로 게이지 */
  .macro-gauge {{
    position: relative; height: 24px; background: linear-gradient(to right, #ef4444, #eab308, #22c55e);
    border-radius: 12px; margin: 16px 0 8px;
  }}
  .macro-gauge .marker {{
    position: absolute; top: -4px; width: 4px; height: 32px;
    background: #fff; border-radius: 2px;
    transform: translateX(-50%);
  }}
  .gauge-labels {{
    display: flex; justify-content: space-between;
    font-size: 10px; color: #666;
  }}

  /* 백테스트 근거 */
  .bt-section {{
    background: #111118; border: 1px solid #2a2a3a; border-radius: 12px;
    padding: 20px; margin-bottom: 24px;
  }}
  .bt-section h2 {{
    font-size: 15px; color: #fff; margin-bottom: 14px;
  }}
  .bt-row {{
    display: flex; align-items: center; padding: 6px 0; gap: 12px;
  }}
  .bt-row.current-zone {{
    background: #1a1a2e; border-radius: 6px; padding: 8px 10px;
    border: 1px solid #3b82f6;
  }}
  .bt-label {{ width: 130px; font-size: 12px; color: #aaa; flex-shrink: 0; }}
  .bt-bar-wrap {{ flex: 1; display: flex; align-items: center; gap: 8px; }}
  .bt-bar {{ height: 16px; border-radius: 4px; min-width: 4px; }}
  .bt-val {{ font-size: 11px; color: #ccc; white-space: nowrap; }}
  .bt-row.current-zone .bt-label {{ color: #fff; font-weight: 600; }}
  .bt-row.current-zone .bt-val {{ color: #fff; font-weight: 600; }}

  /* 핵심 로직 설명 */
  .logic-box {{
    background: #0d1117; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px; margin-bottom: 24px; font-size: 13px; line-height: 1.8;
  }}
  .logic-box strong {{ color: #fff; }}
  .logic-box .formula {{ color: #7ee787; font-family: monospace; }}

  /* 섹션 */
  .section {{ margin-bottom: 24px; }}
  .section-title {{
    font-size: 15px; font-weight: 600; color: #fff;
    margin-bottom: 10px; padding-left: 10px;
    border-left: 3px solid #555;
  }}
  .section-title.leverage {{ border-color: #ef4444; }}
  .section-title.macro {{ border-color: #3b82f6; }}
  .section-title.market {{ border-color: #22c55e; }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
    gap: 12px;
  }}
  .card {{
    border: 1px solid; border-radius: 10px;
    padding: 16px; background: #111118;
  }}
  .card-header {{
    display: flex; align-items: center; gap: 6px;
    margin-bottom: 8px; flex-wrap: wrap;
  }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .card-title {{ font-weight: 600; font-size: 13px; color: #fff; flex: 1; }}
  .weight {{ font-size: 10px; color: #666; }}
  .opp-badge {{
    font-size: 9px; padding: 2px 6px; border-radius: 4px; font-weight: 600;
  }}
  .opp-badge.buy {{ background: #0d4d2b; color: #4ade80; }}
  .opp-badge.caution {{ background: #4a3c00; color: #facc15; }}
  .card-value {{ font-size: 26px; font-weight: 700; margin-bottom: 4px; }}
  .card-desc {{ font-size: 12px; color: #ccc; margin-bottom: 2px; }}
  .card-detail {{ font-size: 11px; color: #777; }}

  .footer {{ text-align: center; margin-top: 24px; font-size: 11px; color: #555; }}
</style>
</head>
<body>
<div class="header">
  <h1>매크로 신호등 v3 (역발상)</h1>
  <div class="sub">"남들이 두려워할 때 탐욕스러워라" -- 백테스트 검증 기반 | {now}</div>
</div>

<div class="main-panel">
  <div class="panel" style="border: 2px solid {opp_c['border']};">
    <div class="p-label">역발상 매수 기회</div>
    <div class="p-value" style="color:{opp_c['text']};">{opp_score:.0f}<span class="p-max"> / 100</span></div>
    <div class="p-verdict" style="color:{opp_c['text']};">{opp_verdict}</div>
    <div class="p-detail">{opp_detail}</div>
  </div>

  <div class="panel" style="border: 1px solid #333;">
    <div class="p-label">매크로 환경 (원래 점수)</div>
    <div class="p-value" style="color:#aaa;">{macro:.0f}<span class="p-max"> / 100</span></div>
    <div class="p-verdict" style="color:#888;">{bt_zone['label']} 구간</div>
    <div class="macro-gauge">
      <div class="marker" style="left:{macro_pos}%;"></div>
    </div>
    <div class="gauge-labels">
      <span>0 (공포)</span><span>50 (중립)</span><span>100 (낙관)</span>
    </div>
  </div>
</div>

<div class="bt-section">
  <h2>백테스트 근거 (2015~2026, KOSPI 3개월 후 수익률)</h2>
  {bt_bars}
  <div style="margin-top: 10px; font-size: 11px; color: #666;">
    현재 구간 파란 테두리 표시 | 높은 막대 = 더 좋은 매수 기회였음
  </div>
</div>

<div class="logic-box">
  <strong>핵심 로직:</strong> 매크로 환경이 나빠 보일수록(점수 낮을수록) 역사적 매수 기회였음<br>
  <span class="formula">극공포(0-25점) -> 3개월 후 +9.8%, 승률 95%</span><br>
  <span class="formula">극낙관(80-100점) -> 3개월 후 +0.3%, 승률 53%</span><br>
  시장은 공포를 이미 가격에 반영하므로, "바닥에서 사는" 전략이 유효
</div>

<div class="section">
  <div class="section-title leverage">레버리지 / 투기 지표</div>
  <div class="grid">{cat_html.get('leverage', '')}</div>
</div>
<div class="section">
  <div class="section-title macro">매크로 환경</div>
  <div class="grid">{cat_html.get('macro', '')}</div>
</div>
<div class="section">
  <div class="section-title market">시장 추세</div>
  <div class="grid">{cat_html.get('market', '')}</div>
</div>

<div class="footer">
  데이터: Yahoo Finance | 과거 성과는 미래를 보장하지 않습니다 | 투자 자문이 아닙니다
</div>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"HTML 대시보드: {output_path}")


# ============================================================
# 메인
# ============================================================


def main():
    data = fetch_all()
    if not data:
        print("데이터 없음")
        return

    print(f"수집: {len(data)}개 지표\n")
    results = analyze(data)
    macro = macro_score(results)
    bt_zone = get_backtest_zone(macro)
    opp = contrarian_opportunity(macro)
    opp_sig = opportunity_signal(opp)

    print_terminal(results, macro, opp, opp_sig, bt_zone)

    html_path = Path(__file__).parent / "dashboard.html"
    generate_html(results, macro, opp, opp_sig, bt_zone, html_path)


if __name__ == "__main__":
    main()
