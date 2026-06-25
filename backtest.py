#!/usr/bin/env python3
"""
매크로 신호등 백테스트
과거 10년 데이터로 신호등 점수의 실제 예측력을 검증합니다.

핵심 질문: "이 점수가 낮았을 때 진짜로 KOSPI가 떨어졌나?"
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ============================================================
# 설정 (macro_signal.py와 동일한 기준)
# ============================================================

TICKERS = {
    "VIX": "^VIX",
    "SKEW": "^SKEW",
    "HYG": "HYG",
    "USDKRW": "KRW=X",
    "US10Y": "^TNX",
    "NASDAQ": "^IXIC",
    "KOSPI": "^KS11",
    "WTI": "CL=F",
    "SOXX": "SOXX",
    "DXY": "DX-Y.NYB",
}

WEIGHTS = {
    "VIX": 0.12,
    "VIX_ACCEL": 0.10,
    "SKEW": 0.07,
    "HYG": 0.07,
    "USDKRW": 0.15,
    "US10Y": 0.10,
    "NASDAQ": 0.12,
    "KOSPI": 0.06,
    "WTI": 0.05,
    "SOXX": 0.12,
    "DXY": 0.04,
}

LEVERAGE_KEYS = ["VIX", "VIX_ACCEL", "SKEW", "HYG"]

# 전진 수익률 측정 기간 (영업일)
FORWARD_DAYS = {
    20: "1개월",
    60: "3개월",
    120: "6개월",
}

# 점수 구간
SCORE_BINS = [0, 25, 40, 55, 65, 80, 100]
SCORE_LABELS = ["0-25 (극악)", "25-40 (악재)", "40-55 (혼조-)", "55-65 (혼조+)", "65-80 (호재)", "80-100 (극호)"]


# ============================================================
# 데이터 다운로드
# ============================================================


def download_data(start="2015-01-01"):
    end = datetime.now().strftime("%Y-%m-%d")
    ticker_str = " ".join(TICKERS.values())

    print(f"데이터 다운로드 중... ({start} ~ {end})")
    raw = yf.download(ticker_str, start=start, end=end, progress=False)

    data = {}
    for name, ticker in TICKERS.items():
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                col = raw["Close"][ticker].dropna()
            else:
                col = raw["Close"].dropna()
            if len(col) > 200:
                data[name] = col
                print(f"  {name}: {len(col)}일")
            else:
                print(f"  ! {name}: 데이터 부족 ({len(col)}일)")
        except (KeyError, Exception) as e:
            print(f"  ! {name}: 실패 ({e})")

    return data


# ============================================================
# 벡터화된 신호 계산 (매일의 점수를 한번에 계산)
# ============================================================


def compute_signal_series(data):
    """각 지표별 일별 신호 점수 (0=RED, 1=YELLOW, 2=GREEN)"""
    signals = {}

    # VIX 수준
    if "VIX" in data:
        vix = data["VIX"]
        s = pd.Series(1.0, index=vix.index)
        s[vix < 14] = 2.0
        s[vix >= 20] = 0.0
        signals["VIX"] = s

    # VIX 가속도
    if "VIX" in data:
        vix = data["VIX"]
        chg5 = vix.pct_change(5) * 100
        ma20 = vix.rolling(20).mean()
        std20 = vix.rolling(20).std()
        z = (vix - ma20) / std20.replace(0, np.nan)

        s = pd.Series(2.0, index=vix.index)  # 기본 GREEN
        s[(chg5 > 15) | (z > 1)] = 1.0
        s[(chg5 > 30) | (z > 2)] = 0.0
        signals["VIX_ACCEL"] = s

    # SKEW
    if "SKEW" in data:
        skew = data["SKEW"]
        s = pd.Series(1.0, index=skew.index)
        s[skew < 135] = 2.0
        s[skew >= 150] = 0.0
        signals["SKEW"] = s

    # HYG (정크본드)
    if "HYG" in data:
        hyg = data["HYG"]
        ma50 = hyg.rolling(50).mean()
        ma200 = hyg.rolling(200).mean()
        s = pd.Series(0.0, index=hyg.index)
        s[hyg > ma200] = 1.0
        s[(hyg > ma50) & (hyg > ma200)] = 2.0
        signals["HYG"] = s

    # USD/KRW
    if "USDKRW" in data:
        fx = data["USDKRW"]
        s = pd.Series(1.0, index=fx.index)
        s[fx < 1320] = 2.0
        s[fx >= 1400] = 0.0
        signals["USDKRW"] = s

    # US 10Y
    if "US10Y" in data:
        y = data["US10Y"]
        s = pd.Series(1.0, index=y.index)
        s[y < 3.8] = 2.0
        s[y >= 4.5] = 0.0
        signals["US10Y"] = s

    # 지수 추세 (NASDAQ, KOSPI, SOXX)
    for name in ["NASDAQ", "KOSPI", "SOXX"]:
        if name in data:
            p = data[name]
            ma50 = p.rolling(50).mean()
            ma200 = p.rolling(200).mean()
            s = pd.Series(0.0, index=p.index)
            s[p > ma200] = 1.0
            s[(p > ma50) & (p > ma200)] = 2.0
            signals[name] = s

    # WTI
    if "WTI" in data:
        oil = data["WTI"]
        s = pd.Series(1.0, index=oil.index)
        s[(oil >= 55) & (oil <= 80)] = 2.0
        s[(oil > 95) | (oil < 40)] = 0.0
        signals["WTI"] = s

    # DXY
    if "DXY" in data:
        dxy = data["DXY"]
        s = pd.Series(1.0, index=dxy.index)
        s[dxy < 100] = 2.0
        s[dxy >= 105] = 0.0
        signals["DXY"] = s

    return signals


def compute_composite_series(signals):
    """일별 종합 점수 (0~100)"""
    # 모든 신호를 하나의 DataFrame으로
    df = pd.DataFrame(signals)
    # 날짜 정렬 후 forward fill (미국/한국 거래일 차이)
    df = df.sort_index().ffill()

    score = pd.Series(0.0, index=df.index)
    total_w = pd.Series(0.0, index=df.index)

    for col in df.columns:
        w = WEIGHTS.get(col, 0)
        if w == 0:
            continue
        mask = df[col].notna()
        score[mask] += df[col][mask] * w
        total_w[mask] += w

    composite = (score / (total_w * 2)) * 100
    composite[total_w == 0] = np.nan
    return composite


def compute_leverage_series(signals):
    """레버리지 하위 점수"""
    lev_signals = {k: v for k, v in signals.items() if k in LEVERAGE_KEYS}
    if not lev_signals:
        return None
    return compute_composite_series(lev_signals)


# ============================================================
# 전진 수익률 측정
# ============================================================


def measure_forward_returns(kospi, composite, leverage=None):
    """
    각 날짜의 종합 점수 vs N일 후 KOSPI 수익률
    """
    # KOSPI 거래일만 사용
    kospi = kospi.dropna()
    composite = composite.reindex(kospi.index).ffill().dropna()

    # 200MA 계산에 필요한 초기 기간 제거
    composite = composite.iloc[200:]

    records = []
    for fwd_days, fwd_label in FORWARD_DAYS.items():
        fwd_ret = kospi.pct_change(fwd_days).shift(-fwd_days) * 100
        # composite와 fwd_ret 정렬
        aligned = pd.DataFrame({
            "score": composite,
            "fwd_ret": fwd_ret,
        }).dropna()

        if leverage is not None:
            lev = leverage.reindex(aligned.index).ffill()
            aligned["lev_score"] = lev

        for _, row in aligned.iterrows():
            rec = {
                "date": _,
                "score": row["score"],
                "fwd_ret": row["fwd_ret"],
                "fwd_days": fwd_days,
                "fwd_label": fwd_label,
            }
            if "lev_score" in row:
                rec["lev_score"] = row["lev_score"]
            records.append(rec)

    return pd.DataFrame(records)


# ============================================================
# 분석
# ============================================================


def analyze_by_score_bins(df):
    """점수 구간별 전진 수익률 분석"""
    df["score_bin"] = pd.cut(
        df["score"], bins=SCORE_BINS, labels=SCORE_LABELS, include_lowest=True
    )

    results = []
    for fwd_days, fwd_label in FORWARD_DAYS.items():
        subset = df[df["fwd_days"] == fwd_days]
        for bin_label in SCORE_LABELS:
            group = subset[subset["score_bin"] == bin_label]["fwd_ret"]
            if len(group) < 10:
                continue
            results.append({
                "horizon": fwd_label,
                "fwd_days": fwd_days,
                "score_range": bin_label,
                "n": len(group),
                "avg_ret": group.mean(),
                "median_ret": group.median(),
                "win_rate": (group > 0).mean() * 100,
                "worst": group.min(),
                "best": group.max(),
                "std": group.std(),
            })
    return pd.DataFrame(results)


def analyze_by_signal_zone(df):
    """GREEN/YELLOW/RED 존별 분석"""
    def zone(s):
        if s >= 65:
            return "GREEN (65+)"
        elif s >= 40:
            return "YELLOW (40-65)"
        else:
            return "RED (0-40)"

    df["zone"] = df["score"].apply(zone)

    results = []
    for fwd_days, fwd_label in FORWARD_DAYS.items():
        subset = df[df["fwd_days"] == fwd_days]
        for z in ["GREEN (65+)", "YELLOW (40-65)", "RED (0-40)"]:
            group = subset[subset["zone"] == z]["fwd_ret"]
            if len(group) < 5:
                continue
            results.append({
                "horizon": fwd_label,
                "zone": z,
                "n": len(group),
                "avg_ret": group.mean(),
                "median_ret": group.median(),
                "win_rate": (group > 0).mean() * 100,
                "worst": group.min(),
                "best": group.max(),
            })
    return pd.DataFrame(results)


# ============================================================
# 터미널 출력
# ============================================================


def print_results(zone_df, bin_df):
    bar = "=" * 72

    print(f"\n{bar}")
    print("  백테스트 결과: 신호등 존별 KOSPI 전진 수익률")
    print(bar)

    for fwd_label in FORWARD_DAYS.values():
        print(f"\n  --- {fwd_label} 후 수익률 ---\n")
        subset = zone_df[zone_df["horizon"] == fwd_label]
        print(f"  {'존':<20} {'횟수':>6} {'평균':>8} {'중간값':>8} {'승률':>7} {'최악':>8} {'최고':>8}")
        print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")
        for _, row in subset.iterrows():
            print(
                f"  {row['zone']:<20} {row['n']:>6} "
                f"{row['avg_ret']:>+7.1f}% {row['median_ret']:>+7.1f}% "
                f"{row['win_rate']:>6.1f}% {row['worst']:>+7.1f}% {row['best']:>+7.1f}%"
            )

    print(f"\n{bar}")
    print("  세분화 점수 구간별 (3개월 기준)")
    print(bar)

    subset = bin_df[bin_df["fwd_days"] == 60]
    if not subset.empty:
        print(f"\n  {'구간':<20} {'횟수':>6} {'평균':>8} {'승률':>7} {'최악':>8}")
        print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*7} {'-'*8}")
        for _, row in subset.iterrows():
            print(
                f"  {row['score_range']:<20} {row['n']:>6} "
                f"{row['avg_ret']:>+7.1f}% {row['win_rate']:>6.1f}% "
                f"{row['worst']:>+7.1f}%"
            )

    print(f"\n{bar}\n")


# ============================================================
# HTML 리포트
# ============================================================


def generate_html_report(zone_df, bin_df, composite, kospi, output_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    start_date = composite.index[0].strftime("%Y-%m-%d")
    end_date = composite.index[-1].strftime("%Y-%m-%d")
    total_days = len(composite)

    # 존별 분포
    green_pct = (composite >= 65).sum() / total_days * 100
    yellow_pct = ((composite >= 40) & (composite < 65)).sum() / total_days * 100
    red_pct = (composite < 40).sum() / total_days * 100

    # 존별 결과 테이블 HTML
    def zone_table_html(fwd_label):
        subset = zone_df[zone_df["horizon"] == fwd_label]
        if subset.empty:
            return "<p>데이터 부족</p>"
        rows = ""
        for _, r in subset.iterrows():
            zone = r["zone"]
            if "GREEN" in zone:
                color = "#22c55e"
            elif "YELLOW" in zone:
                color = "#eab308"
            else:
                color = "#ef4444"
            avg_color = "#4ade80" if r["avg_ret"] > 0 else "#f87171"
            rows += f"""<tr>
                <td><span style="color:{color};">{zone}</span></td>
                <td>{r['n']}</td>
                <td style="color:{avg_color}; font-weight:700;">{r['avg_ret']:+.1f}%</td>
                <td>{r['median_ret']:+.1f}%</td>
                <td>{r['win_rate']:.0f}%</td>
                <td style="color:#f87171;">{r['worst']:+.1f}%</td>
                <td style="color:#4ade80;">{r['best']:+.1f}%</td>
            </tr>"""
        return f"""<table>
            <tr><th>존</th><th>횟수</th><th>평균 수익률</th><th>중간값</th><th>승률</th><th>최악</th><th>최고</th></tr>
            {rows}
        </table>"""

    # 세분화 테이블 (3개월)
    bin_subset = bin_df[bin_df["fwd_days"] == 60]
    bin_rows = ""
    for _, r in bin_subset.iterrows():
        avg_color = "#4ade80" if r["avg_ret"] > 0 else "#f87171"
        # 바 차트
        bar_w = min(abs(r["avg_ret"]) * 3, 100)
        bar_color = "#22c55e" if r["avg_ret"] > 0 else "#ef4444"
        bar_dir = "right" if r["avg_ret"] > 0 else "left"
        bin_rows += f"""<tr>
            <td>{r['score_range']}</td>
            <td>{r['n']}</td>
            <td style="color:{avg_color}; font-weight:700;">{r['avg_ret']:+.1f}%</td>
            <td>{r['win_rate']:.0f}%</td>
            <td>{r['worst']:+.1f}%</td>
            <td><div class="bar-cell"><div class="bar" style="width:{bar_w}%; background:{bar_color};"></div></div></td>
        </tr>"""

    # 신호등 예측력 판정
    zone_3m = zone_df[zone_df["horizon"] == "3개월"]
    if not zone_3m.empty:
        green_avg = zone_3m[zone_3m["zone"].str.contains("GREEN")]["avg_ret"].values
        red_avg = zone_3m[zone_3m["zone"].str.contains("RED")]["avg_ret"].values
        green_avg = green_avg[0] if len(green_avg) > 0 else 0
        red_avg = red_avg[0] if len(red_avg) > 0 else 0
        spread = green_avg - red_avg

        if spread > 5:
            verdict = "유의미한 예측력 있음"
            verdict_color = "#22c55e"
            verdict_detail = f"GREEN 존 평균 수익률이 RED 존보다 {spread:.1f}%p 높음"
        elif spread > 2:
            verdict = "약한 예측력"
            verdict_color = "#eab308"
            verdict_detail = f"차이가 {spread:.1f}%p로 유의미하나 강하지 않음"
        else:
            verdict = "예측력 미흡"
            verdict_color = "#ef4444"
            verdict_detail = f"GREEN과 RED 차이가 {spread:.1f}%p에 불과"
    else:
        verdict = "데이터 부족"
        verdict_color = "#888"
        verdict_detail = ""
        spread = 0

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>매크로 신호등 백테스트</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, 'Segoe UI', sans-serif;
    background: #0a0a0f; color: #e0e0e0;
    padding: 24px; max-width: 1100px; margin: 0 auto;
  }}
  h1 {{ font-size: 26px; color: #fff; text-align: center; margin-bottom: 4px; }}
  .sub {{ text-align: center; color: #888; font-size: 13px; margin-bottom: 24px; }}

  /* 판정 박스 */
  .verdict-box {{
    background: #111118;
    border: 2px solid {verdict_color};
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    margin-bottom: 24px;
  }}
  .verdict-box .label {{ color: #999; font-size: 13px; }}
  .verdict-box .result {{
    font-size: 28px; font-weight: 800;
    color: {verdict_color}; margin: 8px 0;
  }}
  .verdict-box .detail {{ color: #aaa; font-size: 14px; }}

  /* 분포 바 */
  .dist {{
    display: flex; gap: 0; height: 32px; border-radius: 8px;
    overflow: hidden; margin-bottom: 24px;
  }}
  .dist div {{ display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 600; color: #000; }}

  /* 섹션 */
  .section {{ margin-bottom: 28px; }}
  .section h2 {{
    font-size: 17px; color: #fff; margin-bottom: 12px;
    border-left: 3px solid #555; padding-left: 10px;
  }}

  /* 테이블 */
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #1a1a24; color: #999; padding: 8px 10px; text-align: left;
    font-weight: 500; font-size: 11px; text-transform: uppercase; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #1a1a24; }}
  tr:hover {{ background: #111118; }}

  /* 바 차트 셀 */
  .bar-cell {{ width: 120px; }}
  .bar {{ height: 14px; border-radius: 3px; min-width: 2px; }}

  .note {{
    text-align: center; margin-top: 24px;
    font-size: 11px; color: #555;
  }}

  .insight {{
    background: #111118; border: 1px solid #333; border-radius: 10px;
    padding: 16px; margin-bottom: 20px; font-size: 13px; line-height: 1.7;
  }}
  .insight strong {{ color: #fff; }}
</style>
</head>
<body>
<h1>매크로 신호등 백테스트</h1>
<div class="sub">
  {start_date} ~ {end_date} ({total_days:,}일) | 생성: {now}
</div>

<div class="verdict-box">
  <div class="label">3개월 전진 수익률 기준 신호등 예측력</div>
  <div class="result">{verdict}</div>
  <div class="detail">{verdict_detail}</div>
</div>

<div class="section">
  <h2>신호등 존 분포 (전체 기간)</h2>
  <div class="dist">
    <div style="width:{green_pct:.0f}%; background:#22c55e;">GREEN {green_pct:.0f}%</div>
    <div style="width:{yellow_pct:.0f}%; background:#eab308;">YELLOW {yellow_pct:.0f}%</div>
    <div style="width:{red_pct:.0f}%; background:#ef4444;">RED {red_pct:.0f}%</div>
  </div>
</div>

<div class="section">
  <h2>존별 KOSPI 전진 수익률: 1개월 후</h2>
  {zone_table_html("1개월")}
</div>

<div class="section">
  <h2>존별 KOSPI 전진 수익률: 3개월 후</h2>
  {zone_table_html("3개월")}
</div>

<div class="section">
  <h2>존별 KOSPI 전진 수익률: 6개월 후</h2>
  {zone_table_html("6개월")}
</div>

<div class="section">
  <h2>세분화 점수 구간별 (3개월 기준)</h2>
  <table>
    <tr><th>구간</th><th>횟수</th><th>평균 수익률</th><th>승률</th><th>최악</th><th>분포</th></tr>
    {bin_rows}
  </table>
</div>

<div class="insight">
  <strong>해석 가이드:</strong><br>
  - <strong>평균 수익률</strong>: 해당 존에서 투자 시작했을 때 N개월 후 KOSPI 평균 등락률<br>
  - <strong>승률</strong>: 수익이 난 비율 (50% 이상이면 동전보다 나음)<br>
  - <strong>최악</strong>: 해당 존에서도 발생한 최대 손실 (리스크 인지용)<br>
  - GREEN과 RED의 평균 수익률 차이가 클수록 신호등의 예측력이 강함<br>
  - 단, 과거 성과는 미래를 보장하지 않으며, 기준선(VIX 14, 환율 1400 등)이 변할 수 있음
</div>

<div class="note">
  데이터: Yahoo Finance | 이 분석은 투자 자문이 아닙니다
</div>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"HTML 리포트 생성: {output_path}")


# ============================================================
# 메인
# ============================================================


def main():
    data = download_data("2015-01-01")

    if "KOSPI" not in data:
        print("KOSPI 데이터 없이 백테스트 불가")
        return

    print(f"\n신호 계산 중...")
    signals = compute_signal_series(data)
    composite = compute_composite_series(signals)
    leverage = compute_leverage_series(signals)

    # NaN 제거
    composite = composite.dropna()
    print(f"유효 데이터: {len(composite)}일")

    print("전진 수익률 측정 중...")
    results_df = measure_forward_returns(data["KOSPI"], composite, leverage)

    print("분석 중...\n")
    zone_df = analyze_by_signal_zone(results_df)
    bin_df = analyze_by_score_bins(results_df)

    print_results(zone_df, bin_df)

    html_path = Path(__file__).parent / "backtest_report.html"
    generate_html_report(zone_df, bin_df, composite, data["KOSPI"], html_path)


if __name__ == "__main__":
    main()
