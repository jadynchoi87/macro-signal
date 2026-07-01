"""
매크로 신호등 regime.json 클라이언트.
김장열봇, Ross Scanner 등 다른 봇에서 복사해서 사용.

사용법:
    from regime_client import fetch_regime
    regime = fetch_regime()
    print(regime["regime"])           # "극공포" | "공포" | "혼조-" | ...
    print(regime["opportunity_score"]) # 0~100
    print(regime["regime_signal"])     # "GREEN" | "YELLOW" | "RED"
"""

import json
import urllib.request
from datetime import datetime

REGIME_URL = (
    "https://raw.githubusercontent.com/jadynchoi87/macro-signal/master/regime.json"
)

# 오프라인/장애 시 안전 기본값 (중립)
_DEFAULT_REGIME = {
    "updated_at": "",
    "macro_score": 50.0,
    "opportunity_score": 45.0,
    "regime": "혼조-",
    "regime_signal": "YELLOW",
    "backtest": {"avg_return_3m": 3.8, "win_rate": 56.5, "sample_n": 354},
    "red_indicators": [],
    "green_indicators": [],
    "signal_counts": {"green": 0, "yellow": 0, "red": 0},
    "_fallback": True,
}


def fetch_regime(url: str = REGIME_URL, timeout: int = 10) -> dict:
    """GitHub에서 regime.json을 가져온다. 실패 시 안전 기본값 반환."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "regime-client/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        data["_fallback"] = False

        # 데이터 신선도 체크 (3일 이상 지나면 경고)
        try:
            updated = datetime.fromisoformat(data["updated_at"])
            age_hours = (datetime.now() - updated).total_seconds() / 3600
            data["_stale"] = age_hours > 72
            data["_age_hours"] = round(age_hours, 1)
        except (ValueError, KeyError):
            data["_stale"] = True
            data["_age_hours"] = -1

        return data
    except Exception as e:
        print(f"[regime_client] fetch 실패 ({e}), 기본값 사용")
        return {**_DEFAULT_REGIME}
