#!/usr/bin/env python3
"""
Auto-update XSMB Top-3 prediction
- Crawl ketqua04.net kết quả XSMB (2 số cuối của tất cả giải)
- Lưu vào xsmb_history.csv
- Tính điểm hot/cold (30 ngày gần nhất)
- Xuất ra index.html (full) và index_top3_only.html (gọn)
"""

import requests, re, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "xsmb_history.csv"

# ----------------------------
# 1. Crawl dữ liệu XSMB
# ----------------------------
def fetch_latest_from_ketqua():
    url = "https://ketqua04.net/xo-so-mien-bac"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    html = r.text

    soup = BeautifulSoup(html, "html.parser")
    # tìm tất cả số có 2 chữ số trở lên
    nums = re.findall(r"\b\d{2,5}\b", html)
    if not nums:
        raise RuntimeError("Không tìm thấy số nào từ ketqua04.net")

    # chỉ giữ lại 2 số cuối
    last2 = [str(int(x[-2:])).zfill(2) for x in nums]

    today = datetime.now().strftime("%Y-%m-%d")
    return today, " ".join(nums), last2


# ----------------------------
# 2. Tính đặc trưng hot/cold
# ----------------------------
def build_features(df, window_days=30):
    rows = []
    for _, r in df.iterrows():
        date = pd.to_datetime(r["date"]).normalize()
        for s in str(r["all_prizes"]).split():
            digits = "".join(ch for ch in s if s.isdigit())
            if len(s) >= 2:
                rows.append({"date": date, "num": int(s[-2:])})
    hist = pd.DataFrame(rows)
    if hist.empty:
        return pd.DataFrame({"num": range(100), "freq_30d": 0,
                             "recency_days": np.nan, "score": 0}), None

    latest_day = pd.to_datetime(hist["date"].max()).normalize()
    window_start = latest_day - timedelta(days=29)
    freq = hist[hist["date"] >= window_start].groupby("num").size().reindex(range(100), fill_value=0)
    last_seen = hist.groupby("num")["date"].max().reindex(range(100))
    recency = (latest_day - last_seen).dt.days

    out = pd.DataFrame({"num": range(100)})
    out["freq_30d"] = freq.values
    out["recency_days"] = recency.values
    fw = out["freq_30d"].astype(float)
    z_hot = (fw - fw.mean()) / (fw.std(ddof=1) or 1.0)
    rc = out["recency_days"].fillna(out["recency_days"].max())
    z_cold = (rc - rc.mean()) / (rc.std(ddof=1) or 1.0)
    out["score"] = 0.6 * z_hot + 0.4 * z_cold
    out = out.sort_values(["score", "num"], ascending=[False, True]).reset_index(drop=True)
    return out, latest_day


# ----------------------------
# 3. Xuất HTML
# ----------------------------
def rebuild_html(feature_df, latest_day, topk=3):
    best = feature_df.head(topk).copy()
    best["num"] = best["num"].apply(lambda x: f"{int(x):02d}")
    disclaimer = ("⚠️ Miễn trừ trách nhiệm: Đây chỉ là công cụ thống kê mô tả dựa trên dữ liệu quá khứ. "
                  "Không đảm bảo kết quả, nội dung chỉ mang tính chất tham khảo. Vui lòng chơi có trách nhiệm.")

    html = f"""<!doctype html>
<html lang="vi"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>XSMB Top-3 dự đoán</title>
</head><body>
<h1>Top-3 gợi ý</h1>
<p>Dữ liệu đến: {'' if latest_day is None else latest_day.strftime('%Y-%m-%d')}</p>
<ul>
{''.join([f"<li>{row['num']} – score {row['score']:.3f}, 30d: {int(row['freq_30d'])}, recency: {'' if pd.isna(row['recency_days']) else int(row['recency_days'])} ngày</li>" for _, row in best.iterrows()])}
</ul>
<p style="color:red">{disclaimer}</p>
</body></html>"""

    (ROOT / "index.html").write_text(html, encoding="utf-8")


def rebuild_html_compact(feature_df, latest_day, topk=3):
    best = feature_df.head(topk).copy()
    best["num"] = best["num"].apply(lambda x: f"{int(x):02d}")
    disclaimer = ("⚠️ Miễn trừ trách nhiệm: Đây chỉ là công cụ thống kê mô tả dựa trên dữ liệu quá khứ. "
                  "Không đảm bảo kết quả, nội dung chỉ mang tính chất tham khảo. Vui lòng chơi có trách nhiệm.")

    html = f"""<!doctype html>
<html lang="vi"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>XSMB Top-3 (Compact)</title>
</head><body>
<h1>Top-3 gợi ý</h1>
<p>Dữ liệu đến: {'' if latest_day is None else latest_day.strftime('%Y-%m-%d')}</p>
<ol>
{''.join([f"<li>{row['num']} – score {row['score']:.3f}</li>" for _, row in best.iterrows()])}
</ol>
<p style="color:red">{disclaimer}</p>
</body></html>"""

    (ROOT / "index_top3_only.html").write_text(html, encoding="utf-8")


# ----------------------------
# 4. Main
# ----------------------------
def main(force=False):
    df = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame(columns=["date", "all_prizes"])

    try:
        today, all_prizes, last2 = fetch_latest_from_ketqua()
        if force or today not in df["date"].astype(str).values:
            df = pd.concat([df, pd.DataFrame([{"date": today, "all_prizes": all_prizes}])],
                           ignore_index=True)
            df.to_csv(CSV_PATH, index=False)
            print("Đã thêm dữ liệu ngày", today)
    except Exception as e:
        print("Không lấy được dữ liệu mới:", e)

    features, latest_day = build_features(df, 30)
    rebuild_html(features, latest_day)
    rebuild_html_compact(features, latest_day)


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force)
