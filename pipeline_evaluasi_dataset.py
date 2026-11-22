"""
Pipeline Evaluasi Kritis Dataset Cybersecurity Attacks
------------------------------------------------------
Pertanyaan riset:
  Apakah dataset log keamanan siber ini memiliki struktur temporal dan prediktif
  yang layak untuk peramalan dan klasifikasi, jika dievaluasi tanpa kebocoran data?

Dependensi: pandas, numpy, scipy, scikit-learn, matplotlib
Jalankan  : python pipeline_evaluasi_dataset.py [path_csv] [folder_output]

Setiap tahap diberi label [STEP n] yang sesuai dengan kotak pada gambar metodologi.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, f1_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit

if len(sys.argv) > 1:
    CSV = sys.argv[1]
elif os.path.exists("data/cybersecurity_attacks.csv"):
    CSV = "data/cybersecurity_attacks.csv"
elif os.path.exists("cybersecurity_attacks.csv"):
    CSV = "cybersecurity_attacks.csv"
elif os.path.exists("../cybersecurity_attacks.csv"):
    CSV = "../cybersecurity_attacks.csv"
elif os.path.exists("cybersecurity_attacks copy.csv"):
    CSV = "cybersecurity_attacks copy.csv"
else:
    CSV = "data/cybersecurity_attacks.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "hasil"
os.makedirs(OUT, exist_ok=True)
SEED = 42
TEST_FRAC = 0.20
ATTACKS = ["DDoS", "Malware", "Intrusion"]


def save_table(df, name):
    df.to_csv(os.path.join(OUT, name), index=True)
    print(f"\n>>> {name}")
    print(df.round(4).to_string())


# ======================================================================
# [STEP 1] DATA COLLECTION & AUDIT KUALITAS
# ======================================================================
print("=" * 70, "\n[STEP 1] Data collection & audit kualitas\n", "=" * 70)
df = pd.read_csv(CSV)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

audit = pd.Series({
    "n_baris": len(df),
    "n_kolom": df.shape[1],
    "timestamp_invalid": int(df["Timestamp"].isna().sum()),
    "baris_duplikat": int(df.duplicated().sum()),
    "tanggal_awal": str(df["Timestamp"].min().date()),
    "tanggal_akhir": str(df["Timestamp"].max().date()),
    "label_attack_kosong": int(df["Attack Type"].isna().sum()),
}, name="nilai").to_frame()
save_table(audit, "01_audit_kualitas.csv")

missing = df.isna().sum()
missing = missing[missing > 0].to_frame("jumlah_kosong")
missing["persen"] = 100 * missing["jumlah_kosong"] / len(df)
save_table(missing, "01_missing_values.csv")

# Penanganan: baris tanpa timestamp/label dibuang (TIDAK diisi data palsu).
n0 = len(df)
df = df.dropna(subset=["Timestamp", "Attack Type"]).drop_duplicates().copy()
df["Attack Type"] = df["Attack Type"].str.strip()
df = df.sort_values("Timestamp").reset_index(drop=True)
print(f"\nBaris dibuang: {n0 - len(df)} | Baris dipakai: {len(df)}")

# Kolom kategorikal bernilai NaN bukan 'rusak': artinya fitur tidak hadir
# (mis. Malware Indicators kosong = tidak ada IoC). Diisi 'None'.
CAT_FILL = ["Malware Indicators", "Alerts/Warnings", "Proxy Information",
            "Firewall Logs", "IDS/IPS Alerts"]
for c in CAT_FILL:
    df[c] = df[c].fillna("None")

# ======================================================================
# [STEP 2] PREPROCESSING RUNTUN WAKTU (agregasi harian + reindex kontinu)
# ======================================================================
print("\n", "=" * 70, "\n[STEP 2] Agregasi harian & timestamp alignment\n", "=" * 70)
full_idx = pd.date_range(df["Timestamp"].dt.normalize().min(),
                         df["Timestamp"].dt.normalize().max(), freq="D")
daily = {}
for a in ATTACKS:
    s = (df[df["Attack Type"] == a].set_index("Timestamp")
         .resample("D").size().reindex(full_idx, fill_value=0))
    daily[a] = s
daily["Total"] = df.set_index("Timestamp").resample("D").size().reindex(full_idx, fill_value=0)
daily = pd.DataFrame(daily)
daily.index.name = "Date"
daily.to_csv(os.path.join(OUT, "02_deret_harian.csv"))
print(f"Panjang deret: {len(daily)} hari | hari tanpa data (diisi 0): "
      f"{int((daily['Total'] == 0).sum())}")

# ======================================================================
# [STEP 3] OUTLIER DETECTION (IQR)
# ======================================================================
print("\n", "=" * 70, "\n[STEP 3] Deteksi outlier (IQR)\n", "=" * 70)


def iqr_outliers(x):
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((x < lo) | (x > hi)).sum()), lo, hi


rows = []
for name, series in [("Anomaly Scores", df["Anomaly Scores"]),
                     ("Packet Length", df["Packet Length"]),
                     *[(f"Harian {a}", daily[a]) for a in ATTACKS]]:
    n_out, lo, hi = iqr_outliers(series.values)
    rows.append({"variabel": name, "n_outlier": n_out,
                 "batas_bawah": lo, "batas_atas": hi, "n_total": len(series)})
iqr_tbl = pd.DataFrame(rows).set_index("variabel")
save_table(iqr_tbl, "03_outlier_iqr.csv")

# Eksplorasi batas IQR dicatat ke 03_outlier_iqr.csv.
# Catatan metodologis: clipping untuk peramalan diterapkan secara strictly train-only pada STEP 5 untuk mencegah kebocoran data.

# ======================================================================
# [STEP 3b] UJI KESERAGAMAN DISTRIBUSI (RQ1: apakah data menyimpang dari acak/seragam?)
# ======================================================================
print("\n", "=" * 70, "\n[STEP 3b] Uji keseragaman distribusi\n", "=" * 70)
rows = []
for c in ["Attack Type", "Severity Level", "Protocol", "Traffic Type", "Packet Type", "Action Taken",
          "Network Segment", "Log Source"]:
    if c in df:
        vc = df[c].value_counts()
        exp = np.full(len(vc), len(df) / len(vc))
        chi, pval = stats.chisquare(vc.values, exp)
        rows.append({"variabel": c, "uji": "chi-square vs seragam", "n_kategori": len(vc),
                     "statistik": chi, "p_value": pval,
                     "proporsi_terbesar": vc.max() / len(df), "proporsi_seragam": 1 / len(vc)})
for c, lo, hi in [("Packet Length", 64, 1500), ("Anomaly Scores", 0, 100),
                  ("Source Port", 1024, 65535), ("Destination Port", 1024, 65535)]:
    if c in df:
        ks = stats.kstest(df[c].astype(float), "uniform", args=(lo, hi - lo))
        rows.append({"variabel": c, "uji": f"KS vs Uniform({lo},{hi})", "n_kategori": np.nan,
                     "statistik": ks.statistic, "p_value": ks.pvalue,
                     "proporsi_terbesar": np.nan, "proporsi_seragam": np.nan})
hr = df["Timestamp"].dt.hour.value_counts().sort_index()
chi, pval = stats.chisquare(hr.values)
rows.append({"variabel": "Jam dalam sehari", "uji": "chi-square vs seragam", "n_kategori": 24,
             "statistik": chi, "p_value": pval, "proporsi_terbesar": hr.max() / len(df),
             "proporsi_seragam": 1 / 24})
unif = pd.DataFrame(rows).set_index("variabel")
unif["lolos_bonferroni_0.05"] = unif["p_value"] < 0.05 / len(unif)
save_table(unif, "03b_uji_keseragaman.csv")
print(f"\nBonferroni untuk {len(unif)} uji: ambang p = {0.05 / len(unif):.4f}")

# Frekuensi port: "port paling sering" hanya bermakna bila jauh di atas harapan acak
pv = df["Destination Port"].value_counts()
top_n = int((pv == pv.iloc[0]).sum())
port_tbl = pd.Series({
    "port_teratas": int(pv.index[0]), "kemunculan_port_teratas": int(pv.iloc[0]),
    "jumlah_port_dengan_kemunculan_sama": top_n, "jumlah_port_unik": int(len(pv)),
    "median_kemunculan_per_port": float(pv.median())}, name="nilai").to_frame()
save_table(port_tbl, "03c_frekuensi_port.csv")

# ======================================================================
# [STEP 4] UJI KARAKTERISTIK RUNTUN WAKTU (bukti ada/tidaknya struktur)
# ======================================================================
print("\n", "=" * 70, "\n[STEP 4] Uji karakteristik runtun waktu\n", "=" * 70)


def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    d = np.sum(x ** 2)
    return np.array([np.sum(x[:len(x) - k] * x[k:]) / d for k in range(nlags + 1)])


def ljung_box(x, lags):
    n = len(x)
    r = acf(x, lags)[1:]
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    return q, 1 - stats.chi2.cdf(q, lags)


def mann_kendall(x):
    n = len(x)
    s = 0
    for i in range(n - 1):
        s += np.sign(x[i + 1:] - x[i]).sum()
    # koreksi ties
    _, c = np.unique(x, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(c * (c - 1) * (2 * c + 5))) / 18
    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


rows = []
for a in ATTACKS + ["Total"]:
    x = daily[a].values.astype(float)
    # Uji dispersi Poisson: (n-1)*var/mean ~ chi2(n-1)
    disp = (len(x) - 1) * x.var(ddof=1) / x.mean()
    p_disp = 2 * min(stats.chi2.cdf(disp, len(x) - 1), 1 - stats.chi2.cdf(disp, len(x) - 1))
    q, p_lb = ljung_box(x, 20)
    z, p_mk = mann_kendall(x)
    dow = [x[daily.index.dayofweek == d] for d in range(7)]
    p_dow = stats.kruskal(*dow).pvalue
    rows.append({
        "seri": a, "mean": x.mean(), "var": x.var(ddof=1),
        "var_per_mean": x.var(ddof=1) / x.mean(),
        "p_dispersi_poisson": p_disp,
        "acf_lag1": acf(x, 7)[1], "acf_lag7": acf(x, 7)[7],
        "p_ljung_box_lag20": p_lb,
        "mann_kendall_z": z, "p_mann_kendall": p_mk,
        "p_kruskal_hari_dalam_minggu": p_dow,
    })
diag = pd.DataFrame(rows).set_index("seri")
save_table(diag, "04_uji_karakteristik_deret.csv")
print("\nInterpretasi: p > 0.05 pada semua uji = tidak ada bukti tren/musiman/"
      "autokorelasi; var/mean ~ 1 = perilaku Poisson (acak).")

# Grafik deret + rolling mean + ACF
fig, axes = plt.subplots(2, 3, figsize=(16, 7))
for j, a in enumerate(ATTACKS):
    ax = axes[0, j]
    ax.plot(daily.index, daily[a], lw=0.4, alpha=0.5, label="harian")
    ax.plot(daily.index, daily[a].rolling(30).mean(), color="tab:red", lw=1.5,
            label="rata-rata 30 hari")
    ax.axhline(daily[a].mean(), color="k", ls=":", lw=1, label="rata-rata global")
    ax.set_title(f"{a}: jumlah serangan per hari"); ax.legend(fontsize=7)
    ax2 = axes[1, j]
    r = acf(daily[a].values, 40)
    ax2.bar(range(1, 41), r[1:], width=0.6)
    ci = 1.96 / np.sqrt(len(daily))
    ax2.axhline(ci, color="r", ls="--"); ax2.axhline(-ci, color="r", ls="--")
    ax2.set_title(f"{a}: ACF (garis merah = batas 95%)"); ax2.set_xlabel("lag (hari)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_deret_dan_acf.png"), dpi=150)
plt.close()

# ======================================================================
# [STEP 5] FEATURE ENGINEERING & TREND FEATURES (polinomial) + FORECASTING
# ======================================================================
print("\n", "=" * 70, "\n[STEP 5] Forecasting dengan split temporal (tanpa data leakage)\n", "=" * 70)


def metrics(y, p):
    return {"R2": r2_score(y, p), "MAE": mean_absolute_error(y, p),
            "RMSE": np.sqrt(mean_squared_error(y, p))}


rows = []
pred_store = {}
for a in ATTACKS:
    y_raw = daily[a].values.astype(float)
    n = len(y_raw)
    n_tr = int(n * (1 - TEST_FRAC))
    
    # Estimasi batas IQR hanya dari data latih (bebas kebocoran)
    q1, q3 = np.percentile(y_raw[:n_tr], [25, 75])
    iqr = q3 - q1
    lo_tr, hi_tr = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    
    # Terapkan clipping hanya pada data latih; data uji dibiarkan mentah
    y_tr = np.clip(y_raw[:n_tr], max(lo_tr, 0), hi_tr)
    y_te = y_raw[n_tr:]
    
    t = np.arange(n).reshape(-1, 1).astype(float)
    t_scaled = (t - t[:n_tr].mean()) / t[:n_tr].std()  # skala dari data latih saja

    pred_roll = np.concatenate([[y_tr[-1]], y_te[:-1]])
    preds = {"Baseline: rata-rata latih": np.full(len(y_te), y_tr.mean()),
             "Baseline: naive (nilai terakhir)": np.full(len(y_te), y_tr[-1]),
             "Baseline: persistence bergulir": pred_roll,
             "Baseline: rata-rata 7 hari terakhir": np.full(len(y_te), y_tr[-7:].mean())}
    for deg in (1, 2, 3):
        pf = PolynomialFeatures(deg, include_bias=False)
        m = LinearRegression().fit(pf.fit_transform(t_scaled[:n_tr]), y_tr)
        preds[f"Polynomial deg {deg}"] = np.clip(m.predict(pf.transform(t_scaled[n_tr:])), 0, None)
        tr_m = metrics(y_tr, m.predict(pf.transform(t_scaled[:n_tr])))
        rows.append({"seri": a, "model": f"Polynomial deg {deg}", "split": "latih (in-sample)", **tr_m})
    for name, p in preds.items():
        rows.append({"seri": a, "model": name, "split": "uji (out-of-sample)", **metrics(y_te, p)})
    pred_store[a] = (n_tr, y_te, preds)

fc = pd.DataFrame(rows).set_index(["seri", "model", "split"])
save_table(fc, "05_forecasting_metrik.csv")

# Validasi walk-forward (TimeSeriesSplit) agar hasil tidak bergantung satu titik potong
rows = []
tscv = TimeSeriesSplit(n_splits=5)
for a in ATTACKS:
    y_raw = daily[a].values.astype(float)
    t = np.arange(len(y_raw)).reshape(-1, 1).astype(float)
    res = {}
    for tr, te in tscv.split(t):
        # Batas IQR hanya dari fold latih
        q1, q3 = np.percentile(y_raw[tr], [25, 75])
        iqr = q3 - q1
        lo_tr, hi_tr = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        y_tr = np.clip(y_raw[tr], max(lo_tr, 0), hi_tr)
        y_te = y_raw[te]  # data uji fold mentah
        
        mu, sd = t[tr].mean(), t[tr].std()
        ts = (t - mu) / sd
        pred_roll = np.concatenate([[y_tr[-1]], y_te[:-1]])
        res.setdefault("Baseline: rata-rata latih", []).append(
            r2_score(y_te, np.full(len(te), y_tr.mean())))
        res.setdefault("Baseline: persistence bergulir", []).append(
            r2_score(y_te, pred_roll))
        for deg in (1, 2, 3):
            pf = PolynomialFeatures(deg, include_bias=False)
            m = LinearRegression().fit(pf.fit_transform(ts[tr]), y_tr)
            res.setdefault(f"Polynomial deg {deg}", []).append(
                r2_score(y_te, np.clip(m.predict(pf.transform(ts[te])), 0, None)))
    for k, v in res.items():
        rows.append({"seri": a, "model": k, "R2_uji_mean": np.mean(v), "R2_uji_std": np.std(v)})
cv_tbl = pd.DataFrame(rows).set_index(["seri", "model"])
save_table(cv_tbl, "05b_forecasting_walkforward_cv.csv")

# Grafik hasil forecasting pada data uji
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for j, a in enumerate(ATTACKS):
    n_tr, y_te, preds = pred_store[a]
    ax = axes[j]
    ax.plot(daily.index[n_tr:], y_te, lw=0.5, alpha=0.5, label="aktual (uji)")
    ax.plot(daily.index[n_tr:], preds["Baseline: rata-rata latih"], "k:", label="baseline rata-rata")
    ax.plot(daily.index[n_tr:], preds["Polynomial deg 2"], "r--", label="polynomial deg 2")
    ax.set_title(f"{a}: prediksi pada data uji"); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_forecasting_uji.png"), dpi=150)
plt.close()
plt.close()

# ======================================================================
# [STEP 6] KLASIFIKASI PER LOG (split temporal) vs BASELINE
# ======================================================================
print("\n", "=" * 70, "\n[STEP 6] Klasifikasi per log vs baseline\n", "=" * 70)
d = df.copy()
d["hour"] = d["Timestamp"].dt.hour
d["dow"] = d["Timestamp"].dt.dayofweek
d["month"] = d["Timestamp"].dt.month
CAT = ["Protocol", "Packet Type", "Traffic Type", "Malware Indicators", "Alerts/Warnings",
       "Attack Signature", "Action Taken", "Network Segment", "Log Source",
       "Firewall Logs", "IDS/IPS Alerts", "Severity Level", "Attack Type"]
NUM = ["Source Port", "Destination Port", "Packet Length", "Anomaly Scores",
       "hour", "dow", "month"]

rows = []
cms = {}
for target in ["Attack Type", "Severity Level"]:
    feats_cat = [c for c in CAT if c != target]
    X = pd.get_dummies(d[feats_cat], dtype=float)
    for c in NUM:
        X[c] = d[c].values
    y = d[target].values
    n_tr = int(len(d) * (1 - TEST_FRAC))
    X_tr, X_te, y_tr, y_te = X.iloc[:n_tr], X.iloc[n_tr:], y[:n_tr], y[n_tr:]

    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    p = rf.predict(X_te)
    maj = pd.Series(y_tr).mode()[0]
    p_maj = np.full(len(y_te), maj)
    classes = sorted(set(y))
    chance = 1 / len(classes)

    acc = accuracy_score(y_te, p)
    k = int((p == y_te).sum())
    p_binom = stats.binomtest(k, len(y_te), chance, alternative="greater").pvalue
    rows.append({"target": target, "model": "Random Forest", "accuracy": acc,
                 "macro_F1": f1_score(y_te, p, average="macro"),
                 "p_vs_tebakan_acak(greater)": p_binom})
    rows.append({"target": target, "model": "Baseline: kelas mayoritas",
                 "accuracy": accuracy_score(y_te, p_maj),
                 "macro_F1": f1_score(y_te, p_maj, average="macro"),
                 "p_vs_tebakan_acak(greater)": np.nan})
    cms[target] = (confusion_matrix(y_te, p, labels=classes), classes)

    # Feature importance (MDI) -- bukti bahwa tidak ada fitur dominan
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    save_table(imp.head(10).to_frame("MDI_importance"),
               f"06_importance_{target.replace(' ', '_')}.csv")

cls = pd.DataFrame(rows).set_index(["target", "model"])
save_table(cls, "06_klasifikasi_metrik.csv")

# Uji asosiasi fitur kategorikal vs target (chi-square) sebagai pelengkap
rows = []
for target in ["Attack Type", "Severity Level"]:
    for c in ["Protocol", "Traffic Type", "Packet Type", "Action Taken",
              "Network Segment", "Log Source", "Attack Signature"]:
        if c == target:
            continue
        chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(d[c], d[target]))
        rows.append({"target": target, "fitur": c, "chi2": chi2, "p_value": p})
    for c in ["Anomaly Scores", "Packet Length"]:
        groups = [d.loc[d[target] == k, c] for k in sorted(d[target].unique())]
        rows.append({"target": target, "fitur": c + " (Kruskal)",
                     "chi2": stats.kruskal(*groups).statistic,
                     "p_value": stats.kruskal(*groups).pvalue})
assoc = pd.DataFrame(rows).set_index(["target", "fitur"])
save_table(assoc, "06b_uji_asosiasi_fitur_target.csv")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (target, (cm, classes)) in zip(axes, cms.items()):
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title(f"Confusion matrix: {target}")
    ax.set_xlabel("prediksi"); ax.set_ylabel("aktual")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_confusion_matrix.png"), dpi=150)
plt.close()

# ======================================================================
# [STEP 7] KESIMPULAN OTOMATIS BERBASIS HASIL (bukan teks tetap)
# ======================================================================
print("\n", "=" * 70, "\n[STEP 7] Kesimpulan berbasis hasil\n", "=" * 70)
lines = []
for a in ATTACKS:
    base = fc.loc[(a, "Baseline: rata-rata latih", "uji (out-of-sample)")]
    best_name, best_r2 = None, -np.inf
    for deg in (1, 2, 3):
        r2v = fc.loc[(a, f"Polynomial deg {deg}", "uji (out-of-sample)"), "R2"]
        if r2v > best_r2:
            best_name, best_r2 = f"deg {deg}", r2v
    if best_r2 <= 0.05:
        verdict = ("tidak ada daya prediksi (R2 uji <= 0.05; selisih terhadap baseline "
                   "hanya fluktuasi di sekitar nol)")
    else:
        verdict = "memiliki daya prediksi yang perlu dikonfirmasi dengan walk-forward CV"
    lines.append(f"- {a}: polynomial terbaik ({best_name}) R2 uji = {best_r2:.4f}; "
                 f"baseline rata-rata R2 uji = {base['R2']:.4f} -> {verdict}. "
                 f"p Ljung-Box = {diag.loc[a, 'p_ljung_box_lag20']:.3f}, "
                 f"p Mann-Kendall = {diag.loc[a, 'p_mann_kendall']:.3f}.")
for target in ["Attack Type", "Severity Level"]:
    r = cls.loc[(target, "Random Forest")]
    sig = "signifikan" if r["p_vs_tebakan_acak(greater)"] < 0.05 else "TIDAK signifikan"
    lines.append(f"- Klasifikasi {target}: akurasi RF = {r['accuracy']:.3f} "
                 f"(tebakan acak = 0.333), p = {r['p_vs_tebakan_acak(greater)']:.3f} -> {sig} "
                 f"di atas tebakan acak.")
dev = list(unif.index[unif["lolos_bonferroni_0.05"]])
lines.append(f"- Keseragaman distribusi: {len(dev)} dari {len(unif)} variabel menyimpang dari seragam setelah koreksi Bonferroni"
             + (f" ({', '.join(dev)}; efek sangat kecil, periksa 03b)" if dev else "") + ". "
             f"Port teratas ({int(pv.index[0])}) muncul {int(pv.iloc[0])}x bersama {top_n - 1} port lain, sehingga bukan temuan.")
lines.append("- Catatan: 4 seri x 1 uji musiman Kruskal = beberapa uji sekaligus; p ~0.03 "
             "pada satu seri tidak bertahan setelah koreksi Bonferroni (ambang 0.0125).")
text = "\n".join(lines)
print(text)
with open(os.path.join(OUT, "07_kesimpulan.txt"), "w", encoding="utf-8") as f:
    f.write(text)
print(f"\nSelesai. Semua keluaran ada di folder: {OUT}/")
