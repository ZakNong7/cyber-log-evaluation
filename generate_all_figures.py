# -*- coding: utf-8 -*-
"""
Script to generate all publication-grade figures for the JuTISI manuscript:
- fig1_alur_metodologi.png: Crisp, legible text (larger fonts), removed 'acak sintetis'
- fig2_daily_acf.png: True Figure 2 (Daily time series 1,380 days + 30-day rolling mean + ACF lag 1-40)
- fig3_forecast_test.png: True Figure 3 (Out-of-sample forecasting on 80/20 test split with clean non-overlapping date ticks)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def calc_acf(x, nlags=40):
    x_diff = x - np.mean(x)
    c0 = np.dot(x_diff, x_diff) / len(x)
    r = [1.0]
    for k in range(1, nlags + 1):
        ck = np.dot(x_diff[:-k], x_diff[k:]) / len(x)
        r.append(ck / c0)
    return np.array(r)

def generate_figure_1(output_path="fig1_alur_metodologi.png"):
    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    c_card_bg = "#F8FAFC"
    c_card_border = "#0F172A"
    c_phase_bg = "#0F172A"
    c_title = "#0F172A"
    c_text = "#1E293B"
    c_divider = "#CBD5E1"

    phases = [
        {
            "num": "FASE 1",
            "title": "Ingesti Data & Audit Distribusi Statistik (RQ1)",
            "points": [
                "Korpus 40.000 log keamanan siber (1.380 hari kalender kontinu, 2020–2023)",
                "Audit integritas: 0 timestamp rusak, 0 baris duplikat, verifikasi missing value",
                "Eksplorasi batas pencilan non-parametrik IQR [LB, UB] pada korpus utuh (Tabel 1)",
                "13 Uji Goodness-of-Fit univariat (Chi-Square & KS) vs distribusi seragam (α_adj = 0,0038)"
            ],
            "y": 76.5
        },
        {
            "num": "FASE 2",
            "title": "Karakterisasi Struktur Temporal Deret Waktu (RQ2)",
            "points": [
                "Agregasi frekuensi serangan harian (DDoS, Malware, dan Intrusion)",
                "Uji dispersi Poisson (evaluasi sifat ekuidispersi s² / x̄ ≈ 1 proses tanpa memori)",
                "Analisis autokorelasi serial ACF (lag 1–20) & uji portmanteau Ljung-Box Q(20)",
                "Uji tren deterministik Mann-Kendall & uji siklus musiman Kruskal-Wallis"
            ],
            "y": 51.5
        },
        {
            "num": "FASE 3",
            "title": "Protokol Partisi Kronologis Bebas Kebocoran Data (RQ2 & RQ3)",
            "points": [
                "Pembagian temporal kronologis 80/20 tanpa peracakan data (strictly no random shuffling)",
                "Data latih: 1.104 hari (2020 s.d. 8 Jan 2023) | Data uji: 276 hari (9 Jan s.d. 11 Okt 2023)",
                "Penskalaan fitur & batas atas IQR clipping dihitung murni train-only bebas kebocoran",
                "Validasi silang walk-forward (expanding window) 5-lipatan pada horizon peramalan"
            ],
            "y": 26.5
        },
        {
            "num": "FASE 4",
            "title": "Evaluasi Model vs Baseline Pembanding Objektif (RQ2 & RQ3)",
            "points": [
                "Peramalan: Regresi Polinomial (Ordo 1–3) vs Baseline Rata-rata Latih, Nilai Akhir, & Persistence",
                "Evaluasi R² out-of-sample pada data uji unseen & pembuktian divergensi ekstrapolasi",
                "Klasifikasi: Random Forest (300 pohon) vs Baseline Tebakan Acak (1/3) & Kelas Mayoritas",
                "Uji Binomial Tepat satu sisi, Matriks Kebingungan, & Bobot Kepentingan Fitur MDI"
            ],
            "y": 1.5
        }
    ]

    card_w = 96.0
    card_h = 20.8
    card_x = 2.0

    for i, p in enumerate(phases):
        y_bot = p["y"]

        # Outer Card with crisp border
        card = patches.FancyBboxPatch(
            (card_x, y_bot), card_w, card_h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor=c_card_bg, edgecolor=c_card_border,
            linewidth=1.2
        )
        ax.add_patch(card)

        # Phase Badge (Top-left pill)
        badge_w = 14.0
        badge_h = 4.2
        badge_x = card_x + 1.8
        badge_y = y_bot + card_h - 4.8
        badge = patches.FancyBboxPatch(
            (badge_x, badge_y), badge_w, badge_h,
            boxstyle="round,pad=0.2,rounding_size=0.8",
            facecolor=c_phase_bg, edgecolor=c_phase_bg,
            linewidth=1.0
        )
        ax.add_patch(badge)
        ax.text(badge_x + badge_w / 2.0, badge_y + badge_h / 2.0, p["num"],
                fontsize=9.0, fontweight="bold", color="#FFFFFF", ha="center", va="center")

        # Phase Title
        ax.text(badge_x + badge_w + 2.2, badge_y + badge_h / 2.0, p["title"],
                fontsize=9.8, fontweight="bold", color=c_title, va="center")

        # Divider line
        div_y = badge_y - 1.2
        ax.plot([card_x + 1.8, card_x + card_w - 1.8], [div_y, div_y], color=c_divider, lw=0.8)

        # Bullets - Legible font size 7.9pt at 300 DPI
        bullets_text = "\n".join([f"•  {pt}" for pt in p["points"]])
        ax.text(card_x + 2.8, div_y - 1.2, bullets_text,
                fontsize=7.9, color=c_text, va="top", linespacing=1.36)

        # Connecting Arrow downwards
        if i < len(phases) - 1:
            next_top = phases[i+1]["y"] + card_h
            ax.annotate(
                "",
                xy=(50.0, next_top),
                xytext=(50.0, y_bot),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=c_card_border,
                    lw=1.5,
                    mutation_scale=12
                )
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gambar 1 tersimpan: {output_path}")

def generate_figure_2_and_3():
    # Load daily series from hasil
    daily = pd.read_csv("hasil/02_deret_harian.csv")
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.set_index("Date")
    ATTACKS = ["DDoS", "Malware", "Intrusion"]

    # ==========================================
    # TRUE FIGURE 2: Deret Waktu Harian & ACF (2 rows, 3 cols)
    # ==========================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 6.8), dpi=300)
    for j, a in enumerate(ATTACKS):
        # Top panel: daily counts and 30-day rolling mean
        ax_top = axes2[0, j]
        ax_top.plot(daily.index, daily[a], lw=0.45, alpha=0.45, color="gray", label="harian")
        ax_top.plot(daily.index, daily[a].rolling(30).mean(), color="tab:red", lw=1.6, label="rata-rata 30 hari")
        ax_top.axhline(daily[a].mean(), color="black", ls=":", lw=1.2, label=f"rata-rata global ({daily[a].mean():.2f})")
        ax_top.set_title(f"{a}: Jumlah Serangan per Hari", fontsize=11, fontweight="bold")
        ax_top.set_ylabel("Jumlah Insiden", fontsize=9.5)
        ax_top.legend(fontsize=8, loc="upper right")
        ax_top.grid(True, ls=":", alpha=0.5)
        ax_top.xaxis.set_major_locator(mdates.YearLocator())
        ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Bottom panel: ACF (lag 1 to 40)
        ax_bot = axes2[1, j]
        r = calc_acf(daily[a].values, nlags=40)
        ax_bot.bar(range(1, 41), r[1:], width=0.55, color="#1E3A8A", alpha=0.85)
        ci = 1.96 / np.sqrt(len(daily))
        ax_bot.axhline(ci, color="tab:red", ls="--", lw=1.2)
        ax_bot.axhline(-ci, color="tab:red", ls="--", lw=1.2)
        ax_bot.axhline(0, color="black", lw=0.8)
        ax_bot.set_title(f"{a}: Autocorrelation Function (ACF)", fontsize=11, fontweight="bold")
        ax_bot.set_xlabel("Lag (Hari)", fontsize=9.5)
        ax_bot.set_ylabel("Autokorelasi r_k", fontsize=9.5)
        ax_bot.set_ylim(-0.15, 0.15)
        ax_bot.grid(True, ls=":", alpha=0.5)

    fig2.tight_layout()
    fig2.savefig("fig2_daily_acf.png", dpi=300)
    fig2.savefig("hasil/fig1_deret_dan_acf.png", dpi=300)
    plt.close(fig2)
    print("Gambar 2 (Deret Harian & ACF) tersimpan: fig2_daily_acf.png (300 DPI)")

    # ==========================================
    # TRUE FIGURE 3: Prediksi pada Data Uji 80/20 (1 row, 3 cols)
    # With clean, non-overlapping date ticks!
    # ==========================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.2), dpi=300)
    TEST_FRAC = 0.20
    
    for j, a in enumerate(ATTACKS):
        y_raw = daily[a].values.astype(float)
        n = len(y_raw)
        n_tr = int(n * (1 - TEST_FRAC))
        
        q1, q3 = np.percentile(y_raw[:n_tr], [25, 75])
        iqr = q3 - q1
        lo_tr, hi_tr = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        y_tr = np.clip(y_raw[:n_tr], max(lo_tr, 0), hi_tr)
        y_te = y_raw[n_tr:]
        
        t = np.arange(n).reshape(-1, 1).astype(float)
        t_scaled = (t - t[:n_tr].mean()) / t[:n_tr].std()
        
        # Fit Polynomial Degree 2
        pf = PolynomialFeatures(2, include_bias=False)
        m = LinearRegression().fit(pf.fit_transform(t_scaled[:n_tr]), y_tr)
        pred_poly2 = np.clip(m.predict(pf.transform(t_scaled[n_tr:])), 0, None)
        pred_base_mean = np.full(len(y_te), y_tr.mean())

        test_dates = daily.index[n_tr:]
        ax = axes3[j]
        ax.plot(test_dates, y_te, lw=0.6, alpha=0.45, color="gray", label="Aktual (Uji)")
        ax.plot(test_dates, pred_base_mean, color="black", ls=":", lw=1.5, label="Baseline Rata-rata Latih")
        ax.plot(test_dates, pred_poly2, color="tab:red", ls="--", lw=1.6, label="Polinomial Derajat 2")
        
        ax.set_title(f"{a}: Prediksi pada Data Uji", fontsize=11, fontweight="bold")
        ax.set_ylabel("Jumlah Serangan Harian", fontsize=9.5)
        ax.set_xlabel("Horizon Evaluasi (2023)", fontsize=9.5)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, ls=":", alpha=0.5)
        
        # Clean x-axis date ticks: every 2 months, format 'Jan 2023', no collision!
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.tick_params(axis='x', labelsize=8.5)

    fig3.tight_layout()
    fig3.savefig("fig3_forecast_test.png", dpi=300)
    fig3.savefig("hasil/fig2_forecasting_uji.png", dpi=300)
    plt.close(fig3)
    print("Gambar 3 (Prediksi Data Uji) tersimpan: fig3_forecast_test.png (300 DPI, label tanggal bersih)")

if __name__ == "__main__":
    generate_figure_1()
    generate_figure_2_and_3()
