# Evaluasi Kritis Dataset Keamanan Siber Publik untuk Peramalan dan Klasifikasi Serangan
### *Critical Evaluation of Public Cybersecurity Dataset for Threat Forecasting and Classification*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Journal: JuTISI](https://img.shields.io/badge/Journal-JuTISI%20(SINTA%202)-success.svg)](https://journal.maranatha.edu/index.php/jutisi)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14042815-blue)](https://doi.org/10.5281/zenodo.14042815)
[![Reproducibility: 100% Verified](https://img.shields.io/badge/Reproducibility-100%25%20Verified-brightgreen.svg)]()

Repositori resmi kode sumber, artefak visual, dan data replikasi untuk artikel ilmiah:
> **Ahmad Zakky Nurfauzan**, *"Evaluasi Kritis Dataset Keamanan Siber Publik untuk Peramalan dan Klasifikasi Serangan"*, **Jurnal Teknik Informatika dan Sistem Informasi (JuTISI)**, 2026.

---

## 📌 Ringkasan Eksekutif (*Executive Summary*)

Penerapan pembelajaran mesin (*Machine Learning*) dan peramalan deret waktu pada log keamanan siber berkembang pesat untuk mitigasi insiden siber secara proaktif. Namun, kelemahan metodologis yang meluas di literatur—terutama **kebocoran data (*data leakage*)** akibat pengacakan temporal (*random shuffling*), ketiadaan **baseline pembanding naif**, dan minimnya **audit statistik distribusi dataset**—berisiko menghasilkan estimasi performa semu yang menyesatkan.

Repositori ini menyediakan kerangka kerja evaluasi empiris diagnostik bebas kebocoran data (*leakage-free evaluation framework*) terhadap dataset publik populer **Cyber Security Attacks** (Kaggle Incribo, 40.000 log, 1.380 hari kalender).

### Temuan Empiris Utama:
1. **Karakteristik Data Sintetis:** 12 dari 13 variabel fitur terbukti konsisten dengan distribusi seragam acak (*uniform distribution*) di bawah uji *Goodness-of-Fit* Chi-Square dan Kolmogorov-Smirnov dengan koreksi Bonferroni ($\alpha_{adj} = 0,0038$), dengan 0 pencilan kontinu.
2. **Proses Poisson Nir-Memori:** Frekuensi serangan harian konsisten dengan proses Poisson homogen tanpa memori ($\text{Varians} / \text{Mean} \approx 1$; uji Ljung-Box $p > 0,05$), menandakan ketiadaan autokorelasi serial maupun tren deterministik nyata.
3. **Patahan Ilusi Peramalan ($R^2 \le 0$):** Seluruh model regresi polinomial (derajat 1–3) menghasilkan nilai $R^2$ *out-of-sample* negatif ($R^2 \le 0$, hingga $-3,24$ pada validasi silang *walk-forward* 5-lipatan), membantah klaim literatur sebelumnya ($R^2 = 0,91$) yang timbul akibat kebocoran data.
4. **Klasifikasi Setara Tebakan Acak:** Akurasi Random Forest multi-kelas (~0,33) tidak berbeda secara signifikan dari tebakan acak seragam ($1/3$) (uji binomial eksak $p > 0,35$).

---

## 🏗️ Kerangka Metodologi 4-Fase (*Framework Architecture*)

Alur kerja penelitian dirancang secara modular dan dapat direplikasi secara penuh:

```
+-----------------------------------------------------------------------------------+
| FASE 1: INGESTI DATA & AUDIT DISTRIBUSI STATISTIK (RQ1)                           |
| - Korpus 40.000 log siber (1.380 hari kalender kontinu, 2020-2023)                |
| - Audit integritas: 0 timestamp rusak, 0 baris duplikat, verifikasi missing value |
| - Eksplorasi batas pencilan non-parametrik IQR [LB, UB] (Tabel 1)                  |
| - 13 Uji Goodness-of-Fit univariat (Chi-Square & KS) vs seragam (alpha_adj=0.0038)|
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| FASE 2: KARAKTERISASI STRUKTUR TEMPORAL DERET WAKTU (RQ2)                         |
| - Agregasi frekuensi serangan harian (DDoS, Malware, dan Intrusion)               |
| - Uji dispersi Poisson (s² / x̄ ≈ 1 proses tanpa memori)                            |
| - Analisis autokorelasi serial ACF (lag 1-40) & uji portmanteau Ljung-Box Q(20)   |
| - Uji tren deterministik Mann-Kendall & uji siklus musiman Kruskal-Wallis         |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| FASE 3: PROTOKOL PARTISI KRONOLOGIS BEBAS KEBOCORAN (RQ2 & RQ3) ★ KONTRIBUSI UTAMA|
| - Pembagian temporal kronologis 80/20 tanpa peracakan (strictly no shuffling)     |
| - Data latih: 1.104 hari (2020 s.d. 8 Jan 2023) | Data uji: 276 hari (s.d. Okt)  |
| - Penskalaan fitur & batas atas IQR clipping dihitung murni train-only            |
| - Validasi silang walk-forward (expanding window) 5-lipatan pada horizon peramalan|
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| FASE 4: EVALUASI MODEL VS BASELINE PEMBANDING OBJEKTIF (RQ2 & RQ3)                 |
| - Peramalan: Regresi Polinomial (Ordo 1-3) vs Baseline Rata-rata Latih,           |
|   Nilai Akhir, & Persistence Bergulir                                             |
| - Evaluasi R² out-of-sample & pembuktian divergensi ekstrapolasi                  |
| - Klasifikasi: Random Forest (300 pohon) vs Baseline Tebakan Acak (1/3) & Mayoritas|
| - Uji Binomial Tepat satu sisi, Matriks Kebingungan, & Bobot Kepentingan Fitur MDI|
+-----------------------------------------------------------------------------------+
```

Visualisasi resolusi tinggi diagram di atas dapat diakses di [`figures/fig1_alur_metodologi.png`](figures/fig1_alur_metodologi.png) dan berkas vektor [`figures/fig1_alur_metodologi.svg`](figures/fig1_alur_metodologi.svg).

---

## 📂 Struktur Repositori (*Repository Structure*)

```
cyber-log-evaluation/
├── citation/
│   ├── referensi_jutisi.bib        # 21 Sitasi IEEE dalam format BibTeX
│   └── referensi_jutisi.ris        # 21 Sitasi IEEE dalam format RIS (Mendeley/Zotero)
├── data/
│   ├── README.md                   # Dokumentasi atribut dan sumber dataset
│   └── cybersecurity_attacks.csv   # Korpus data (40.000 log, 1.380 hari kalender)
├── figures/
│   ├── fig1_alur_metodologi.png    # Gambar 1: Alur metodologi 4-fase (540+ DPI)
│   ├── fig1_alur_metodologi.svg    # Format vektor master Gambar 1
│   ├── fig1_alur_metodologi.html   # Sumber diagram HTML aksesibel
│   ├── fig2_daily_acf.png          # Gambar 2: Deret waktu harian & plot ACF
│   ├── fig3_forecast_test.png      # Gambar 3: Peramalan out-of-sample vs data riil
│   └── fig4_confusion_matrix.png   # Gambar 4: Matriks kebingungan Random Forest
├── LICENSE                         # GNU General Public License v3.0
├── pipeline_evaluasi_dataset.py    # Pipeline eksperimen empiris lengkap (Step 1 - 6)
├── generate_all_figures.py         # Generator visualisasi publikasi
├── requirements.txt                # Dependensi pustaka Python
└── README.md                       # Panduan komprehensif repositori ini
```

---

## 🚀 Panduan Memulai & Replikasi (*Quickstart & Reproduction*)

### 1. Kloning Repositori & Persiapan Lingkungan
```bash
git clone https://github.com/ZakNong7/cyber-log-evaluation.git
cd cyber-log-evaluation

# (Opsional) Buat virtual environment
python3 -m venv venv
source venv/bin/activate  # Untuk Linux/macOS
# venv\Scripts\activate   # Untuk Windows

# Pasang dependensi
pip install -r requirements.txt
```

### 2. Menjalankan Pipeline Evaluasi Penuh
Jalankan skrip evaluasi utama untuk mereplikasi seluruh tabel statistik (Tabel 1 hingga Tabel 6 pada naskah):
```bash
python pipeline_evaluasi_dataset.py
```
*Output hasil analisis statistik dan tabel komparasi CSV akan disimpan secara otomatis di folder `hasil/`.*

### 3. Membangun Ulang Visualisasi Publikasi
Untuk menghasilkan ulang grafik visualisasi beresolusi tinggi (300+ DPI):
```bash
python generate_all_figures.py
```

---

## 📊 Ringkasan Hasil Empiris Utama (*Key Results*)

| Metrik Evaluasi | Nilai Observasi | Interpretasi Metodologis |
| :--- | :---: | :--- |
| **Keseragaman Fitur** | 12 / 13 ($p > 0,0038$) | Fitur kontinu & diskrit terdistribusi seragam tanpa sinyal pola nyata. |
| **Pencilan Kontinu (IQR)** | 0 pencilan ($0,00\%$) | Tidak ada anomali ekstrem (karakteristik batas ketat fungsi random). |
| **Dispersi Poisson** | $\text{Var}/\text{Mean} \in [1,009; 1,039]$ | Proses stokastik murni tanpa memori temporal. |
| **Autokorelasi Serial** | Ljung-Box $p \in [0,218; 0,783]$ | Tidak terdapat dependensi temporal antar hari. |
| **Peramalan Polinomial (OOS)** | $R^2 \in [-0,83; -2,44]$ | Gagal mengungguli baseline naif konstan ($R^2 \le 0$). |
| **Validasi Silang Walk-Forward** | $R^2 \in [-1,44; -3,24]$ | Model mengalami divergensi ekstrapolasi pada data tak terlihat. |
| **Akurasi Random Forest** | $\approx 0,33$ (Uji Binomial $p > 0,35$) | Performa setara dengan tebakan acak seragam ($1/3$). |

---

## 📄 Lisensi (*License*)

Kode sumber dan naskah dalam repositori ini dilisensikan di bawah [GNU General Public License v3.0](LICENSE). Dataset asli dilindungi oleh lisensi publik open-source Kaggle oleh [Team Incribo](https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks).

---

## ✉️ Kontak & Korespondensi
- **Ahmad Zakky Nurfauzan** — [ahmadzakky2862@gmail.com](mailto:ahmadzakky2862@gmail.com)
- **Program Studi D4 Teknik Informatika**, Universitas Logistik dan Bisnis Internasional (ULBI), Bandung, Indonesia.
