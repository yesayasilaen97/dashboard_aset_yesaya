import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("🛠️ Sistem Pendukung Keputusan Pemeliharaan Motor (MTR-01 s.d. MTR-12)")
st.markdown("Peringkat prioritas pemeliharaan berbasis **Cost-Sensitive Machine Learning** & **SHAP Explainability**.")

# Load model & data terkini 12 aset
artefak = joblib.load('model_pemeliharaan.pkl')
threshold_opt = artefak['threshold']
df_aset = pd.read_csv('status_aset_terkini.csv')

# Logika Level Alarm & Rekomendasi Tindakan Teknisi
def evaluasi_alarm(prob, thresh):
    if prob >= thresh:
        return (
            "🔴 KRITIS (Alarm Merah)",
            "Prioritas 1: Jadwalkan inspeksi & perbaikan segera (< 24 jam), periksa pelumasan, bantalan (bearing), dan beban kerja."
        )
    elif prob >= (thresh * 0.6):
        return (
            "🟡 WASPADA (Alarm Kuning)",
            "Prioritas 2: Tingkatkan frekuensi monitoring sensor pada shift berikutnya & siapkan suku cadang."
        )
    else:
        return (
            "🟢 NORMAL (Alarm Hijau)",
            "Operasi normal: Lanjutkan jadwal pemeliharaan preventif standar."
        )

hasil = df_aset['probabilitas_rusak'].apply(lambda p: evaluasi_alarm(p, threshold_opt))
df_aset['Level Alarm'] = [h[0] for h in hasil]
df_aset['Rekomendasi Tindakan'] = [h[1] for h in hasil]

# Urutkan peringkat dari probabilitas kerusakan tertinggi ke terendah
df_aset = df_aset.sort_values(by='probabilitas_rusak', ascending=False).reset_index(drop=True)
df_aset.insert(0, 'Peringkat', range(1, len(df_aset) + 1))

# Ringkasan KPI di bagian atas
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cost-Based Threshold", f"{threshold_opt:.2f}")
c2.metric("Total Aset Dipantau", f"{len(df_aset)} Unit")
c3.metric("Alarm Merah (KRITIS)", len(df_aset[df_aset['Level Alarm'].str.contains("KRITIS")]))
c4.metric("Alarm Kuning (WASPADA)", len(df_aset[df_aset['Level Alarm'].str.contains("WASPADA")]))

st.divider()

# Tabel Peringkat Aset
st.subheader("📋 Peringkat Prioritas Pemeliharaan Aset")
kolom_tampil = df_aset[[
    'Peringkat', 'asset_id', 'operating_hours', 'Level Alarm',
    'probabilitas_rusak', 'alasan_utama', 'Rekomendasi Tindakan'
]].copy()

kolom_tampil.columns = [
    'Peringkat', 'ID Aset', 'Jam Operasi', 'Level Alarm',
    'Skor Risiko', 'Alasan Utama (SHAP)', 'Rekomendasi Tindakan Teknisi'
]
kolom_tampil['Skor Risiko'] = (kolom_tampil['Skor Risiko'] * 100).round(1).astype(str) + '%'

st.dataframe(kolom_tampil, use_container_width=True, hide_index=True)

# Grafik batang sederhana skor risiko per aset
st.subheader("📊 Visualisasi Skor Risiko Kerusakan per Aset")
chart_data = df_aset.set_index('asset_id')['probabilitas_rusak']
st.bar_chart(chart_data)