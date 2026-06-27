# Dashboard AgriMarket — Kelompok 19 PSD

Dashboard analisis data platform AgriMarket berbasis Streamlit.

## Cara Jalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Cloud (Gratis)

1. Push folder ini ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Login dengan GitHub
4. Klik **New app** → pilih repo → pilih `app.py`
5. Klik **Deploy**
6. Bagikan link ke asprak ✅

## Isi Dashboard

- 📊 Summary cards (total transaksi, pendapatan, rating, kepuasan)
- 📅 Tren penjualan per bulan
- ⏰ Peak hours transaksi
- 🏆 Top 10 produk terlaris
- 🗂️ Distribusi per kategori & status organik
- ⭐ Distribusi rating & kepuasan konsumen
- 💳 Metode pembayaran
- 🌿 Organik vs Non-Organik
- 📐 Matriks korelasi Pearson
- 💰 Scatter plot harga vs subtotal

## Filter

Dashboard dilengkapi filter sidebar: Kategori Produk, Metode Bayar, Status Organik.
