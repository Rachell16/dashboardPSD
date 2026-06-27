import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard AgriMarket — Kelompok 19",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8faf8; }
    .block-container { padding: 1.5rem 2rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #e8f5e9;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-val { font-size: 1.6rem; font-weight: 700; color: #2C5F2D; }
    .metric-lbl { font-size: 0.72rem; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-sub { font-size: 0.7rem; color: #9ca3af; margin-top: 2px; }
    .section-title { font-size: 1rem; font-weight: 700; color: #1a3a1b; margin-bottom: 0.2rem; }
    .section-sub { font-size: 0.75rem; color: #9ca3af; margin-bottom: 0.8rem; }
    .stPlotlyChart { border-radius: 12px; }
    div[data-testid="stMetric"] { background: white; border-radius: 12px; padding: 0.8rem; border: 1px solid #e8f5e9; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("agrimarket_dataset_clean_v2.csv")
    df['waktu_pesanan'] = pd.to_datetime(df['waktu_pesanan'])
    df['bulan'] = df['waktu_pesanan'].dt.to_period('M').astype(str)
    df['is_puas'] = (df['rating_transaksi'] >= 4).astype(int)
    df['label_organik'] = df['is_organik'].map({1: 'Organik', 0: 'Non-Organik'})
    df['label_puas'] = df['is_puas'].map({1: 'Puas (≥4)', 0: 'Tidak Puas (<4)'})
    return df

df = load_data()

# ── Warna ─────────────────────────────────────────────────────
HIJAU = ['#2C5F2D','#388E3C','#43A047','#66BB6A','#81C784','#A5D6A7']
MERAH_HIJAU = ['#E53E3E','#ED8936','#ECC94B','#68D391','#2C5F2D']

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #2C5F2D 0%, #388E3C 100%);
     padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;'>
    <h1 style='color:white; margin:0; font-size:1.6rem;'>🌾 Dashboard Analisis Data AgriMarket</h1>
    <p style='color:#A5D6A7; margin:0.3rem 0 0; font-size:0.85rem;'>
        Platform E-Commerce Pertanian · Kelompok Tunas Muda Cianjur · Kelompok 19 PSD
    </p>
</div>
""", unsafe_allow_html=True)

# ── Filter Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filter Data")
    kategori_list = ['Semua'] + sorted(df['kategori_produk'].unique().tolist())
    selected_kat = st.selectbox("Kategori Produk", kategori_list)
    metode_list = ['Semua'] + sorted(df['metode_bayar'].unique().tolist())
    selected_metode = st.selectbox("Metode Bayar", metode_list)
    organik_list = ['Semua', 'Organik', 'Non-Organik']
    selected_organik = st.selectbox("Status Organik", organik_list)
    st.markdown("---")
    st.markdown("**Info Dataset**")
    st.caption(f"Total baris: {len(df):,}")
    st.caption(f"Produk unik: {df['id_produk'].nunique()}")
    st.caption(f"Konsumen unik: {df['id_user'].nunique()}")

# Apply filters
dff = df.copy()
if selected_kat != 'Semua':
    dff = dff[dff['kategori_produk'] == selected_kat]
if selected_metode != 'Semua':
    dff = dff[dff['metode_bayar'] == selected_metode]
if selected_organik != 'Semua':
    dff = dff[dff['label_organik'] == selected_organik]

# ── Summary Cards ─────────────────────────────────────────────
st.markdown("### 📊 Ringkasan Data")
c1, c2, c3, c4, c5, c6 = st.columns(6)

total_tx = len(dff)
total_rev = dff['subtotal_transaksi'].sum()
avg_rating = dff['rating_transaksi'].mean()
pct_puas = dff['is_puas'].mean() * 100
avg_subtotal = dff['subtotal_transaksi'].mean()
total_produk = dff['id_produk'].nunique()

with c1:
    st.metric("📦 Total Transaksi", f"{total_tx:,}")
with c2:
    st.metric("💰 Total Pendapatan", f"Rp {total_rev/1_000_000:.1f}jt")
with c3:
    st.metric("⭐ Rating Rata-rata", f"{avg_rating:.2f} / 5")
with c4:
    st.metric("😊 Konsumen Puas", f"{pct_puas:.1f}%")
with c5:
    st.metric("🛒 Rata-rata Subtotal", f"Rp {avg_subtotal/1000:.0f}rb")
with c6:
    st.metric("🌱 Produk Unik", f"{total_produk}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Baris 1: Tren Penjualan & Peak Hours ──────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<p class="section-title">📅 Tren Penjualan per Bulan</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Total subtotal transaksi & jumlah order per bulan</p>', unsafe_allow_html=True)
    bulan_df = dff.groupby('bulan').agg(
        pendapatan=('subtotal_transaksi','sum'),
        jumlah_order=('id_order_detail','count')
    ).reset_index().sort_values('bulan')

    fig_tren = make_subplots(specs=[[{"secondary_y": True}]])
    fig_tren.add_trace(go.Scatter(x=bulan_df['bulan'], y=bulan_df['pendapatan'],
        name='Pendapatan (Rp)', line=dict(color='#2C5F2D', width=2.5),
        fill='tozeroy', fillcolor='rgba(44,95,45,0.08)'), secondary_y=False)
    fig_tren.add_trace(go.Scatter(x=bulan_df['bulan'], y=bulan_df['jumlah_order'],
        name='Jumlah Order', line=dict(color='#66BB6A', width=2, dash='dot'),
        mode='lines+markers', marker=dict(size=6)), secondary_y=True)
    fig_tren.update_layout(height=270, margin=dict(t=10,b=10,l=10,r=10),
        legend=dict(orientation='h', y=-0.2), paper_bgcolor='white',
        plot_bgcolor='white', font=dict(size=11))
    fig_tren.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig_tren.update_yaxes(showgrid=True, gridcolor='#f0f0f0', secondary_y=False)
    fig_tren.update_yaxes(showgrid=False, secondary_y=True)
    st.plotly_chart(fig_tren, use_container_width=True)

with col2:
    st.markdown('<p class="section-title">⏰ Peak Hours Transaksi</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Distribusi transaksi per jam (00-23)</p>', unsafe_allow_html=True)
    jam_df = dff.groupby('jam_pesanan').size().reset_index(name='jumlah')
    jam_df['warna'] = jam_df['jam_pesanan'].apply(lambda x: '#2C5F2D' if 7 <= x <= 10 else '#A5D6A7')
    fig_jam = px.bar(jam_df, x='jam_pesanan', y='jumlah',
        color='warna', color_discrete_map='identity')
    fig_jam.update_layout(height=270, margin=dict(t=10,b=10,l=10,r=10),
        showlegend=False, paper_bgcolor='white', plot_bgcolor='white',
        font=dict(size=10), xaxis_title="Jam", yaxis_title="Transaksi")
    fig_jam.update_xaxes(tickvals=list(range(0,24)), showgrid=False)
    fig_jam.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig_jam, use_container_width=True)
    st.caption("🟢 Hijau tua = peak hour (07:00–10:00)")

# ── Baris 2: Kategori & Produk Terlaris ───────────────────────
col3, col4 = st.columns([2, 3])

with col3:
    st.markdown('<p class="section-title">🗂️ Transaksi per Kategori</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Volume transaksi & status organik</p>', unsafe_allow_html=True)
    kat_df = dff.groupby(['kategori_produk','label_organik']).size().reset_index(name='jumlah')
    fig_kat = px.bar(kat_df, x='kategori_produk', y='jumlah', color='label_organik',
        color_discrete_map={'Organik':'#2C5F2D','Non-Organik':'#A5D6A7'},
        barmode='group', text='jumlah')
    fig_kat.update_traces(textposition='outside', textfont_size=10)
    fig_kat.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10),
        legend=dict(orientation='h', y=-0.25, title=''),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(size=10), xaxis_title='', yaxis_title='Transaksi')
    fig_kat.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig_kat, use_container_width=True)

with col4:
    st.markdown('<p class="section-title">🏆 Top 10 Produk Terlaris</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Berdasarkan jumlah unit terjual</p>', unsafe_allow_html=True)
    top10 = dff.groupby('nama_produk')['jumlah_pembelian'].sum().sort_values(ascending=True).tail(10)
    fig_top = px.bar(top10, orientation='h',
        color=top10.values, color_continuous_scale=['#A5D6A7','#2C5F2D'])
    fig_top.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10),
        showlegend=False, coloraxis_showscale=False,
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(size=10), xaxis_title='Unit Terjual', yaxis_title='')
    fig_top.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig_top.update_yaxes(showgrid=False)
    st.plotly_chart(fig_top, use_container_width=True)

# ── Baris 3: Rating, Kepuasan, Metode Bayar, Organik ─────────
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.markdown('<p class="section-title">⭐ Distribusi Rating</p>', unsafe_allow_html=True)
    rating_df = dff['rating_transaksi'].value_counts().sort_index().reset_index()
    rating_df.columns = ['rating','jumlah']
    fig_rat = px.bar(rating_df, x='rating', y='jumlah',
        color='rating', color_continuous_scale=MERAH_HIJAU,
        text='jumlah')
    fig_rat.update_traces(textposition='outside', textfont_size=10)
    fig_rat.update_layout(height=230, margin=dict(t=10,b=10,l=10,r=10),
        showlegend=False, coloraxis_showscale=False,
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(size=10), xaxis_title='Rating', yaxis_title='')
    fig_rat.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig_rat, use_container_width=True)

with col6:
    st.markdown('<p class="section-title">😊 Kepuasan Konsumen</p>', unsafe_allow_html=True)
    puas_df = dff['label_puas'].value_counts().reset_index()
    puas_df.columns = ['label','jumlah']
    fig_puas = px.pie(puas_df, values='jumlah', names='label',
        hole=0.5, color='label',
        color_discrete_map={'Puas (≥4)':'#2C5F2D','Tidak Puas (<4)':'#E53E3E'})
    fig_puas.update_traces(textposition='inside', textinfo='percent')
    fig_puas.update_layout(height=230, margin=dict(t=10,b=30,l=10,r=10),
        legend=dict(orientation='h', y=-0.15, font=dict(size=9)),
        paper_bgcolor='white', font=dict(size=10))
    st.plotly_chart(fig_puas, use_container_width=True)

with col7:
    st.markdown('<p class="section-title">💳 Metode Pembayaran</p>', unsafe_allow_html=True)
    metode_df = dff['metode_bayar'].value_counts().reset_index()
    metode_df.columns = ['metode','jumlah']
    fig_met = px.pie(metode_df, values='jumlah', names='metode',
        hole=0.45, color_discrete_sequence=['#2C5F2D','#1565C0','#E65100'])
    fig_met.update_traces(textposition='inside', textinfo='percent+label')
    fig_met.update_layout(height=230, margin=dict(t=10,b=30,l=10,r=10),
        showlegend=False, paper_bgcolor='white', font=dict(size=10))
    st.plotly_chart(fig_met, use_container_width=True)

with col8:
    st.markdown('<p class="section-title">🌿 Organik vs Non-Organik</p>', unsafe_allow_html=True)
    org_df = dff.groupby('label_organik').agg(
        jumlah_transaksi=('id_order_detail','count'),
        total_terjual=('jumlah_pembelian','sum')
    ).reset_index()
    fig_org = px.bar(org_df, x='label_organik', y=['jumlah_transaksi','total_terjual'],
        barmode='group',
        color_discrete_map={'jumlah_transaksi':'#2C5F2D','total_terjual':'#A5D6A7'},
        labels={'value':'Jumlah','variable':'Metrik'})
    fig_org.update_layout(height=230, margin=dict(t=10,b=10,l=10,r=10),
        legend=dict(orientation='h', y=-0.25, title='', font=dict(size=9)),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(size=10), xaxis_title='', yaxis_title='')
    fig_org.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig_org, use_container_width=True)

# ── Baris 4: Matriks Korelasi ─────────────────────────────────
st.markdown("---")
col9, col10 = st.columns([2, 3])

with col9:
    st.markdown('<p class="section-title">📐 Matriks Korelasi Pearson</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Korelasi antar variabel numerik</p>', unsafe_allow_html=True)
    num_cols = ['harga_asli_produk','stok_aktual','total_terjual_global',
                'rating_rata_rata_produk','jumlah_pembelian','subtotal_transaksi']
    corr = dff[num_cols].corr().round(2)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdYlGn',
        zmin=-1, zmax=1, aspect='auto')
    fig_corr.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10),
        paper_bgcolor='white', font=dict(size=9))
    st.plotly_chart(fig_corr, use_container_width=True)

with col10:
    st.markdown('<p class="section-title">💰 Scatter: Harga vs Subtotal Transaksi</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Hubungan harga produk dengan nilai transaksi per kategori</p>', unsafe_allow_html=True)
    sample = dff.sample(min(1000, len(dff)), random_state=42)
    fig_scatter = px.scatter(sample, x='harga_asli_produk', y='subtotal_transaksi',
        color='kategori_produk', size='jumlah_pembelian',
        color_discrete_sequence=HIJAU,
        labels={'harga_asli_produk':'Harga Produk (Rp)',
                'subtotal_transaksi':'Subtotal Transaksi (Rp)',
                'kategori_produk':'Kategori'},
        opacity=0.6)
    fig_scatter.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10),
        legend=dict(orientation='h', y=-0.2, title=''),
        paper_bgcolor='white', plot_bgcolor='white', font=dict(size=10))
    fig_scatter.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig_scatter.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#9ca3af; font-size:0.75rem; padding: 0.5rem 0 1rem'>
    🌾 Dashboard AgriMarket · Kelompok 19 · Pengantar Sains Data · IPB University 2026<br>
    Rachel Rehoboth Lumbantobing · Widya Aulianti · Cornelius Bernard Harefa · Fildzah Wahyu Izzati
</div>
""", unsafe_allow_html=True)
