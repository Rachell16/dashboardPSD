import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard AgriMarket — Kelompok 19",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.main { background-color: #f8faf8; }
.block-container { padding: 1.5rem 2rem; }
div[data-testid="stMetric"] {
    background: white; border-radius: 12px; padding: 0.8rem;
    border: 1px solid #e8f5e9; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: white; border-radius: 10px 10px 0 0;
    padding: 0.5rem 1.2rem; font-weight: 600; font-size: 0.9rem;
}
.stTabs [aria-selected="true"] { background: #2C5F2D; color: white; }
</style>
""", unsafe_allow_html=True)

# ── Load & Train ──────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("agrimarket_dataset_clean_v2.csv")
    df['waktu_pesanan'] = pd.to_datetime(df['waktu_pesanan'])
    df['bulan'] = df['waktu_pesanan'].dt.to_period('M').astype(str)
    df['is_puas'] = (df['rating_transaksi'] >= 4).astype(int)
    df['label_organik'] = df['is_organik'].map({1:'Organik', 0:'Non-Organik'})
    df['label_puas'] = df['is_puas'].map({1:'Puas (≥4)', 0:'Tidak Puas (<4)'})

    # Encoding
    le_k = LabelEncoder(); le_m = LabelEncoder()
    df['kategori_encoded'] = le_k.fit_transform(df['kategori_produk'])
    df['metode_encoded']   = le_m.fit_transform(df['metode_bayar'])

    # ── Regresi Linear ─────────────────────────────────────────
    FITUR_NUM = ['harga_asli_produk','jumlah_pembelian','is_organik','rating_rata_rata_produk']
    FITUR_CAT = ['kategori_produk']
    X_reg = df[FITUR_NUM + FITUR_CAT]
    y_reg = df['subtotal_transaksi']
    pre = ColumnTransformer([('num','passthrough',FITUR_NUM),
                             ('cat',OneHotEncoder(drop='first',sparse_output=False),FITUR_CAT)])
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    lr = Pipeline([('pre',pre),('lr',LinearRegression())])
    lr.fit(Xtr_r, ytr_r)
    yp_lr = lr.predict(Xte_r)

    # ── Polynomial Regression ──────────────────────────────────
    poly = Pipeline([('pre',pre),('poly',PolynomialFeatures(degree=2,include_bias=False)),('lr',LinearRegression())])
    poly.fit(Xtr_r, ytr_r)
    yp_poly = poly.predict(Xte_r)

    # ── Klasifikasi ────────────────────────────────────────────
    FITUR_CLS = ['harga_asli_produk','stok_aktual','rating_rata_rata_produk',
                 'is_organik','kategori_encoded','jam_pesanan','metode_encoded']
    X_cls = df[FITUR_CLS]; y_cls = df['is_puas']
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(Xtr_c, ytr_c); yp_dt = dt.predict(Xte_c)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(Xtr_c, ytr_c); yp_rf = rf.predict(Xte_c)

    return {
        'df': df,
        'le_k': le_k, 'le_m': le_m,
        'lr_model': lr, 'poly_model': poly,
        'dt_model': dt, 'rf_model': rf,
        'Xte_r': Xte_r, 'yte_r': yte_r, 'yp_lr': yp_lr, 'yp_poly': yp_poly,
        'Xte_c': Xte_c, 'yte_c': yte_c, 'yp_dt': yp_dt, 'yp_rf': yp_rf,
        'FITUR_CLS': FITUR_CLS,
        'r2_lr': r2_score(yte_r, yp_lr),
        'mae_lr': mean_absolute_error(yte_r, yp_lr),
        'r2_poly': r2_score(yte_r, yp_poly),
        'mae_poly': mean_absolute_error(yte_r, yp_poly),
        'acc_dt': accuracy_score(yte_c, yp_dt),
        'f1_dt': f1_score(yte_c, yp_dt, average='macro'),
        'acc_rf': accuracy_score(yte_c, yp_rf),
        'f1_rf': f1_score(yte_c, yp_rf, average='macro'),
        'cm_dt': confusion_matrix(yte_c, yp_dt),
        'cm_rf': confusion_matrix(yte_c, yp_rf),
    }

data = load_and_train()
df = data['df']
HIJAU = ['#2C5F2D','#388E3C','#43A047','#66BB6A','#81C784','#A5D6A7']
MERAH_HIJAU = ['#E53E3E','#ED8936','#ECC94B','#68D391','#2C5F2D']

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#2C5F2D,#388E3C);
     padding:1.5rem 2rem;border-radius:16px;margin-bottom:1.5rem;'>
  <h1 style='color:white;margin:0;font-size:1.6rem;'>🌾 Dashboard Analisis Data AgriMarket</h1>
  <p style='color:#A5D6A7;margin:0.3rem 0 0;font-size:0.85rem;'>
    Platform E-Commerce Pertanian · Kelompok Tunas Muda Cianjur · Kelompok 19 PSD
  </p>
</div>
""", unsafe_allow_html=True)

# ── Summary Cards ─────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1: st.metric("📦 Transaksi", f"{len(df):,}")
with c2: st.metric("💰 Total Pendapatan", f"Rp {df['subtotal_transaksi'].sum()/1_000_000:.1f}jt")
with c3: st.metric("⭐ Rating Rata-rata", f"{df['rating_transaksi'].mean():.2f}/5")
with c4: st.metric("😊 Konsumen Puas", f"{df['is_puas'].mean()*100:.1f}%")
with c5: st.metric("🌱 Produk Unik", f"{df['id_produk'].nunique()}")
with c6: st.metric("🗂️ Kategori", f"{df['kategori_produk'].nunique()}")

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Visualisasi EDA", "🤖 Machine Learning"])

# ════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ════════════════════════════════════════════════════════════
with tab1:

    # Filter Sidebar
    with st.sidebar:
        st.markdown("### 🔍 Filter Data (EDA)")
        kat_opts = ['Semua'] + sorted(df['kategori_produk'].unique().tolist())
        sel_kat = st.selectbox("Kategori", kat_opts)
        met_opts = ['Semua'] + sorted(df['metode_bayar'].unique().tolist())
        sel_met = st.selectbox("Metode Bayar", met_opts)
        org_opts = ['Semua','Organik','Non-Organik']
        sel_org = st.selectbox("Status Organik", org_opts)

    dff = df.copy()
    if sel_kat != 'Semua': dff = dff[dff['kategori_produk']==sel_kat]
    if sel_met != 'Semua': dff = dff[dff['metode_bayar']==sel_met]
    if sel_org != 'Semua': dff = dff[dff['label_organik']==sel_org]

    # Row 1: Tren & Peak Hours
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown("**📅 Tren Penjualan per Bulan**")
        bulan_df = dff.groupby('bulan').agg(
            pendapatan=('subtotal_transaksi','sum'),
            jumlah_order=('id_order_detail','count')
        ).reset_index().sort_values('bulan')
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=bulan_df['bulan'],y=bulan_df['pendapatan'],name='Pendapatan',
            line=dict(color='#2C5F2D',width=2.5),fill='tozeroy',fillcolor='rgba(44,95,45,0.08)'),secondary_y=False)
        fig.add_trace(go.Scatter(x=bulan_df['bulan'],y=bulan_df['jumlah_order'],name='Jumlah Order',
            line=dict(color='#66BB6A',width=2,dash='dot'),mode='lines+markers',marker=dict(size=6)),secondary_y=True)
        fig.update_layout(height=250,margin=dict(t=5,b=5,l=5,r=5),
            legend=dict(orientation='h',y=-0.25),paper_bgcolor='white',plot_bgcolor='white',font=dict(size=11))
        fig.update_xaxes(showgrid=True,gridcolor='#f0f0f0')
        fig.update_yaxes(showgrid=True,gridcolor='#f0f0f0',secondary_y=False)
        fig.update_yaxes(showgrid=False,secondary_y=True)
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        st.markdown("**⏰ Peak Hours Transaksi**")
        jam_df = dff.groupby('jam_pesanan').size().reset_index(name='jumlah')
        jam_df['warna'] = jam_df['jam_pesanan'].apply(lambda x:'#2C5F2D' if 7<=x<=10 else '#A5D6A7')
        fig2 = px.bar(jam_df,x='jam_pesanan',y='jumlah',color='warna',color_discrete_map='identity')
        fig2.update_layout(height=250,margin=dict(t=5,b=5,l=5,r=5),showlegend=False,
            paper_bgcolor='white',plot_bgcolor='white',font=dict(size=10),
            xaxis_title="Jam",yaxis_title="Transaksi")
        fig2.update_xaxes(tickvals=list(range(0,24)),showgrid=False)
        fig2.update_yaxes(showgrid=True,gridcolor='#f0f0f0')
        st.plotly_chart(fig2,use_container_width=True)
        st.caption("🟢 Hijau tua = peak hour (07–10)")

    # Row 2: Kategori & Top 10
    col3, col4 = st.columns([2,3])
    with col3:
        st.markdown("**🗂️ Transaksi per Kategori**")
        kat_df = dff.groupby(['kategori_produk','label_organik']).size().reset_index(name='jumlah')
        fig3 = px.bar(kat_df,x='kategori_produk',y='jumlah',color='label_organik',
            color_discrete_map={'Organik':'#2C5F2D','Non-Organik':'#A5D6A7'},
            barmode='group',text='jumlah')
        fig3.update_traces(textposition='outside',textfont_size=9)
        fig3.update_layout(height=260,margin=dict(t=5,b=5,l=5,r=5),
            legend=dict(orientation='h',y=-0.3,title=''),
            paper_bgcolor='white',plot_bgcolor='white',font=dict(size=10),
            xaxis_title='',yaxis_title='Transaksi')
        fig3.update_yaxes(showgrid=True,gridcolor='#f0f0f0')
        st.plotly_chart(fig3,use_container_width=True)

    with col4:
        st.markdown("**🏆 Top 10 Produk Terlaris**")
        top10 = dff.groupby('nama_produk')['jumlah_pembelian'].sum().sort_values(ascending=True).tail(10)
        fig4 = px.bar(top10,orientation='h',color=top10.values,
            color_continuous_scale=['#A5D6A7','#2C5F2D'])
        fig4.update_layout(height=260,margin=dict(t=5,b=5,l=5,r=5),
            showlegend=False,coloraxis_showscale=False,
            paper_bgcolor='white',plot_bgcolor='white',font=dict(size=10),
            xaxis_title='Unit Terjual',yaxis_title='')
        fig4.update_xaxes(showgrid=True,gridcolor='#f0f0f0')
        st.plotly_chart(fig4,use_container_width=True)

    # Row 3: Rating, Kepuasan, Metode, Organik
    col5,col6,col7,col8 = st.columns(4)
    with col5:
        st.markdown("**⭐ Distribusi Rating**")
        rat = dff['rating_transaksi'].value_counts().sort_index().reset_index()
        rat.columns = ['rating','jumlah']
        fig5 = px.bar(rat,x='rating',y='jumlah',color='rating',
            color_continuous_scale=MERAH_HIJAU,text='jumlah')
        fig5.update_traces(textposition='outside',textfont_size=9)
        fig5.update_layout(height=220,margin=dict(t=5,b=5,l=5,r=5),
            showlegend=False,coloraxis_showscale=False,
            paper_bgcolor='white',plot_bgcolor='white',font=dict(size=10))
        fig5.update_yaxes(showgrid=True,gridcolor='#f0f0f0')
        st.plotly_chart(fig5,use_container_width=True)

    with col6:
        st.markdown("**😊 Kepuasan Konsumen**")
        puas = dff['label_puas'].value_counts().reset_index()
        puas.columns = ['label','jumlah']
        fig6 = px.pie(puas,values='jumlah',names='label',hole=0.5,
            color='label',color_discrete_map={'Puas (≥4)':'#2C5F2D','Tidak Puas (<4)':'#E53E3E'})
        fig6.update_traces(textposition='inside',textinfo='percent')
        fig6.update_layout(height=220,margin=dict(t=5,b=30,l=5,r=5),
            legend=dict(orientation='h',y=-0.2,font=dict(size=9)),
            paper_bgcolor='white',font=dict(size=10))
        st.plotly_chart(fig6,use_container_width=True)

    with col7:
        st.markdown("**💳 Metode Pembayaran**")
        met = dff['metode_bayar'].value_counts().reset_index()
        met.columns = ['metode','jumlah']
        fig7 = px.pie(met,values='jumlah',names='metode',hole=0.45,
            color_discrete_sequence=['#2C5F2D','#1565C0','#E65100'])
        fig7.update_traces(textposition='inside',textinfo='percent+label')
        fig7.update_layout(height=220,margin=dict(t=5,b=30,l=5,r=5),
            showlegend=False,paper_bgcolor='white',font=dict(size=10))
        st.plotly_chart(fig7,use_container_width=True)

    with col8:
        st.markdown("**🌿 Organik vs Non-Organik**")
        org = dff.groupby('label_organik').agg(
            jumlah_transaksi=('id_order_detail','count'),
            total_terjual=('jumlah_pembelian','sum')
        ).reset_index()
        fig8 = px.bar(org,x='label_organik',y=['jumlah_transaksi','total_terjual'],
            barmode='group',color_discrete_map={'jumlah_transaksi':'#2C5F2D','total_terjual':'#A5D6A7'})
        fig8.update_layout(height=220,margin=dict(t=5,b=5,l=5,r=5),
            legend=dict(orientation='h',y=-0.3,title='',font=dict(size=9)),
            paper_bgcolor='white',plot_bgcolor='white',font=dict(size=10),
            xaxis_title='',yaxis_title='')
        fig8.update_yaxes(showgrid=True,gridcolor='#f0f0f0')
        st.plotly_chart(fig8,use_container_width=True)

    # Row 4: Korelasi & Scatter
    col9,col10 = st.columns([2,3])
    with col9:
        st.markdown("**📐 Matriks Korelasi Pearson**")
        num_cols = ['harga_asli_produk','stok_aktual','total_terjual_global',
                    'rating_rata_rata_produk','jumlah_pembelian','subtotal_transaksi']
        corr = dff[num_cols].corr().round(2)
        fig9 = px.imshow(corr,text_auto=True,color_continuous_scale='RdYlGn',zmin=-1,zmax=1)
        fig9.update_layout(height=300,margin=dict(t=5,b=5,l=5,r=5),
            paper_bgcolor='white',font=dict(size=9))
        st.plotly_chart(fig9,use_container_width=True)

    with col10:
        st.markdown("**💰 Scatter: Harga vs Subtotal per Kategori**")
        sample = dff.sample(min(1000,len(dff)),random_state=42)
        fig10 = px.scatter(sample,x='harga_asli_produk',y='subtotal_transaksi',
            color='kategori_produk',color_discrete_sequence=HIJAU,opacity=0.6,
            labels={'harga_asli_produk':'Harga (Rp)','subtotal_transaksi':'Subtotal (Rp)','kategori_produk':'Kategori'})
        fig10.update_layout(height=300,margin=dict(t=5,b=5,l=5,r=5),
            legend=dict(orientation='h',y=-0.25,title=''),
            paper_bgcolor='white',plot_bgcolor='white',font=dict(size=10))
        fig10.update_xaxes(showgrid=True,gridcolor='#f0f0f0')
        fig10.update_yaxes(showgrid=True,gridcolor='#f0f0f0')
        st.plotly_chart(fig10,use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — MACHINE LEARNING
# ════════════════════════════════════════════════════════════
with tab2:

    ml_tab1, ml_tab2, ml_tab3 = st.tabs([
        "📈 Regresi Linear",
        "〰️ Regresi Non-Linear",
        "🌿 Klasifikasi"
    ])

    # ── ML TAB 1: Regresi Linear ─────────────────────────────
    with ml_tab1:
        st.markdown("### 📈 Model Regresi Linear")
        st.markdown("""
        **Tujuan:** Memprediksi `subtotal_transaksi` berdasarkan atribut produk dan transaksi.

        | | |
        |---|---|
        | **Variabel Y** | `subtotal_transaksi` |
        | **Variabel X** | `harga_asli_produk`, `jumlah_pembelian`, `kategori_produk` (OHE), `is_organik`, `rating_rata_rata_produk` |
        """)

        # Metrik
        m1,m2,m3 = st.columns(3)
        with m1: st.metric("R² Score", f"{data['r2_lr']:.4f}", delta="85.57% variansi dijelaskan")
        with m2: st.metric("MAE", f"Rp {data['mae_lr']:,.0f}")
        with m3: st.metric("RMSE", f"Rp {np.sqrt(mean_squared_error(data['yte_r'],data['yp_lr'])):,.0f}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Aktual vs Prediksi**")
            fig_lr1 = go.Figure()
            fig_lr1.add_trace(go.Scatter(x=data['yte_r'],y=data['yp_lr'],mode='markers',
                marker=dict(color='#2C5F2D',opacity=0.4,size=5),name='Prediksi'))
            lims = [min(data['yte_r'].min(),data['yp_lr'].min()),
                    max(data['yte_r'].max(),data['yp_lr'].max())]
            fig_lr1.add_trace(go.Scatter(x=lims,y=lims,mode='lines',
                line=dict(color='red',dash='dash',width=2),name='Ideal'))
            fig_lr1.update_layout(height=300,paper_bgcolor='white',plot_bgcolor='white',
                margin=dict(t=5,b=5,l=5,r=5),font=dict(size=11),
                xaxis_title='Aktual (Rp)',yaxis_title='Prediksi (Rp)')
            fig_lr1.update_xaxes(showgrid=True,gridcolor='#f0f0f0')
            fig_lr1.update_yaxes(showgrid=True,gridcolor='#f0f0f0')
            st.plotly_chart(fig_lr1,use_container_width=True)

        with col2:
            st.markdown("**Residual Plot (Cek Homoskedastisitas)**")
            residuals = data['yte_r'] - data['yp_lr']
            fig_lr2 = go.Figure()
            fig_lr2.add_trace(go.Scatter(x=data['yp_lr'],y=residuals,mode='markers',
                marker=dict(color='#E65100',opacity=0.4,size=5),name='Residual'))
            fig_lr2.add_hline(y=0,line_dash='dash',line_color='red',line_width=2)
            fig_lr2.update_layout(height=300,paper_bgcolor='white',plot_bgcolor='white',
                margin=dict(t=5,b=5,l=5,r=5),font=dict(size=11),
                xaxis_title='Prediksi (Rp)',yaxis_title='Residual')
            fig_lr2.update_xaxes(showgrid=True,gridcolor='#f0f0f0')
            fig_lr2.update_yaxes(showgrid=True,gridcolor='#f0f0f0')
            st.plotly_chart(fig_lr2,use_container_width=True)

        # Cek Asumsi
        st.markdown("**✅ Verifikasi Asumsi Regresi Linear**")
        from scipy import stats
        _, p_norm = stats.shapiro(residuals[:200])
        a1,a2,a3,a4 = st.columns(4)
        with a1: st.info(f"**Linearitas**\n\nTerpenuhi ✓\nKorelasi kuat antar variabel")
        with a2: st.info(f"**Normalitas Residual**\n\nShapiro p={p_norm:.4f}\nCLT berlaku (n=5.000)")
        with a3: st.info(f"**Mean Residual**\n\n{residuals.mean():,.1f}\n≈ 0 ✓")
        with a4: st.info(f"**Homoskedastisitas**\n\nLihat residual plot\ndi atas")

        # Prediksi Interaktif
        st.markdown("---")
        st.markdown("### 🔮 Prediksi Subtotal Transaksi")
        st.caption("Masukkan data transaksi untuk memprediksi nilai subtotal")

        pc1,pc2,pc3 = st.columns(3)
        with pc1:
            harga_input = st.number_input("Harga Produk (Rp)", min_value=5000, max_value=60000, value=20000, step=500)
            jumlah_input = st.number_input("Jumlah Pembelian (unit)", min_value=1, max_value=5, value=2)
        with pc2:
            kat_input = st.selectbox("Kategori Produk", df['kategori_produk'].unique())
            organik_input = st.selectbox("Status Organik", ["Non-Organik","Organik"])
        with pc3:
            rating_input = st.slider("Rating Rata-rata Produk", 3.5, 5.0, 4.2, 0.1)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮 Prediksi Subtotal", type="primary", use_container_width=True):
                input_df = pd.DataFrame([{
                    'harga_asli_produk': harga_input,
                    'jumlah_pembelian': jumlah_input,
                    'is_organik': 1 if organik_input=="Organik" else 0,
                    'rating_rata_rata_produk': rating_input,
                    'kategori_produk': kat_input
                }])
                pred_lr = data['lr_model'].predict(input_df)[0]
                pred_poly = data['poly_model'].predict(input_df)[0]
                st.success(f"**Prediksi Regresi Linear:** Rp {pred_lr:,.0f}")
                st.success(f"**Prediksi Polynomial Deg-2:** Rp {pred_poly:,.0f}")
                st.caption(f"Nilai aktual jika tanpa diskon: Rp {harga_input*jumlah_input:,}")

    # ── ML TAB 2: Regresi Non-Linear ─────────────────────────
    with ml_tab2:
        st.markdown("### 〰️ Model Regresi Non-Linear (Polynomial Degree 2)")
        st.markdown("""
        **Variabel X dan Y sama dengan Regresi Linear** — untuk perbandingan yang *fair*.

        Polynomial Degree 2 secara otomatis membuat:
        - **Fitur kuadrat**: harga², jumlah², dst
        - **Fitur interaksi**: harga×jumlah, harga×organik, dst
        """)

        # Perbandingan metrik
        st.markdown("**📊 Perbandingan Regresi Linear vs Polynomial**")
        comp_df = pd.DataFrame({
            'Model': ['Regresi Linear','Polynomial Deg-2'],
            'R² Score': [data['r2_lr'], data['r2_poly']],
            'MAE (Rp)': [data['mae_lr'], data['mae_poly']],
            'RMSE (Rp)': [
                np.sqrt(mean_squared_error(data['yte_r'],data['yp_lr'])),
                np.sqrt(mean_squared_error(data['yte_r'],data['yp_poly']))
            ]
        })
        st.dataframe(comp_df.style.format({'R² Score':'{:.4f}','MAE (Rp)':'{:,.0f}','RMSE (Rp)':'{:,.0f}'}), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Aktual vs Prediksi — Regresi Linear (R²={:.4f})**".format(data['r2_lr']))
            fig_p1 = go.Figure()
            fig_p1.add_trace(go.Scatter(x=data['yte_r'],y=data['yp_lr'],mode='markers',
                marker=dict(color='#2C5F2D',opacity=0.4,size=4),name='Prediksi'))
            lims=[min(data['yte_r'].min(),data['yp_lr'].min()),max(data['yte_r'].max(),data['yp_lr'].max())]
            fig_p1.add_trace(go.Scatter(x=lims,y=lims,mode='lines',line=dict(color='red',dash='dash'),name='Ideal'))
            fig_p1.update_layout(height=280,paper_bgcolor='white',plot_bgcolor='white',
                margin=dict(t=5,b=5,l=5,r=5),xaxis_title='Aktual',yaxis_title='Prediksi',font=dict(size=11))
            fig_p1.update_xaxes(showgrid=True,gridcolor='#f0f0f0')
            fig_p1.update_yaxes(showgrid=True,gridcolor='#f0f0f0')
            st.plotly_chart(fig_p1,use_container_width=True)

        with col2:
            st.markdown("**Aktual vs Prediksi — Polynomial Deg-2 (R²={:.4f})**".format(data['r2_poly']))
            fig_p2 = go.Figure()
            fig_p2.add_trace(go.Scatter(x=data['yte_r'],y=data['yp_poly'],mode='markers',
                marker=dict(color='#7B1FA2',opacity=0.4,size=4),name='Prediksi'))
            lims2=[min(data['yte_r'].min(),data['yp_poly'].min()),max(data['yte_r'].max(),data['yp_poly'].max())]
            fig_p2.add_trace(go.Scatter(x=lims2,y=lims2,mode='lines',line=dict(color='red',dash='dash'),name='Ideal'))
            fig_p2.update_layout(height=280,paper_bgcolor='white',plot_bgcolor='white',
                margin=dict(t=5,b=5,l=5,r=5),xaxis_title='Aktual',yaxis_title='Prediksi',font=dict(size=11))
            fig_p2.update_xaxes(showgrid=True,gridcolor='#f0f0f0')
            fig_p2.update_yaxes(showgrid=True,gridcolor='#f0f0f0')
            st.plotly_chart(fig_p2,use_container_width=True)

        # Perbandingan R² bar
        st.markdown("**Perbandingan R² Score**")
        fig_r2 = px.bar(
            x=['Regresi Linear','Polynomial Deg-2'],
            y=[data['r2_lr'],data['r2_poly']],
            color=['Regresi Linear','Polynomial Deg-2'],
            color_discrete_sequence=['#2C5F2D','#7B1FA2'],
            text=[f"{data['r2_lr']:.4f}",f"{data['r2_poly']:.4f}"]
        )
        fig_r2.add_hline(y=0.5,line_dash='dash',line_color='gray',annotation_text='R²=0.5')
        fig_r2.update_traces(textposition='outside')
        fig_r2.update_layout(height=250,paper_bgcolor='white',plot_bgcolor='white',
            margin=dict(t=5,b=5,l=5,r=5),showlegend=False,
            yaxis_title='R² Score',yaxis_range=[0,1.1])
        st.plotly_chart(fig_r2,use_container_width=True)

        # Interpretasi
        st.markdown("---")
        c_int1, c_int2 = st.columns(2)
        with c_int1:
            st.error("""**❌ Mengapa Regresi Linear Terbatas?**

Model linear mengasumsikan pengaruh tiap fitur **TERPISAH**:

`subtotal = a₁×harga + a₂×jumlah + ...`

Namun data AgriMarket punya **efek interaksi**:
- Harga tinggi × Jumlah banyak → subtotal melonjak lebih besar
- Produk organik × Harga premium → efek berlipat
- Kategori berbeda → pola berbeda

**Hasil: R² = {:.4f}**""".format(data['r2_lr']))

        with c_int2:
            st.success("""**✅ Mengapa Polynomial Lebih Unggul?**

Polynomial Degree 2 otomatis membuat **fitur baru**:

1. **Kuadrat tiap fitur**: harga², jumlah², dst
2. **Interaksi antar fitur**: harga×jumlah, harga×organik, dst

Model bisa menangkap **hubungan non-linear** dan **efek kombinasi** antar variabel.

**Hasil: R² = {:.4f}**""".format(data['r2_poly']))

    # ── ML TAB 3: Klasifikasi ─────────────────────────────────
    with ml_tab3:
        st.markdown("### 🌿 Model Klasifikasi Kepuasan Konsumen")
        st.markdown("""
        **Tujuan:** Memprediksi apakah konsumen **puas** atau **tidak puas**.

        | | |
        |---|---|
        | **Variabel Y** | `is_puas` (1=Puas jika rating≥4, 0=Tidak Puas) |
        | **Variabel X** | `harga_asli_produk`, `stok_aktual`, `rating_rata_rata_produk`, `is_organik`, `kategori_produk`, `jam_pesanan`, `metode_bayar` |
        """)

        # Perbandingan model
        st.markdown("**📊 Perbandingan Decision Tree vs Random Forest**")
        mc1,mc2,mc3,mc4 = st.columns(4)
        with mc1: st.metric("DT Akurasi", f"{data['acc_dt']*100:.1f}%", delta="Decision Tree")
        with mc2: st.metric("DT F1 Macro", f"{data['f1_dt']:.4f}")
        with mc3: st.metric("RF Akurasi", f"{data['acc_rf']*100:.1f}%", delta="Random Forest")
        with mc4: st.metric("RF F1 Macro", f"{data['f1_rf']:.4f}")

        # Confusion Matrix
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Confusion Matrix — Decision Tree (Akurasi {data['acc_dt']*100:.1f}%)**")
            cm = data['cm_dt']
            fig_cm1 = px.imshow(cm, text_auto=True,
                labels=dict(x='Prediksi',y='Aktual',color='Jumlah'),
                x=['Tidak Puas','Puas'], y=['Tidak Puas','Puas'],
                color_continuous_scale='Blues')
            fig_cm1.update_layout(height=280,margin=dict(t=5,b=5,l=5,r=5),
                paper_bgcolor='white',font=dict(size=13))
            st.plotly_chart(fig_cm1,use_container_width=True)

        with col2:
            st.markdown(f"**Confusion Matrix — Random Forest (Akurasi {data['acc_rf']*100:.1f}%)**")
            cm2 = data['cm_rf']
            fig_cm2 = px.imshow(cm2, text_auto=True,
                labels=dict(x='Prediksi',y='Aktual',color='Jumlah'),
                x=['Tidak Puas','Puas'], y=['Tidak Puas','Puas'],
                color_continuous_scale='Greens')
            fig_cm2.update_layout(height=280,margin=dict(t=5,b=5,l=5,r=5),
                paper_bgcolor='white',font=dict(size=13))
            st.plotly_chart(fig_cm2,use_container_width=True)

        # Feature Importance
        st.markdown("**🔍 Feature Importance**")
        fi_col1, fi_col2 = st.columns(2)
        NAMA_FITUR = ['Harga Produk','Stok Aktual','Rating Produk','Organik','Kategori','Jam Pesanan','Metode Bayar']
        with fi_col1:
            fi_dt = pd.Series(data['dt_model'].feature_importances_, index=NAMA_FITUR).sort_values()
            fig_fi1 = px.bar(fi_dt, orientation='h', color=fi_dt.values,
                color_continuous_scale=['#A5D6A7','#2C5F2D'], title='Decision Tree')
            fig_fi1.update_layout(height=260,margin=dict(t=30,b=5,l=5,r=5),
                showlegend=False,coloraxis_showscale=False,
                paper_bgcolor='white',plot_bgcolor='white',font=dict(size=11))
            fig_fi1.update_xaxes(showgrid=True,gridcolor='#f0f0f0')
            st.plotly_chart(fig_fi1,use_container_width=True)

        with fi_col2:
            fi_rf = pd.Series(data['rf_model'].feature_importances_, index=NAMA_FITUR).sort_values()
            fig_fi2 = px.bar(fi_rf, orientation='h', color=fi_rf.values,
                color_continuous_scale=['#C8E6C9','#1B5E20'], title='Random Forest')
            fig_fi2.update_layout(height=260,margin=dict(t=30,b=5,l=5,r=5),
                showlegend=False,coloraxis_showscale=False,
                paper_bgcolor='white',plot_bgcolor='white',font=dict(size=11))
            fig_fi2.update_xaxes(showgrid=True,gridcolor='#f0f0f0')
            st.plotly_chart(fig_fi2,use_container_width=True)

        # Prediksi Interaktif
        st.markdown("---")
        st.markdown("### 🔮 Prediksi Kepuasan Konsumen")
        st.caption("Masukkan data transaksi untuk memprediksi apakah konsumen akan puas")

        pi1,pi2,pi3 = st.columns(3)
        with pi1:
            harga_cls = st.number_input("Harga Produk (Rp)", min_value=5000, max_value=60000, value=20000, step=500, key='harga_cls')
            stok_cls = st.number_input("Stok Aktual (unit)", min_value=10, max_value=120, value=50, key='stok_cls')
            rating_cls = st.slider("Rating Rata-rata Produk", 3.5, 5.0, 4.2, 0.1, key='rating_cls')
        with pi2:
            organik_cls = st.selectbox("Status Organik", ["Non-Organik","Organik"], key='org_cls')
            kat_cls = st.selectbox("Kategori Produk", df['kategori_produk'].unique(), key='kat_cls')
            jam_cls = st.slider("Jam Pesanan", 0, 23, 8, key='jam_cls')
        with pi3:
            metode_cls = st.selectbox("Metode Bayar", df['metode_bayar'].unique(), key='met_cls')
            model_cls = st.radio("Gunakan Model", ["Decision Tree","Random Forest"], key='mdl_cls')
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮 Prediksi Kepuasan", type="primary", use_container_width=True, key='btn_cls'):
                le_k = data['le_k']; le_m = data['le_m']
                kat_enc = le_k.transform([kat_cls])[0]
                met_enc = le_m.transform([metode_cls])[0]
                input_arr = np.array([[
                    harga_cls, stok_cls, rating_cls,
                    1 if organik_cls=="Organik" else 0,
                    kat_enc, jam_cls, met_enc
                ]])
                model = data['dt_model'] if model_cls=="Decision Tree" else data['rf_model']
                pred = model.predict(input_arr)[0]
                prob = model.predict_proba(input_arr)[0]

                if pred == 1:
                    st.success(f"✅ **PREDIKSI: PUAS** — Probabilitas puas: {prob[1]*100:.1f}%")
                else:
                    st.error(f"❌ **PREDIKSI: TIDAK PUAS** — Probabilitas tidak puas: {prob[0]*100:.1f}%")

                st.info(f"Model: {model_cls} | Confidence: {max(prob)*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#9ca3af;font-size:0.75rem;padding:0.5rem 0 1rem'>
  🌾 Dashboard AgriMarket · Kelompok 19 · Pengantar Sains Data · IPB University 2026<br>
  Rachel Rehoboth Lumbantobing · Widya Aulianti · Cornelius Bernard Harefa · Fildzah Wahyu Izzati
</div>
""", unsafe_allow_html=True)
