import streamlit as st
import pandas as pd
import numpy as np
import joblib
from google import genai

# 1. Sayfa Düzeni ve Tema Ayarları
st.set_page_config(
    page_title="ChurnAI - Customer Retention Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Stilleri
st.markdown("""
<style>
    /* Ana Arka Plan ve Kart Stilleri */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 5px;
    }
    /* Churn Risk Kutuları */
    .risk-box-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.2) 0%, rgba(185,28,28,0.2) 100%);
        border: 1px solid #ef4444;
        padding: 20px;
        border-radius: 12px;
        color: #fca5a5;
    }
    .risk-box-low {
        background: linear-gradient(135deg, rgba(34,197,94,0.2) 0%, rgba(21,128,61,0.2) 100%);
        border: 1px solid #22c55e;
        padding: 20px;
        border-radius: 12px;
        color: #86efac;
    }
</style>
""", unsafe_allow_html=True)

# 2. Modelleri Yükleme
@st.cache_resource
def load_model():
    model = joblib.load("best_churn_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

try:
    model, model_columns = load_model()
except Exception as e:
    st.error(f"Model Yükleme Hatası: {e}")
    st.stop()

# Header
st.title("⚡ ChurnAI: Müşteri Terk Tahmini & Retention Karar Destek Platformu")
st.caption("XGBoost Makine Öğrenmesi & Google Gemini GenAI ile Müşteri Elde Tutma Stratejileri")
st.divider()

# 3. Sol Menü (Sidebar) & Preset Senaryolar
st.sidebar.image("https://img.icons8.com/color/96/user-menu-male.png", width=64)
st.sidebar.title("👤 Müşteri Profil Girişi")

# Hızlı Senaryo Butonları
st.sidebar.markdown("**⚡ Hızlı Test Senaryoları:**")
col_p1, col_p2 = st.sidebar.columns(2)
if col_p1.button("🔴 Yüksek Risk"):
    st.session_state.update({
        'age': 6, 'spend': 250.0, 'aov': 40.0, 'tickets': 8, 'sat': 1, 'returns': 5, 'days': 65
    })
if col_p2.button("🟢 Düşük Risk"):
    st.session_state.update({
        'age': 36, 'spend': 3400.0, 'aov': 120.0, 'tickets': 1, 'sat': 5, 'returns': 0, 'days': 8
    })

# Form Girdileri (Session State destekli)
customer_age = st.sidebar.number_input("Müşterilik Süresi (Ay)", 1, 120, key='age', value=st.session_state.get('age', 12))
total_spend = st.sidebar.number_input("Toplam Harcama ($)", 0.0, key='spend', value=st.session_state.get('spend', 850.0))
avg_order_val = st.sidebar.number_input("Ort. Sipariş Tutarı ($)", 0.0, key='aov', value=st.session_state.get('aov', 65.0))
support_tickets = st.sidebar.slider("Destek Bilet Sayısı", 0, 20, key='tickets', value=st.session_state.get('tickets', 4))
satisfaction_score = st.sidebar.slider("Memnuniyet Puanı (1-5)", 1, 5, key='sat', value=st.session_state.get('sat', 2))
returns_count = st.sidebar.slider("İade Sayısı", 0, 10, key='returns', value=st.session_state.get('returns', 2))
days_since_last = st.sidebar.number_input("Son Alışverişten Beri Geçen Gün", 0, key='days', value=st.session_state.get('days', 35))

predict_btn = st.sidebar.button("🔮 Risk Skorunu Hesapla", type="primary", use_container_width=True)

# 4. Ana Panel Metrik Kartları
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Toplam Harcama</div><div class="metric-value">${total_spend:,.2f}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Memnuniyet Skor</div><div class="metric-value">⭐ {satisfaction_score}/5</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Destek Talepleri</div><div class="metric-value">🎧 {support_tickets} Adet</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Son Alışveriş</div><div class="metric-value">📅 {days_since_last} Gün Önce</div></div>', unsafe_allow_html=True)

st.write("")

# 5. Tahmin ve Analiz
tab1, tab2 = st.columns([1.1, 1])

# Model Tahmini
input_data = pd.DataFrame(0, index=[0], columns=model_columns)
if 'customer_age_months' in input_data.columns: input_data['customer_age_months'] = customer_age
if 'total_spend_usd' in input_data.columns: input_data['total_spend_usd'] = total_spend
if 'avg_order_value_usd' in input_data.columns: input_data['avg_order_value_usd'] = avg_order_val
if 'support_tickets' in input_data.columns: input_data['support_tickets'] = support_tickets
if 'satisfaction_score' in input_data.columns: input_data['satisfaction_score'] = satisfaction_score
if 'returns_count' in input_data.columns: input_data['returns_count'] = returns_count
if 'days_since_last_purchase' in input_data.columns: input_data['days_since_last_purchase'] = days_since_last

churn_prob = model.predict_proba(input_data)[0][1]
risk_percent = churn_prob * 100

with tab1:
    st.subheader("🎯 ML Model Risk Analizi")
    
    st.write(f"**Tahmini Churn İhtimali:** %{risk_percent:.1f}")
    st.progress(int(risk_percent))
    
    if risk_percent >= 50:
        st.markdown(f"""
        <div class="risk-box-high">
            <h3>⚠️ Yüksek Churn Riski (%{risk_percent:.1f})</h3>
            <p>Müşterinin platformu terk etme olasılığı kritik seviyededir. Memnuniyetsizlik ve destek biletlerinin yoğunluğu ana etkendir.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-box-low">
            <h3>✅ Düşük Churn Riski (%{risk_percent:.1f})</h3>
            <p>Müşteri platformda aktif ve memnun görünmektedir. Sadakat programlarına dahil edilebilir.</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("🤖 GenAI Retention Asistanı")
    
    api_key = st.text_input("🔑 Google Gemini API Key:", type="password", help="Gemini API anahtarınızı buraya girin.")
    
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            
            user_query = st.text_area("Aksiyon Sorusunu Özelleştirin:", value="Bu müşteriyi elde tutmak için ne tür bir indirim, kupon veya aksiyon sunmalıyım?", height=80)
            
            if st.button("🚀 Retention Stratejisi Üret", type="secondary"):
                prompt = f"""
                Sen e-ticaret müşteri elde tutma (retention) uzmanısın.
                Müşteri Profili:
                - Churn Riski: %{risk_percent:.1f}
                - Destek Bilet Sayısı: {support_tickets}
                - Memnuniyet Puanı: {satisfaction_score}/5
                - İade Sayısı: {returns_count}
                - Toplam Harcama: ${total_spend}
                - Son Alışverişten Beri Geçen Gün: {days_since_last}
                
                Soru: {user_query}
                
                Lütfen müşteri temsilcisinin anında uygulayabileceği 3 maddelik somut ve ikna edici aksiyon planı üret.
                """
                with st.spinner("Gemini strateji üretiyor..."):
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                    )
                    st.success("💡 **Önerilen Aksiyon Planı:**")
                    st.write(response.text)
        except Exception as e:
            st.error(f"Gemini API Hatası: {e}")
    else:
        st.info("💡 GenAI asistanından canlı strateji tavsiyesi almak için Gemini API anahtarınızı giriniz.")