import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

st.title("MetroPT3 Anomali Tespiti Uygulaması")

# Veri Yükle
@st.cache_data
def load_data():
    df = pd.read_csv("data/MetroPT3.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.fillna(method='ffill', inplace=True)
    return df

df = load_data()

st.subheader("📄 Veri Önizleme")
st.dataframe(df.head(100))

# Sayısal sütunları al
features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if "Unnamed: 0" in features:
    features.remove("Unnamed: 0")
data = df[features]

# Algoritma seçimi
algo = st.selectbox("🔍 Anomali Algoritması Seçin", ["Isolation Forest", "Local Outlier Factor"])

if st.button("Anomaliyi Tespit Et"):
    if algo == "Isolation Forest":
        model = IsolationForest(contamination=0.01, random_state=42)
        df['anomaly_flag'] = model.fit_predict(data)
    else:
        model = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
        df['anomaly_flag'] = model.fit_predict(data)

    anomalies = df[df['anomaly_flag'] == -1]
    st.success(f"Toplam {len(anomalies)} anomali bulundu. Anomali oranı: %{(len(anomalies)/len(df))*100:.2f}")

    # Zaman içinde anomalileri çiz
    st.subheader("📈 Anomali Zaman Serisi Grafiği (TP2 Değeri)")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['timestamp'], df['TP2'], label='TP2', color='blue')
    ax.scatter(anomalies['timestamp'], anomalies['TP2'], color='red', label='Anomaliler', s=10)
    ax.set_title("TP2 Zaman Serisi ve Anomaliler")
    ax.legend()
    st.pyplot(fig)

    # Aylık dağılım
    st.subheader("🗓️ Aylık Anomali Dağılımı")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    monthly_anomalies = df[df['anomaly_flag'] == -1].resample('M').size()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    monthly_anomalies.plot(kind='bar', ax=ax2)
    ax2.set_title("Aylık Anomali Sayısı")
    ax2.set_ylabel("Anomali Sayısı")
    st.pyplot(fig2)
