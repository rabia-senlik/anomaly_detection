import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os

# Klasör varsa oluştur
os.makedirs('anomaly', exist_ok=True)

# Veri yükle
df = pd.read_csv('data/MetroPT3.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Anomali tespitinde kullanılacak sütunlar
features = ['TP2', 'TP3', 'DV_pressure', 'Motor_current', 'Oil_temperature']
data = df[features].fillna(method='ffill')

# Modeli tanımla ve eğit
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df['anomaly'] = model.fit_predict(data)

# Anomaliler -1 olarak etiketlenir
df['anomaly_flag'] = df['anomaly'] == -1

# Örnek bir grafik çiz (örneğin 'TP2' için)
plt.figure(figsize=(15, 5))
plt.plot(df.index[:2000], df['TP2'][:2000], label='TP2', color='blue')
plt.scatter(df.index[:2000][df['anomaly_flag'][:2000]], df['TP2'][:2000][df['anomaly_flag'][:2000]],
            color='red', label='Anomaly', s=10)
plt.legend()
plt.title("TP2 Sensör Verisi ve Anomaliler (İlk 2000 örnek)")
plt.tight_layout()
plt.savefig('anomaly/tp2_anomalies.png')
print("Anomali grafiği 'anomaly/tp2_anomalies.png' olarak kaydedildi.")
