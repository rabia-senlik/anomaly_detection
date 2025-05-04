import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri yükle (önceki işlemlerde anomaly sütunları eklenmişti)
df = pd.read_csv('data/MetroPT3.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Anomali çıktısı olan veriyi tekrar ekleyelim
from sklearn.ensemble import IsolationForest

features = ['TP2', 'TP3', 'DV_pressure', 'Motor_current', 'Oil_temperature']
data = df[features].fillna(method='ffill')
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df['anomaly'] = model.fit_predict(data)
df['anomaly_flag'] = df['anomaly'] == -1

# Anomali oranı
anomaly_count = df['anomaly_flag'].sum()
total_count = len(df)
anomaly_ratio = anomaly_count / total_count

print(f"Toplam veri sayısı: {total_count}")
print(f"Anomali sayısı: {anomaly_count}")
print(f"Anomali oranı: %{anomaly_ratio*100:.2f}")

# Aylık dağılım grafiği
monthly_anomalies = df.resample('M').anomaly_flag.sum()

plt.figure(figsize=(10, 4))
sns.barplot(x=monthly_anomalies.index.strftime('%Y-%m'), y=monthly_anomalies.values)
plt.xticks(rotation=45)
plt.title("Aylık Anomali Dağılımı")
plt.xlabel("Ay")
plt.ylabel("Anomali Sayısı")
plt.tight_layout()
plt.savefig('anomaly/monthly_anomaly_distribution.png')
print("Aylık dağılım grafiği 'anomaly/monthly_anomaly_distribution.png' olarak kaydedildi.")
