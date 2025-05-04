import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

# Veri yükle
df = pd.read_csv('data/MetroPT3.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

features = ['TP2', 'TP3', 'DV_pressure', 'Motor_current', 'Oil_temperature']
data = df[features].fillna(method='ffill')

# Modeller
models = {
    "Isolation Forest": IsolationForest(contamination=0.01, random_state=42),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.01)
}

results = {}

# Her model için anomali tahmini yap
for name, model in models.items():
    if name == "Local Outlier Factor":
        y_pred = model.fit_predict(data)
    else:
        y_pred = model.fit(data).predict(data)
    
    results[name] = (y_pred == -1).sum()
    print(f"{name} ile bulunan anomali sayısı: {results[name]}")

# Karşılaştırma grafiği
plt.figure(figsize=(8, 4))
plt.bar(results.keys(), results.values(), color=['blue', 'green'])
plt.ylabel("Anomali Sayısı")
plt.title("Algoritmalar Arası Anomali Karşılaştırması")
plt.tight_layout()
plt.savefig("anomaly/model_comparison.png")
print("Model karşılaştırma grafiği 'anomaly/model_comparison.png' olarak kaydedildi.")
