import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri yükleme
df = pd.read_csv('data/MetroPT3.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Zaman eksenini indeks olarak ayarla
df.set_index('timestamp', inplace=True)

# İlgili sütunları seç
columns_to_plot = ['TP2', 'TP3', 'DV_pressure', 'Motor_current', 'Oil_temperature']

# Grafik ayarları
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_plot):
    plt.subplot(len(columns_to_plot), 1, i + 1)
    sns.lineplot(data=df[col][:1000])  # İlk 1000 gözlem ile çizim
    plt.title(col)
    plt.tight_layout()
plt.savefig('eda/sensor_visuals.png')  # Grafik dosyasını kaydeder
print("Grafik 'eda/sensor_visuals.png' olarak kaydedildi.")

