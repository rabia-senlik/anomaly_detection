import pandas as pd

# Veri yolu
file_path = 'data/MetroPT3.csv'

# Veriyi yükle
df = pd.read_csv(file_path)

# 1. Boyut bilgisi
print("Veri boyutu (satır, sütun):", df.shape)

# 2. Eksik veri kontrolü
print("\nEksik veri sayısı:\n", df.isnull().sum())

# 3. Zaman damgasını datetime formatına çevir
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 4. Sayısal sütunlar için özet istatistikler
print("\nSayısal sütun istatistikleri:\n", df.describe())

# İsteğe bağlı: veri türlerini de incele
print("\nVeri türleri:\n", df.dtypes)

