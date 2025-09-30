import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Memuat dataset
df = pd.read_csv('missing-value/train-cleaned-missing-value.csv')

# Mengidentifikasi kolom numerik, kecuali PassengerId dan Survived
numerical_cols = [col for col in df.select_dtypes(include=['number']).columns
                  if col not in ['PassengerId', 'Survived']]
non_numerical_cols = df.select_dtypes(exclude=['number']).columns

# Menangani nilai hilang (jika ada) dengan median untuk kolom numerik
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Menginisialisasi MinMaxScaler
scaler = MinMaxScaler()

# Menerapkan scaler pada kolom numerik
df_numerical_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_cols]),
                                   columns=numerical_cols)

# Menggabungkan kembali kolom numerik yang sudah di-scale dengan kolom non-numerik
df_scaled = pd.concat([df[['PassengerId', 'Survived'] + list(non_numerical_cols)].reset_index(drop=True),
                       df_numerical_scaled.reset_index(drop=True)], axis=1)

# Menyimpan data yang sudah dinormalisasi ke file CSV baru
df_scaled.to_csv('train_normalized.csv', index=False)

print("File 'train_normalized.csv' telah berhasil dibuat dengan data yang sudah di-scale Min-Max.")
