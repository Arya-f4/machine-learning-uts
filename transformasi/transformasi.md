# Strategi Transformasi Machine Learning

Dokumentasi ini menjelaskan langkah-langkah dalam kode Python untuk melakukan preprocessing data, termasuk penanganan nilai hilang dan normalisasi data menggunakan `MinMaxScaler` dari pustaka `scikit-learn`.

## Tujuan
Kode ini bertujuan untuk:
1. Memuat dataset dari file CSV.
2. Mengidentifikasi kolom numerik dan non-numerik.
3. Menangani nilai hilang pada kolom numerik dengan mengisi menggunakan nilai median.
4. Melakukan normalisasi data numerik menggunakan metode Min-Max Scaling.
5. Menggabungkan kembali kolom numerik yang telah dinormalisasi dengan kolom non-numerik.
6. Menyimpan hasil preprocessing ke file CSV baru.

## Dependensi
Kode ini menggunakan pustaka berikut:
- `pandas`: Untuk manipulasi dan analisis data.
- `sklearn.preprocessing.MinMaxScaler`: Untuk normalisasi data dengan metode Min-Max Scaling.

Pastikan pustaka tersebut telah terinstal menggunakan perintah:
```bash
pip install pandas scikit-learn
```

## Penjelasan Kode

### 1. Memuat Dataset
```python
df = pd.read_csv('../missing-value/train-cleaned-missing-value.csv')
```
- Dataset dimuat dari file CSV bernama `train-cleaned-missing-value.csv` menggunakan fungsi `pd.read_csv` dari pustaka `pandas`.
- Dataset disimpan dalam variabel `df` sebagai DataFrame.

### 2. Mengidentifikasi Kolom Numerik dan Non-Numerik
```python
numerical_cols = [col for col in df.select_dtypes(include=['number']).columns
                  if col not in ['PassengerId', 'Survived']]
non_numerical_cols = df.select_dtypes(exclude=['number']).columns
```
- Kolom numerik diidentifikasi menggunakan `select_dtypes(include=['number'])` untuk memilih kolom dengan tipe data numerik (integer atau float).
- Kolom `PassengerId` dan `Survived` dikecualikan dari daftar kolom numerik karena tidak akan dinormalisasi.
- Kolom non-numerik diidentifikasi menggunakan `select_dtypes(exclude=['number'])` untuk memilih kolom dengan tipe data seperti string atau kategori.

### 3. Menangani Nilai Hilang
```python
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
```
- Nilai hilang pada kolom numerik diisi dengan nilai median dari masing-masing kolom menggunakan metode `fillna` dan `median`.
- Median digunakan karena lebih tahan terhadap outlier dibandingkan rata-rata.

### 4. Normalisasi Data Numerik
```python
scaler = MinMaxScaler()
df_numerical_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_cols]),
                                   columns=numerical_cols)
```
- Objek `MinMaxScaler` diinisialisasi untuk melakukan normalisasi data ke rentang [0, 1].
- Metode `fit_transform` diterapkan pada kolom numerik untuk melakukan normalisasi.
- Hasil normalisasi disimpan dalam DataFrame baru bernama `df_numerical_scaled` dengan nama kolom yang sama.

### 5. Menggabungkan Data
```python
df_scaled = pd.concat([df[['PassengerId', 'Survived'] + list(non_numerical_cols)].reset_index(drop=True),
                       df_numerical_scaled.reset_index(drop=True)], axis=1)
```
- Kolom `PassengerId`, `Survived`, dan semua kolom non-numerik digabungkan kembali dengan kolom numerik yang telah dinormalisasi.
- Fungsi `pd.concat` digunakan untuk menggabungkan secara horizontal (`axis=1`).
- `reset_index(drop=True)` digunakan untuk memastikan indeks antar DataFrame selaras.

### 6. Menyimpan Hasil
```python
df_scaled.to_csv('train_normalized.csv', index=False)
```
- DataFrame yang telah diproses disimpan ke file CSV baru bernama `train_normalized.csv`.
- Parameter `index=False` memastikan indeks tidak disimpan dalam file CSV.

### 7. Pesan Konfirmasi
```python
print("File 'train_normalized.csv' telah berhasil dibuat dengan data yang sudah di-scale Min-Max.")
```
- Menampilkan pesan konfirmasi bahwa file telah berhasil disimpan.

## Output
- File `train_normalized.csv` yang berisi dataset dengan kolom numerik yang telah dinormalisasi dan kolom non-numerik yang tetap seperti aslinya.
- Nilai pada kolom numerik telah diskalakan ke rentang [0, 1] menggunakan Min-Max Scaling.

## Catatan
- Pastikan file `train-cleaned-missing-value.csv` tersedia di direktori yang sesuai (`../missing-value/`).
- Kode ini mengasumsikan dataset memiliki kolom `PassengerId` dan `Survived`. Jika dataset berbeda, sesuaikan nama kolom yang dikecualikan dari normalisasi.
- Jika dataset memiliki banyak nilai hilang, pertimbangkan metode penanganan nilai hilang lain (misalnya, interpolasi) untuk hasil yang lebih akurat.
