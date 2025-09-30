# Strategi Penanganan Missing Values untuk Machine Learning
Dokumen ini menjelaskan logika dan pendekatan yang digunakan untuk membersihkan missing values dari dataset Titanic (train.csv).

## 1. Analisis Awal
Langkah pertama adalah mengidentifikasi kolom mana saja yang memiliki data hilang dan seberapa parah. Dengan menggunakan df.isnull().sum(), kita dapat melihat jumlah nilai kosong di setiap kolom.

Kolom yang menjadi fokus utama biasanya adalah:

- Age: Memiliki sejumlah nilai kosong yang signifikan, tetapi merupakan fitur penting.

- Cabin: Sebagian besar nilainya kosong.

- Embarked: Hanya memiliki sedikit sekali nilai kosong.

## 2. Rencana Penanganan per Kolom
Strategi yang dipilih harus didasarkan pada tipe data dan dampak kolom tersebut terhadap model.

### a. Kolom Age (Numerik)
- Masalah: Age adalah fitur demografis yang sangat penting untuk memprediksi kelangsungan hidup. Menghapus baris dengan Age yang kosong akan menghilangkan banyak informasi berharga.

- Solusi: Imputasi menggunakan Median. Median (nilai tengah) adalah pilihan yang lebih baik daripada rata-rata (mean) karena distribusi usia penumpang mungkin tidak normal dan bisa memiliki outlier (misalnya, beberapa penumpang yang sangat tua). Median lebih tahan terhadap nilai-nilai ekstrem ini.

## b. Kolom Embarked (Kategorikal)
Masalah: Kolom ini menunjukkan pelabuhan tempat penumpang naik. Hanya ada beberapa nilai yang hilang (biasanya 2).

- Solusi: Imputasi menggunakan Modus. Modus adalah nilai yang paling sering muncul. Mengisi nilai kosong dengan pelabuhan yang paling umum adalah pendekatan yang aman dan logis, karena kemungkinan besar penumpang tersebut memang berangkat dari sana.

## c. Kolom Cabin (Alfanumerik/Kategorikal)
- Masalah: Kolom Cabin memiliki persentase missing values yang sangat tinggi (lebih dari 75%).

- Solusi: Hapus Kolom (Drop Column). Mencoba mengisi data yang hilang sebanyak ini akan lebih banyak menciptakan noise (gangguan) daripada sinyal yang berguna. Informasi yang bisa didapat dari kolom ini sangat terbatas. Oleh karena itu, strategi paling aman dan paling efektif adalah menghapus kolom ini sepenuhnya dari dataset.

## 3. Kesimpulan
Dengan menerapkan strategi ini, kita akan mendapatkan dataset yang:

1. Tidak lagi memiliki missing values.

2. Mempertahankan sebagian besar baris data asli.

3. Siap untuk tahap selanjutnya seperti feature engineering dan pelatihan model.
