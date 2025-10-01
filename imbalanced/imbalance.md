🚢 Prediksi Keselamatan Penumpang Titanic
Dokumentasi ini adalah panduan lengkap untuk proyek analisis dan pemodelan prediktif pada dataset legendaris Titanic. Skrip utama (imbalance.py) menjalankan alur kerja end-to-end, mulai dari eksplorasi data hingga evaluasi model machine learning yang telah dioptimalkan.

📋 Daftar Isi
Tujuan Proyek

Workflow Proyek

🚀 Cara Menjalankan

⚙️ Penjelasan Detail Kode

📊 Hasil dan Output

🎯 Tujuan Proyek
Proyek ini bertujuan untuk membangun model klasifikasi yang andal untuk menjawab pertanyaan utama: "Faktor apa saja yang paling memengaruhi keselamatan seorang penumpang di kapal Titanic?"

Untuk mencapai tujuan ini, beberapa target spesifik yang harus dicapai adalah:

Memahami Data: Melakukan analisis data eksploratif (EDA) untuk menemukan pola dan wawasan.

Mempersiapkan Data: Membersihkan dan mentransformasi data mentah menjadi format yang siap untuk pemodelan.

Mengatasi Ketidakseimbangan: Menangani masalah data yang tidak seimbang (imbalanced data) pada variabel target (Survived).

Membangun Model: Melatih model RandomForestClassifier yang telah terbukti kuat untuk tugas klasifikasi.

Mengevaluasi Kinerja: Mengukur seberapa baik model dapat memprediksi pada data yang belum pernah dilihat sebelumnya.

🔁 Workflow Proyek
Proyek ini mengikuti alur kerja machine learning yang terstruktur sebagai berikut:

Pemuatan Data

Membaca file train.csv.

Analisis Eksploratif (EDA)

Memvisualisasikan distribusi data dan hubungan antar fitur.

Pra-pemrosesan & Rekayasa Fitur

Mengisi nilai yang hilang, membuat fitur baru (FamilySize, IsAlone), dan melakukan encoding.

Penanganan Data Tidak Seimbang

Menerapkan Random Over-Sampler (ROS) pada data latih.

Pembagian Data

Memisahkan data menjadi set pelatihan (80%) dan pengujian (20%).

Pelatihan Model

Melatih RandomForestClassifier pada data latih yang sudah seimbang.

Evaluasi & Interpretasi

Mengukur performa model pada data uji dan menganalisis hasilnya.

🚀 Cara Menjalankan
1. Prasyarat
Python 3.7 atau versi yang lebih baru.

2. Instalasi Dependensi
Buka terminal dan jalankan perintah berikut untuk menginstal semua library yang dibutuhkan.

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

3. Eksekusi Skrip
Pastikan struktur folder Anda sudah benar. Kemudian, navigasikan ke folder imbalanced dan jalankan skripnya.

# Pindah ke direktori yang benar
cd path/to/your/project/machine-learning-uts/imbalanced

# Jalankan skrip Python
python imbalance.py

⚙️ Penjelasan Detail Kode
<details>
<summary><strong>📄 1. Memuat Dataset</strong></summary>

def main():
    filepath = '../train.csv'
    df = load_data(filepath)

Dataset dimuat dari train.csv yang berada satu direktori di atas skrip. Fungsi load_data menangani proses ini dan juga error jika file tidak ditemukan.

</details>

<details>
<summary><strong>🎨 2. Analisis Data Eksploratif (EDA)</strong></summary>

exploratory_data_analysis(df)

Fungsi ini mencetak informasi dasar DataFrame, menghitung nilai yang hilang, dan yang terpenting, membuat visualisasi untuk memahami data. Plot yang dihasilkan disimpan sebagai eda_plots.png.

</details>

<details>
<summary><strong>🛠️ 3. Pra-pemrosesan Data</strong></summary>

Menangani Nilai Hilang:

df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)

Nilai Age yang hilang diisi dengan median, sedangkan Embarked diisi dengan modus.

Rekayasa Fitur:

df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)

Fitur FamilySize dan IsAlone dibuat untuk menangkap informasi sosial penumpang.

Encoding Variabel Kategorikal:

df_processed = pd.get_dummies(df_processed, columns=['Sex', 'Embarked'], drop_first=True)

Variabel non-numerik diubah menjadi format biner agar dapat dibaca oleh model.

</details>

<details>
<summary><strong>⚖️ 4. Pelatihan dan Evaluasi Model</strong></summary>

Pembagian Data:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

Data dibagi dengan stratify=y untuk memastikan proporsi kelas target tetap sama di set pelatihan dan pengujian.

Penanganan Data Tidak Seimbang (ROS):

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

ROS diterapkan hanya pada data latih untuk mencegah kebocoran informasi ke data uji.

Pelatihan & Evaluasi:

model = RandomForestClassifier(...)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)

Model dilatih pada data yang seimbang dan dievaluasi pada data uji asli.

</details>

📊 Hasil dan Output
Hasil akhir menunjukkan bahwa model memiliki performa yang baik dalam memprediksi keselamatan penumpang, dengan akurasi keseluruhan sekitar 78.77%. Penggunaan ROS berhasil meningkatkan kemampuan model untuk mengidentifikasi kelas minoritas (penumpang yang selamat).

Output yang Dihasilkan
Log Terminal: Menampilkan seluruh proses analisis, termasuk distribusi data dan metrik evaluasi akhir.

eda_plots.png:



Gambar berisi 4 plot dari analisis data eksploratif.

confusion_matrix.png:  Gambar visualisasi matriks konfusi yang merangkum performa prediksi model pada data uji.