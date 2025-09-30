
# -*- coding: utf-8 -*-
"""
Script untuk memproses dan membersihkan missing values dari dataset Titanic.
"""

import pandas as pd

def clean_titanic_data(input_filepath, output_filepath):
    """
    Memuat dataset, menangani missing values berdasarkan strategi yang telah ditentukan,
    dan menyimpan hasilnya ke file baru.

    Args:
        input_filepath (str): Path ke file CSV input (misal: 'train.csv').
        output_filepath (str): Path untuk menyimpan file CSV yang sudah bersih.
    """
    try:
        df = pd.read_csv(input_filepath)
        print(f"Dataset '{input_filepath}' berhasil dimuat.")
        print("-" * 30)

        print("Jumlah missing values SEBELUM pemrosesan:")
        print(df.isnull().sum())
        print("-" * 30)

    except FileNotFoundError:
        print(f"Error: File '{input_filepath}' tidak ditemukan.")
        return

    df_clean = df.copy()

    # pengisian nilai kosong dengan median
    age_median = df_clean['Age'].median()
    df_clean['Age'].fillna(age_median, inplace=True)
    print(f"Kolom 'Age' diisi dengan median: {age_median:.2f}")

    # pengisian nilai kosong dengan modus (nilai paling sering muncul)
    embarked_mode = df_clean['Embarked'].mode()[0]
    df_clean['Embarked'].fillna(embarked_mode, inplace=True)
    print(f"Kolom 'Embarked' diisi dengan modus: '{embarked_mode}'")

    # Penghapusan kolom karena terlalu banyak nilai yang hilang
    df_clean.drop('Cabin', axis=1, inplace=True)
    print("Kolom 'Cabin' telah dihapus.")
    print("-" * 30)

    # Menampilkan jumlah missing values setelah diproses
    print("Jumlah missing values SETELAH pemrosesan:")
    print(df_clean.isnull().sum())
    print("-" * 30)

    df_clean.to_csv(output_filepath, index=False)
    print(f"Data yang sudah bersih berhasil disimpan di '{output_filepath}'")

    return df_clean

if __name__ == "__main__":
    # Menentukan nama file input dan output
    input_file = '../train.csv'
    output_file = 'train_cleaned.csv'

    # Memanggil fungsi untuk membersihkan data
    cleaned_data = clean_titanic_data(input_file, output_file)
