# Klasifikasi Jenis Sampah Berbasis EfficientNet-B0  dan Komparasi SVM-XGBoost untuk Mendukung Pengelolaan Sampah Perkotaan
This repository was made for Machine Learning final project in State University of Surabaya.

-----

Akhir-akhir ini volume sampah di Indonesia, khususnya perkotaan meningkat secara signifikan. Berdasarkan data Kementerian Lingkungan Hidup (KLH) Sebagian besar sampah kota masih berakhir di tempat pembuangan akhir (TPA) terbuka tanpa melalui proses pemilahan yang memadai, sehingga menghambat potensi pengolahan sampah menjadi energi maupun daur ulang. Permasalahan tersebut mendesak karena kapasitas TPA di Indonesia semakin terbatas, sementara pemilahan sampah secara manual tidak efisien. Pengklasifikasi pemilihan berbasis citra menjadi solusi yang menjanjikan, namun belum banyak diimplementasikan secara nyata di Indonesia.

-----
Dataset didapatkan dari https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2, Problem daridataset ini 

![Distribusi Kelas](/gambar/distribusi_kelas.png)
---------
## Hasil ML

| Model | Precission | Recall | F1-Score |
| ----- | ---------- | ------ | -------- |
| SVM   | 0.94  | 0.94 | 0.94 |
| XGBoost  | 0.91 | 0.91 | 0.91 |
