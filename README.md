Spotify Data Mining: Analyzing Popularity & Song Clustering

Proyek ini berfokus pada analisis mendalam terhadap dataset lagu Spotify untuk memahami karakteristik yang mendorong sebuah lagu menjadi populer. Dengan memanfaatkan teknik Data Mining, tim kami mengidentifikasi pola tersembunyi dalam fitur akustik dan melakukan pengelompokan lagu untuk memberikan wawasan strategis bagi pemangku kepentingan di industri musik.

👥 Tim Peneliti (Kelompok 2)
Prana Ichlasul Kautsar - Leader 
Harry Prasad - Member 
Nicodemus Benaya Gavia L - Member 
Fredy Arya Hutama - Member 
Ignatius Kevin - Member 
Muhamad Affan Febryan T - Member 

🎯 Tujuan Proyek
Identifikasi Karakteristik: Menganalisis fitur menonjol seperti genre, total stream, dan pengikut artis pada daftar Top 200 Spotify.
Clustering Lagu: Mengelompokkan lagu berdasarkan kombinasi fitur akustik dan genre menggunakan metode unsupervised learning.
Wawasan Strategis: Menemukan pola signifikan yang dapat digunakan untuk sistem rekomendasi dan strategi pemasaran yang lebih akurat.

🛠️ Metodologi & Fitur Teknis
1. Preprocessing Data
Data Cleaning: Penanganan missing values dengan imputasi mean, penghapusan data duplikat, dan penanganan outlier menggunakan teknik Winsorization.
Scaling: Menggunakan StandardScaler untuk menormalisasi fitur akustik sebelum proses clustering.

2. Unsupervised Learning (Clustering)
Algoritma: K-Means Clustering.
Penentuan Cluster: Menggunakan Elbow Method untuk menentukan jumlah cluster optimal (K=3 dipilih untuk demonstrasi).
Evaluasi: Silhouette Score sebesar 0.161 untuk mengukur kualitas pengelompokan.

3. Supervised Learning (Classification)
Kami membandingkan dua model untuk memprediksi genre berdasarkan fitur musik:
Naive Bayes: Model awal dengan tingkat akurasi sekitar 0.40.
Random Forest: Model yang ditingkatkan dengan engineered features, mencapai akurasi signifikan sebesar 0.99.

📊 Hasil & Temuan
Dominasi Genre: Dataset menunjukkan bahwa genre Pop merupakan genre yang paling frekuen muncul di Top 200, diikuti oleh Dance dan Hip-hop.
Fitur Akustik: Melalui analisis cluster, ditemukan kelompok lagu dengan karakteristik spesifik, misalnya Cluster 2 yang memiliki rata-rata Danceability dan Energy paling tinggi dibandingkan cluster lainnya.

💻 Tech Stack
Bahasa Pemrograman: Python.
Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn.
Tools: Jupyter Notebook / Python Dashboard.
