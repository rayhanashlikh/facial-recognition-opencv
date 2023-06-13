# Facial Detection and Recognition Using OpenCV

### Cara Run Training Model
1. Jalankan command `py faces-train.py` untuk melakukan training data baru
2. Apabila ingin menambah dataset gambar baru, buat terlebih dahulu folder baru didalam folder `images` 
3. Buat nama folder nama dari orang yang akan dijadikan dataset, misal nama folder *barack-obama* (catatan, nama tidak boleh pakai spasi, cukup pakai karakter spesial)
4. Kemudian letakkan gambar dataset didalamnya, dengan catatan nama gambar telah diubah menjadi angka, misal *1.jpg*, *2.jpg*, *3.jpg*, ....
5. Setelah dilakukan penambahan dataset, harus dilakukan ulang training data dengan command `py faces-train.py`

### Cara Run Program Face Detection dan Recognition
1. Jalankan command `py faces.py`, dengan catatan kondisi webcam sudah siap digunakan
2. Akan muncul jendela kamera baru, proses deteksi dan pengenalan wajah sudah siap dilakukan
