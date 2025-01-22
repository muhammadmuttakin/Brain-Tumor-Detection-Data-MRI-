import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import os

# Fungsi untuk memuat model TensorFlow
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Fungsi untuk memproses gambar yang diunggah
# Fungsi untuk memproses gambar yang diunggah
# Fungsi untuk memproses gambar yang diunggah
def preprocess_image(image):
    # Mengubah ukuran gambar menjadi 224x224
    image = image.resize((224, 224))  # Sesuaikan dengan ukuran yang dibutuhkan oleh model
    
    # Pastikan gambar dalam format RGB (bukan grayscale)
    if image.mode != 'RGB':  # Pastikan gambar dalam format RGB
        image = image.convert('RGB')  # Mengonversi gambar menjadi RGB jika belum
    
    # Mengubah gambar ke format yang diterima oleh model
    img = keras_image.img_to_array(image)  # Mengonversi gambar ke array
    img = np.expand_dims(img, axis=0)      # Menambahkan dimensi batch
    img = img / 255.0                     # Normalisasi gambar
    return img
    
# Fungsi untuk melakukan prediksi dengan model
def predict(image, model):
    image = preprocess_image(image)
    predictions = model.predict(image)  # Prediksi dari model
    predicted_class = np.argmax(predictions, axis=1)  # Ambil kelas dengan probabilitas tertinggi
    return predicted_class[0]

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.set_page_config(page_title="Deteksi Tumor Otak MRI", page_icon="🧠", layout="centered")

    # Judul aplikasi
    st.title("Deteksi Tumor Otak Menggunakan Gambar MRI")
    st.write(
        "Aplikasi ini memungkinkan Anda untuk mengunggah gambar MRI otak dan akan memprediksi apakah terdapat tumor menggunakan model yang sudah dilatih."
    )
    st.write("### Langkah-langkah Penggunaan:")
    st.write(
        "1. Unggah gambar MRI otak dalam format `.jpg`, `.jpeg`, atau `.png`."
    )
    st.write("2. Model akan memproses gambar dan memberikan prediksi.")
    st.write("3. Hasil prediksi akan ditampilkan di bawah gambar yang diunggah.")
    
    st.header("Unggah Gambar MRI")

    uploaded_file = st.file_uploader("Pilih gambar MRI...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Menampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar MRI yang Diupload", use_column_width=True)

        # Path model
        model_path = os.path.join("model", "model.keras")
        
        # Muat model
        model = load_model(model_path)
        
        # Lakukan prediksi
        class_id = predict(image, model)
        
        # Hasil prediksi untuk 4 kelas
        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        if class_id == 0:
            st.success(f"Prediksi: {class_names[0]} (Glioma)")
        elif class_id == 1:
            st.error(f"Prediksi: {class_names[1]} (Meningioma)")
        elif class_id == 2:
            st.success(f"Prediksi: {class_names[2]} (No Tumor)")
        elif class_id == 3:
            st.error(f"Prediksi: {class_names[3]} (Pituitary)")
        else:
            st.warning("Prediksi: Tidak dapat menentukan. Coba unggah gambar lain.")
    
    st.write(
        "### Tentang Dataset"
    )
    st.write(
        "Dataset yang digunakan untuk melatih model ini terdiri dari gambar MRI yang diberi label glioma, meningioma, no tumor, dan pituitary. "
        "Dataset ini tersedia secara publik dan sering digunakan untuk tugas klasifikasi gambar medis."
    )
    st.write(
        "[Sumber Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)"
    )

    st.write(
        "### Cara Kerja Model"
    )
    st.write(
        "Model ini menggunakan teknik pembelajaran mendalam, khususnya **Convolutional Neural Networks (CNN)**, "
        "untuk mendeteksi dan mengklasifikasikan tumor pada gambar MRI. Setelah gambar diunggah, model akan memproses gambar "
        "dan menjalankannya melalui jaringan saraf untuk memprediksi apakah ada tumor atau tidak."
    )

    # Tambahkan informasi kontak
    st.write("### Kontak")
    st.write(
        "Jika Anda memiliki pertanyaan atau saran, Anda bisa menghubungi pengembang aplikasi ini."
    )
    st.write(
        "[Email Pengembang](mailto:email@example.com)"
    )

# Jalankan aplikasi Streamlit
if __name__ == "__main__":
    main()
