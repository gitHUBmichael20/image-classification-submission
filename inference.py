import tensorflow as tf
import numpy as np
from PIL import Image
import os

class LandscapeInference:
    """Kelas untuk melakukan prediksi pada gambar lanskap"""
    
    def __init__(self, model_path='saved_model', labels_path='tflite/label.txt'):
        """
        Inisialisasi model dan label
        
        Parameters:
        - model_path: Lokasi model yang sudah disimpan
        - labels_path: Lokasi file label
        """
        # Memuat model tersimpan
        self.model = tf.saved_model.load(model_path)
        
        # Memuat label kelas
        self.labels = self._load_labels(labels_path)
    
    def _load_labels(self, labels_path):
        """
        Membaca label-label kelas dari file
        
        Returns:
        - Daftar label kelas
        """
        with open(labels_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def predict(self, image_path, top_k=3):
        """
        Melakukan prediksi pada gambar
        
        Parameters:
        - image_path: Path gambar yang akan diprediksi
        - top_k: Jumlah prediksi teratas yang ingin ditampilkan
        
        Returns:
        - Daftar prediksi dengan kelas dan probabilitas
        """
        # Praproses gambar
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Sesuaikan dengan ukuran input model
        img_array = np.array(img) / 255.0  # Normalisasi
        img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch
        
        # Prediksi
        predictions = self.model(img_array)
        probabilities = predictions.numpy()[0]
        
        # Ambil top-k prediksi
        top_indices = probabilities.argsort()[::-1][:top_k]
        
        # Susun hasil prediksi
        results = [
            {
                'class': self.labels[idx],
                'confidence': float(probabilities[idx]) * 100
            } 
            for idx in top_indices
        ]
        
        return results

def main():
    """Contoh penggunaan inference script"""
    # Inisialisasi inference
    inference = LandscapeInference()
    
    # Contoh prediksi pada gambar
    test_image = 'image_dataset/seg_test/seg_test/mountain/20058.jpg'
    
    print(f"Prediksi untuk gambar: {test_image}")
    predictions = inference.predict(test_image)
    
    # Tampilkan prediksi
    for pred in predictions:
        print(f"Kelas: {pred['class']}, Kepercayaan: {pred['confidence']:.2f}%")

if __name__ == '__main__':
    main()