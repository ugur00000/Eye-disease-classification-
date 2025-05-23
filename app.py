import streamlit as st
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

# ===============================
# 🔧 AutoKeras özel katmanı
@register_keras_serializable(package="Custom")
class CastToFloat32(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)
# ===============================

# Sınıf adları ve model yolları
IMAGE_SIZE = (32, 32)
CLASS_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

MODEL_FILES = {
    "Neural Network": "neural_network_model.joblib",
    "k-NN": "k-nn_model.joblib",
    "Logistic Regression": "logistic_regression_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "SVC": "svc_model.joblib"
}

FEATURE_MODEL_PATH = "feature_extractor_model.h5"

# ===============================
# Model yükleme fonksiyonları
@st.cache_resource
def load_feature_extractor():
    model = load_model(
        FEATURE_MODEL_PATH,
        compile=False,
        custom_objects={"Custom>CastToFloat32": CastToFloat32}
    )
    return Model(inputs=model.input, outputs=model.get_layer(index=-2).output)

@st.cache_resource
def load_ml_model(path):
    return joblib.load(path)

# Görsel işleme
def preprocess_image(image):
    image = image.resize(IMAGE_SIZE).convert("RGB")
    array = np.array(image) / 255.0
    return array.reshape(1, 32, 32, 3)

# ===============================
# 🌐 Streamlit Arayüzü
st.set_page_config(page_title="Göz Hastalığı Tanı Sistemi", layout="centered")
st.title("👁️ Göz Hastalıklarının Deep Learning ile özelliklerinin çıkartılıp Machine Learning ile Tahmin Sistemi")
st.write("Bir retina görüntüsü yükleyin ve tahmini almak için bir model seçin.")
st.write("Bu sistem şu anda sadece Diyabetik Retinopati, Glokom, Katarakt ve Sağlıklı Gözleri Sınıflandırmaktadır.")

# Görsel yükleme ve model seçimi
uploaded_file = st.file_uploader("📷 Retina görüntüsü yükle", type=["jpg", "jpeg", "png"])
selected_model_name = st.selectbox("🤖 Kullanılacak ML modeli", list(MODEL_FILES.keys()))

# Görsel gösterimi (küçük boyutlu)
if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns([1, 2, 1])  # Ortalanmış 3 sütun

    with col2:
        st.image(image, caption="Yüklenen Görüntü", width=350)


# Tahmin butonu
if uploaded_file and selected_model_name:
    if st.button("🔍 Tahmin Et"):
        try:
            input_tensor = preprocess_image(image)
            feature_extractor = load_feature_extractor()
            features = feature_extractor.predict(input_tensor)

            model_path = MODEL_FILES[selected_model_name]
            ml_model = load_ml_model(model_path)
            prediction = ml_model.predict(features)
            predicted_label = int(prediction[0])

            st.success(f"📌 Tahmin edilen sınıf: **{CLASS_NAMES[predicted_label]}**")

        except Exception as e:
            st.error("❌ Tahmin sırasında bir hata oluştu:")
            st.code(str(e))
