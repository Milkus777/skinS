import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

st.set_page_config(page_title="Диагностика кожных заболеваний", layout="centered")
st.title("🔬 Диагностика кожных заболеваний")
st.markdown("Загрузите изображение кожи, и модель предскажет наиболее вероятное заболевание.")

@st.cache_resource
def load_keras_model():
    return load_model('model.keras')

model = load_keras_model()

class_names = {
    'nv': 'Меланоцитарные невусы',
    'mel': 'Меланома',
    'bkl': 'Доброкачественные кератозоподобные поражения',
    'bcc': 'Базальноклеточная карцинома',
    'akiec': 'Актинические кератозы',
    'vasc': 'Сосудистые поражения',
    'df': 'Дерматофиброма'
}

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Загруженное изображение', use_column_width=False, width=250)

    with open(os.path.join("temp.jpg"), "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = image.load_img(uploaded_file, target_size=(75, 100))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_label = list(class_names.keys())[np.argmax(predictions)]
    confidence = float(predictions[np.argmax(predictions)])

    st.success(f"✅ Прогноз: **{class_names[predicted_label]}**")
    st.info(f"Надёжность: **{round(confidence * 100, 2)}%**")

    os.remove("temp.jpg")
