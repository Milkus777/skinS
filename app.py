import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Настройки страницы
st.set_page_config(page_title="Диагностика кожных заболеваний", page_icon="🩺", layout="centered")

# Загрузка модели
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Словарь классов (взят из ноутбука)
class_names_ru = [
    'Меланоцитарные невусы',
    'Меланома',
    'Доброкачественные кератозоподобные поражения',
    'Базальноклеточная карцинома',
    'Актинические кератозы',
    'Сосудистые поражения',
    'Дерматофиброма'
]

# Баннер / приветствие
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🩺 Диагностика кожных заболеваний</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Загрузите фото кожи, чтобы получить диагноз от нейросети.</p>", unsafe_allow_html=True)

# Контейнер загрузки изображения
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((100, 75))  # Точный размер: ширина=100, высота=75
    img_array = np.array(image) / 255.0  # Нормализация
    img_array = np.expand_dims(img_array, axis=0).astype(model.input.dtype)

    with st.spinner('🧠 Обрабатываем изображение...'):
        predictions = model.predict(img_array)[0]
    
    # Отображение результата
    st.image(image, caption="Ваше изображение", use_column_width=True)

    st.subheader("🔍 Результат анализа:")
    max_prob = 0
    predicted_class = ""

    prediction_text = ""
    for i, class_name in enumerate(class_names_ru):
        prob_percent = round(predictions[i] * 100, 2)
        if prob_percent > max_prob:
            max_prob = prob_percent
            predicted_class = class_name
        prediction_text += f"- {class_name}: {prob_percent}%\n"

    st.markdown(f"**С уверенностью {max_prob}% это {predicted_class}**")
    st.text_area("Подробности:", value=prediction_text, height=200, disabled=True)

else:
    st.info("📸 Пожалуйста, загрузите изображение для анализа.")

# Футер
st.markdown("<hr><p style='text-align:center;'>Made with ❤️ using Streamlit & TensorFlow</p>", unsafe_allow_html=True)
