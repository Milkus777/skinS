import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

# Настройки страницы
st.set_page_config(page_title="Диагностика кожных заболеваний", page_icon="🩺", layout="centered")

# Словарь классов (из Colab-скрипта)
class_names_ru = [
    'Меланоцитарные невусы',
    'Меланома',
    'Доброкачественные кератозоподобные поражения',
    'Базальноклеточная карцинома',
    'Актинические кератозы',
    'Сосудистые поражения',
    'Дерматофиброма'
]

# Архитектура модели (копия из Colab)
def build_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(75, 100, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(150))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

# Загрузка модели с прогревом
@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights("model.keras")  # или full_model.h5

    # Прогреваем модель dummy-батчем
    dummy_input = np.zeros((1, 75, 100, 3))
    model.predict(dummy_input)

    return model

model = load_model()

# Интерфейс
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🩺 Диагностика кожных заболеваний</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Загрузите фото кожи, чтобы получить диагноз от нейросети.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((100, 75))  # Точный размер: ширина=100, высота=75
    img_array = np.array(image) / 255.0  # Нормализация [0, 1]
    img_array = (img_array - 0.5) * 2  # Приведение к диапазону [-1, 1], как в обучении
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
