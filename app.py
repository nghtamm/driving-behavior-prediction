import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import tempfile
import os
import pandas as pd

alex_model = load_model('model/alex_model.h5')
vgg16_model = load_model('model/vgg16_model.h5')
inception_model = load_model('model/inception_model.h5')

class_names = ['Other', 'Safe', 'Talking', 'Texting', 'Turning']

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")  # Chuyển đổi ảnh sang RGB
    img = image.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Thêm chiều batch
    img = img / 255.0  # Chuẩn hóa ảnh
    return img

def predict(model, image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class, predictions

st.title("Driver Behavior Detection")
st.write("##### Tải ảnh hoặc video lên để phân tích.")

uploaded_file = st.file_uploader("Chọn ảnh hoặc video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh đã tải lên', use_column_width=True)

        # Model selection
        model_choice = st.selectbox("Lựa chọn mô hình: ", ["AlexNet", "VGG16", "GoogLeNet (Inception V1)"])

        if model_choice == "AlexNet":
            selected_model = alex_model
        elif model_choice == "VGG16":
            selected_model = vgg16_model
        elif model_choice == "GoogLeNet (Inception V1)":
            selected_model = inception_model

        if st.button("Dự đoán"):
            predicted_class, predictions = predict(selected_model, image)
            st.write(f"#### Nhãn dự đoán: {predicted_class}")
            st.write("Xác suất dự đoán:")
            df = pd.DataFrame(predictions, columns=class_names)
            st.write(df)

    elif file_extension == "mp4":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        st.video(tfile.name)

        model_choice = st.selectbox("Lựa chọn mô hình: ", ["AlexNet", "VGG16", "GoogLeNet (Inception V1)"])

        if model_choice == "AlexNet":
            selected_model = alex_model
        elif model_choice == "VGG16":
            selected_model = vgg16_model
        elif model_choice == "GoogLeNet (Inception V1)":
            selected_model = inception_model

        if st.button("Dự đoán"):
            video = cv2.VideoCapture(tfile.name)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            predictions = []
            frames = []

            with st.spinner('Đang phân tích video...'):
                while video.isOpened():
                    success, frame = video.read()
                    if not success:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    predicted_class, prediction = predict(selected_model, image)
                    predictions.append(prediction[0])
                    frames.append(frame)

            video.release()
            os.remove(tfile.name)

            st.session_state.predictions = predictions
            st.session_state.frames = frames
            st.session_state.total_frames = total_frames

        if 'predictions' in st.session_state:
            st.write("Kết quả dự đoán các khung hình trong video:")

            frame_idx = st.slider("Chọn khung hình", 0, st.session_state.total_frames - 1, 0)
            st.image(st.session_state.frames[frame_idx], caption=f'Khung hình {frame_idx + 1}', use_column_width=True)

            st.write(f"#### Nhãn dự đoán: {class_names[np.argmax(st.session_state.predictions[frame_idx])]}")
            st.write("Xác suất dự đoán:")
            df = pd.DataFrame([st.session_state.predictions[frame_idx]], columns=class_names)
            st.write(df)
