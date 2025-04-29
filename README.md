# 🚗 Ứng dụng học sâu trong phát hiện hành vi của người điều khiển xe ô tô
Bài tập lớn: Trí tuệ nhân tạo (Học kì 2 - Năm 3 - Học viện Ngân hàng)

## Mục lục
* [Thông tin cơ bản](#thông-tin-cơ-bản)
* [Techstack](#techstack)
* [Thư viện sử dụng](#thư-viện-sử-dụng)
* [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)

## Thông tin cơ bản
Xây dựng ba mô hình học sâu theo cấu trúc mạng neuron (neural network) Convolutional Neural Network (hay gọi tắt là CNN) nhằm ứng dụng trong phát hiện hành vi của người điều khiển xe ô tô
- AlexNet
- VGGNet (cụ thể là VGG16)
- GoogLeNet (hay còn gọi là Inception V1)

**Bộ dữ liệu**
- [Driver Behavior Dataset (Kaggle)](https://www.kaggle.com/datasets/robinreni/revitsone-5class)

**Nhóm tác giả**
- [Nguyễn Hoàng Tâm](https://github.com/nghtamm)
- [Phạm Ngọc Nghiệp](https://github.com/xxelxt)
- [Nguyễn Huy Phước](https://github.com/DurkYerunz)
	
## Techstack
- Ngôn ngữ lập trình Python

## Thư viện sử dụng
```
pandas
numpy
matplotlib
seaborn
scikit-learn
opencv
pillow (PIL)
tensorflow
keras
streamlit
```
	
## Hướng dẫn sử dụng
- Chạy notebook ***driver-behavior.ipynb*** thông qua Jupyter Notebook/ JupyterLab hoặc Visual Studio Code (đã cài đặt extension Jupyter Notebook) để tiến hành huấn luyện mô hình (nếu muốn)
- Mở terminal trong thư mục gốc của project để sử dụng giao diện người dùng
```
streamlit run app.py
```
