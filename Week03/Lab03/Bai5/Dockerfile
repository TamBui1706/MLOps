# Sử dụng Pytorch làm nền (bản nhẹ, chạy CPU cũng được)
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Cài đặt thư viện hệ thống (để xử lý ảnh OpenCV)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 git wget

WORKDIR /app

# Tải mã nguồn YOLOv7
RUN git clone https://github.com/WongKinYiu/yolov7.git

# Cài đặt thư viện Python cần thiết
WORKDIR /app/yolov7
RUN pip install -r requirements.txt

# Tải file trọng số (trí tuệ của model)
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

# Thiết lập lệnh chạy mặc định
ENTRYPOINT ["python", "detect.py"]