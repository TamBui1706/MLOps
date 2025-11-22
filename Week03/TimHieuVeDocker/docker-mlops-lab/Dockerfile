# Sử dụng Python base image
FROM python:3.9-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Copy file main.py từ máy tính vào container
COPY main.py .

# Lệnh chạy file python khi container khởi động
CMD ["python", "main.py"]