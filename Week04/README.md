# MLOps Lab 04: System Monitoring with Prometheus & Grafana

Bài tập Lab 04 môn MLOps: Xây dựng hệ thống giám sát tài nguyên máy tính (CPU, RAM, Disk, Network) sử dụng Docker, Prometheus, Node Exporter và Grafana.

## 📺 Demo Video

Video demo chi tiết quá trình setup dashboard và thực hiện các câu query:
https://youtu.be/R1BGewRQr90
**[>> NHẤN VÀO ĐÂY ĐỂ XEM VIDEO TRÊN YOUTUBE <<](https://youtu.be/R1BGewRQr90)**

---

## 🚀 Giới thiệu
Dự án này sử dụng **Docker Compose** để triển khai 3 dịch vụ chính:
1.  **Node Exporter:** Thu thập các chỉ số (metrics) từ phần cứng hệ thống (OS, CPU, RAM...).
2.  **Prometheus:** Time-series database dùng để lưu trữ dữ liệu thu thập được từ Node Exporter.
3.  **Grafana:** Giao diện trực quan hóa dữ liệu (Dashboard) kết nối với Prometheus.

## 📂 Cấu trúc dự án
```text
lab04_mlops/
├── docker-compose.yml    # File cấu hình Docker services
├── prometheus.yml        # File cấu hình nguồn dữ liệu cho Prometheus
└── README.md             # Tài liệu hướng dẫn
