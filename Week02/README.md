# Week 02 — Tổng Quan
Week02 chứa hai bài tập của Tuần 02: Adult Income Classification và CIFAR-10 (subset 5 lớp + Augmentation). .

---

## Nội dung chính
- Bài 1 — Adult Income Classification
  - Mục tiêu: Phân loại thu nhập (>50K vs <=50K). So sánh performance trước & sau tiền xử lý (Raw vs Full preprocessing).
  - Mô hình: LogisticRegression, RandomForest, XGBoost.
  - Step chính: xử lý missing (`?` → NaN), xử lý outlier (IQR), one-hot encoding, scale, train/val/test (stratify.
  - Đánh giá: Accuracy, Precision, Recall, F1, Confusion Matrix.
  - Lưu: sklearn pipeline (`joblib.dump`), bảng metrics CSV/JSON, notebook kết quả, plots.

- Bài 2 — CIFAR-10 (5 lớp)
  - Mục tiêu: So sánh huấn luyện với và không có Data Augmentation.
  - Dữ liệu: 5 lớp, mỗi lớp 1000 ảnh (subset).
  - Augmentation: Flip, Rotation, Translate, Zoom/Crop, Brightness/Contrast.
  - Mô hình: SimpleCNN.
  - Huấn luyện: ~50 epochs, batch size 64, Optimizer: Adam/SGD, chạy 3 seed.
  - Lưu: best model checkpoint (`best_cifar10_model.pth`).
 
  - ## Yêu cầu môi trường
Cài các packages chính:
```bash
pip install pandas numpy scikit-learn torch torchvision seaborn matplotlib joblib tqdm albumentations xgboost optuna
```
