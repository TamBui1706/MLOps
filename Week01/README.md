# AG News Classification API

## Mô tả
API phân loại tin tức vào 4 danh mục:
- World (Thế giới)
- Sports (Thể thao)
- Business (Kinh doanh)
- Sci/Tech (Khoa học/Công nghệ)

## Cài đặt

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NLTK data
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

## Chạy ứng dụng

### Cách 1: Chạy trực tiếp
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Cách 2: Chạy với Docker
```bash
docker build -t ag-news-api .
docker run -p 8000:8000 ag-news-api
```

## Sử dụng API

### 1. Truy cập Swagger UI
Mở trình duyệt: http://localhost:8000/docs

### 2. Test API với curl
```bash
curl -X POST "http://localhost:8000/predict"      -H "Content-Type: application/json"      -d '{"text": "Apple announces new iPhone with advanced AI features"}'
```

### 3. Test API với Python
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "text": "Apple announces new iPhone with advanced AI features"
}

response = requests.post(url, json=data)
print(response.json())
```

## API Endpoints

### GET /
Thông tin về API

### GET /health
Kiểm tra trạng thái API

### POST /predict
Dự đoán danh mục cho bài báo

**Request Body:**
```json
{
  "text": "Your news article text here"
}
```

**Response:**
```json
{
  "category": "Sci/Tech",
  "confidence": 0.95,
  "probabilities": {
    "World": 0.02,
    "Sports": 0.01,
    "Business": 0.02,
    "Sci/Tech": 0.95
  }
}
```

## Model Performance
- Model: Logistic Regression with TF-IDF
- Test Accuracy: ~92%
- Training Data: 120,000 samples
- Features: 10,000 TF-IDF features
