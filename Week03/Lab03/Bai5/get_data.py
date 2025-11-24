import os, zipfile, requests, shutil

# 1. Tải file zip COCO128
print("Dang tai du lieu...")
r = requests.get('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip')
with open('coco128.zip', 'wb') as f: f.write(r.content)

# 2. Giải nén
print("Dang giai nen...")
with zipfile.ZipFile('coco128.zip', 'r') as zip_ref: zip_ref.extractall(".")

# 3. Sắp xếp ảnh vào thư mục 'images'
if not os.path.exists('images'): os.makedirs('images')

source = os.path.join('coco128', 'images', 'train2017')
for filename in os.listdir(source):
    shutil.move(os.path.join(source, filename), os.path.join('images', filename))

# 4. Dọn dẹp rác
shutil.rmtree('coco128')
if os.path.exists('coco128.zip'): os.remove('coco128.zip')

print("XONG! Da co 128 anh trong thu muc 'images'.")