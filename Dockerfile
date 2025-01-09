# Base image
FROM python:3.9-slim

# ตั้งค่าโฟลเดอร์ทำงานใน container
WORKDIR /app

# คัดลอกไฟล์ทั้งหมด
COPY . /app

# ติดตั้ง dependencies
RUN apt-get update && apt-get install -y libprotobuf-dev protobuf-compiler
RUN pip install --no-cache-dir transformers datasets sentencepiece
RUN pip install torch torchvision


# รันไฟล์ model.py เพื่อฝึกโมเดล
CMD ["python", "model.py"]
