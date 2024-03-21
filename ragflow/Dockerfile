FROM python:3.10

RUN apt-get update && apt-get install -y \
    libmagic-dev ffmpeg libsm6 libxext6  tesseract-ocr poppler-utils
WORKDIR /app
COPY requirements.txt.old requirements.txt
COPY ModelFactories ModelFactories
COPY pipelines pipelines
COPY utils  utils
COPY main.py .
COPY tasks.py .
RUN pip3 install -r requirements.txt.old

EXPOSE 5000