FROM python:3.11

RUN apt-get update && apt-get install -y \
    libmagic-dev ffmpeg libsm6 libxext6  tesseract-ocr poppler-utils
WORKDIR /app
COPY requirements.txt requirements.txt
COPY ModelFactories ModelFactories
COPY ragflow-0.0.1-py3-none-any.whl ragflow-0.0.1-py3-none-any.whl
COPY pipelines pipelines
COPY utils  utils
COPY main.py .
COPY tasks.py .
#RUN pip3 install -r requirements.txt
RUN pip3 install ragflow-0.0.1-py3-none-any.whl
RUN pip3 install unstructured[all-docs]==0.12.2
RUN pip3 install unstructured[local-inference]==0.12.2
RUN pip3 install pgvector==0.2.5

EXPOSE 5000