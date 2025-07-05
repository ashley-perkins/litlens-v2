FROM python:3.10-slim

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Set HF environment variables BEFORE code runs
ENV HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface \
    HF_DATASETS_CACHE=/tmp/huggingface \
    HF_METRICS_CACHE=/tmp/huggingface \
    HUGGINGFACE_HUB_CACHE=/tmp/huggingface \
    NLTK_DATA=/tmp/nltk_data

COPY . .

RUN python -m nltk.downloader punkt -d /tmp/nltk_data

EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]