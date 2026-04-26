FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_md && \
    python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('averaged_perceptron_tagger_eng', quiet=True); nltk.download('wordnet', quiet=True)"

COPY configs/ configs/
COPY src/ src/
COPY main.py .

RUN mkdir -p data/processed outputs/figures reports Dataset

ENTRYPOINT ["python", "main.py"]
CMD ["--config", "configs/default.yaml"]
