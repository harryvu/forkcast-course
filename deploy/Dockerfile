# Forkcast ML service. Build: docker build -t forkcast-ml -f deploy/Dockerfile .   Run: docker run -p 8000:8000 forkcast-ml
FROM python:3.11-slim
WORKDIR /srv
COPY pyproject.toml README.md ./
COPY forkcast ./forkcast
COPY deploy ./deploy
COPY artifacts ./artifacts
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir . fastapi uvicorn
EXPOSE 8000
CMD ["uvicorn", "deploy.app:app", "--host", "0.0.0.0", "--port", "8000"]
