# ML service, registry bootstrap, retrain job, responder. One image, four commands.
# Build from the repository root, tag = commit (deploy/up.py does this):
#   minikube image build -t mlops-labs-ml:$(git rev-parse --short HEAD) \
#     -f deploy/docker/Dockerfile.ml .
FROM python:3.12-slim

ARG GIT_COMMIT=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    GIT_COMMIT=${GIT_COMMIT}
LABEL org.opencontainers.image.revision=${GIT_COMMIT}

COPY deploy/requirements/ml.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
COPY src/ /app/src/
COPY deploy/seed/ /app/deploy/seed/

EXPOSE 8000
CMD ["uvicorn", "mlops_labs.serve.ml_service:app", "--host", "0.0.0.0", "--port", "8000"]
