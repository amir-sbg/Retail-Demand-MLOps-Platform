FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-api.txt

COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "mlops_platform.api:app", "--host", "0.0.0.0", "--port", "8000"]
