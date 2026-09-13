FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .[dev]
ENTRYPOINT ["python", "-m", "ctrag.benchmarks"]
