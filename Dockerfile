FROM python:3.11-slim

WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -e .

CMD ["python", "scripts/run_offline.py", "--help"]
