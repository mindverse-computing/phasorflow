FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install Python dependencies first for better build-layer caching.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy repository as a package directory to keep imports stable.
COPY . /workspace/phasorflow

# Default command: lightweight smoke check.
CMD ["python", "-c", "import phasorflow as pf; print('PhasorFlow', pf.__version__)"]
