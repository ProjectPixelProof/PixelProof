FROM python:3.12-slim

RUN python -m pip install --no-cache-dir numpy==2.5.1 pillow==12.3.0 pytest==9.1.1

WORKDIR /tests
COPY . /tests/
RUN chmod +x /tests/test.sh
