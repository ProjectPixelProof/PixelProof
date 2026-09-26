FROM python:3.12-slim

RUN mkdir -p /tests /logs/verifier
COPY test.sh /tests/test.sh
COPY verify_auth_smoke.py /tests/verify_auth_smoke.py
RUN chmod +x /tests/test.sh

WORKDIR /workspace
