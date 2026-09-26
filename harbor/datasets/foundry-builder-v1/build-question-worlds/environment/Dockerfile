FROM python:3.12-slim

RUN useradd --create-home --shell /bin/bash agent \
    && mkdir -p /workspace /logs/artifacts \
    && chown -R agent:agent /workspace /logs

WORKDIR /workspace
COPY --chown=agent:agent starter/ /workspace/
RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt
