FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Optional extras, e.g. EXTRAS="[knowledge,embeddings]" for the ingestion
# worker. Empty by default so the CLI image stays lean.
ARG EXTRAS=""

WORKDIR /build
COPY . .
RUN pip install --no-cache-dir ".${EXTRAS}"

FROM python:3.13-slim

# TRADINGAGENTS_IN_DOCKER lets an update notice name the rebuild command.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRADINGAGENTS_IN_DOCKER=1

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home appuser \
 && install -d -m 0755 -o appuser -g appuser /home/appuser/.tradingagents \
 # Created here, not left to the volume mount: Docker creates a missing mount
 # point as root, which would make the embedding-model cache unwritable.
 && install -d -m 0755 -o appuser -g appuser /home/appuser/.cache/huggingface
USER appuser
WORKDIR /home/appuser/app

COPY --from=builder --chown=appuser:appuser /build .

ENTRYPOINT ["tradingagents"]
