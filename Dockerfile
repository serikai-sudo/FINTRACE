# Base pinned by digest (multi-arch index of python:3.11-slim) for reproducible builds.
ARG PYTHON_BASE=python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

# ---------- stage 1: resolve and install pinned runtime deps into an isolated venv ----------
FROM ${PYTHON_BASE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
# Drop what the runtime never uses: installer tooling and bundled package test suites.
# .pyc files are kept on purpose: cold-start import time counts against the 60 s/episode budget.
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
 && /opt/venv/bin/pip uninstall --yes setuptools pip \
 && find /opt/venv -type d -name tests -prune -exec rm -rf {} +

# ---------- stage 2: minimal runtime, non-root, offline-ready ----------
FROM ${PYTHON_BASE} AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/src:/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin fintrace

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/ /app/src/
# the jury's connection protocol: `import judge_adapter` from /app
COPY judge_adapter.py /app/judge_adapter.py
# Precompile once at build time: PYTHONDONTWRITEBYTECODE + read-only /app would otherwise
# recompile the package on every container start.
RUN python -m compileall -q /app/src /app/judge_adapter.py

USER fintrace

ENTRYPOINT ["python", "-m", "fintrace"]
