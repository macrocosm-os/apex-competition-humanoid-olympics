# hello_world REFEREE image (the scorer sandbox: ground truth + scoring).
#
# VENDORED SDK pattern: build on FROM python:3.12-slim and COPY the vendored gym_v1/ in — we do
# NOT `FROM apex-referee-base` (not published to any registry; would fail in release CI). This
# image holds ALL ground truth (here: the expected sort order) and is isolated from the miner
# submission. Runs by the platform convention /app/referee.py.
#
# Build (context = THIS repo root):
#   docker build -f referee/Dockerfile -t hello-world-referee .
FROM python:3.12-slim

# Match how the platform runs sandboxes: non-root uid 1000, /app + /data owned by it. /data is
# where the referee writes result.json.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app /data && chown -R 1000:1000 /app /data
WORKDIR /app

# No pip install: the vendored gym_v1 is stdlib-only, and hello_world needs nothing else.

COPY referee/gym_v1/ /app/gym_v1/
COPY referee/referee.py /app/referee.py

USER 1000

# spec.referee runs /app/referee.py by convention.
CMD ["python", "/app/referee.py"]
