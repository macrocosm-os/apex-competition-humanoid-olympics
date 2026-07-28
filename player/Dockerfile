# hello_world PLAYER image (the miner submission server).
#
# VENDORED gym_v1 pattern: we build on FROM python:3.12-slim and COPY the vendored gym_v1/ into
# the image — we do NOT `FROM apex-player-base` (that base image is not published to any
# registry, so it would fail in release CI). Vendoring is what every shipped competition does.
#
# Build (context = THIS repo root):
#   docker build -f player/Dockerfile -t hello-world-player .
#
# The submission is NOT baked in: the platform (and `apex-dev`) writes the miner's
# submission.py to /app/submission.py at run time.
FROM python:3.12-slim

# Match how the platform runs sandboxes: non-root uid 1000, /app + /data owned by it.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app /data && chown -R 1000:1000 /app /data
WORKDIR /app

# No pip install: the vendored gym_v1 is stdlib-only, and hello_world needs nothing else.
# A real competition pins its extra deps here with exact versions.

COPY player/gym_v1/ /app/gym_v1/
COPY player/launch.py /app/launch.py

USER 1000

# spec.entrypoints.evaluate.command is ["python", "/app/launch.py", "--port", "8000"].
CMD ["python", "/app/launch.py", "--port", "8000"]
