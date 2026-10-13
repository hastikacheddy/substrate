# syntax=docker/dockerfile:1
# The control plane: the API and the workers are the same image with different commands.
# Pin the base image by digest in a release build (Dependabot's docker ecosystem keeps the digest current once it is there):
#   FROM python:3.13-slim-bookworm@sha256:<digest>
FROM python:3.13-slim-bookworm AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md requirements-lock.txt requirements-platform-lock.txt ./
COPY src ./src
# Every wheel is built here, with the versions pinned in the lock files used as constraints, so the final image installs from this directory only.
RUN pip wheel --wheel-dir /wheels -c requirements-platform-lock.txt ".[platform,postgres,otlp,redis,s3]"

FROM python:3.13-slim-bookworm
LABEL org.opencontainers.image.title="substrate control plane" \
      org.opencontainers.image.source="https://github.com/hastikacheddy/substrate" \
      org.opencontainers.image.licenses="NOASSERTION"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SUBSTRATE_WORKER_ISOLATION=process
# SUBSTRATE_IMAGE_DIGEST is set by the deployment (an image cannot know its own digest): it is recorded with every run and part of the environment hash.
# No shell, no login, no home: a fixed unprivileged user that owns only its data directory.
RUN useradd --system --uid 10001 --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin substrate \
 && mkdir -p /data/artifacts && chown -R 10001:10001 /data
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels "substrate[platform,postgres,otlp,redis,s3]" && rm -rf /wheels
USER 10001:10001
WORKDIR /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"]
ENTRYPOINT ["python", "-m", "substrate"]
CMD ["platform", "serve", "--host", "0.0.0.0", "--allow-external"]
