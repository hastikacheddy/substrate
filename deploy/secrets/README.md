# Secrets for `docker-compose.yml`

The compose file reads six files from this directory and mounts each into the containers that need it under `/run/secrets/`. They are never environment
variables, so they do not appear in `docker inspect`, in a process listing, or in a crash dump of the environment. Everything in this directory except this
file is ignored by git and by the image build (`.gitignore`, `.dockerignore`).

| File | What it holds | Who reads it |
|------|---------------|--------------|
| `db_password` | the PostgreSQL password for the `substrate` user | `db` |
| `db_url` | `postgresql+psycopg2://substrate:<the same password>@db:5432/substrate` | `api`, `worker` |
| `grafana_admin_password` | Grafana's administrator password | `grafana` |
| `prometheus_token` | an API key of an `OPERATOR` principal | `prometheus` |
| `redis_conf` | one line, `requirepass <password>` | `redis` |
| `redis_url` | `redis://:<the same password>@redis:6379/0` | `api`, `worker` |

Create them with random values (nothing is printed):

```bash
cd deploy/secrets
python -c "import secrets; print(secrets.token_urlsafe(32))" > db_password
python -c "import secrets; print(secrets.token_urlsafe(24))" > grafana_admin_password
printf 'postgresql+psycopg2://substrate:%s@db:5432/substrate' "$(cat db_password)" > db_url
python -c "import secrets; print(secrets.token_urlsafe(24))" > redis_password
printf 'requirepass %s
' "$(cat redis_password)" > redis_conf
printf 'redis://:%s@redis:6379/0' "$(cat redis_password)" > redis_url
rm redis_password
chmod 444 db_password db_url grafana_admin_password redis_conf redis_url
chmod 700 .
```

Why 444 on the files and 700 on the directory: with compose (outside Swarm) a file secret is a bind mount that keeps the host file's owner and mode, and the
containers run as other users (10001, 70, 472, 65534), so a mode-600 file would be unreadable inside. The directory's mode is what keeps other users on the host
away from them; the Docker daemon, which does the mounting, is root.

The Prometheus token comes from the platform itself, after the first start (until it exists Prometheus reports the API as down):

```bash
docker compose -f deploy/docker-compose.yml run --rm api platform init --tenant lab        # prints the administrator's key once
read -rs SUBSTRATE_API_KEY && export SUBSTRATE_API_KEY                                     # paste that key; it is not echoed or kept in the shell history
docker compose -f deploy/docker-compose.yml run --rm -e SUBSTRATE_API_KEY api platform apikey create prometheus --role OPERATOR
# write the key it prints into deploy/secrets/prometheus_token (chmod 444), then:  docker compose -f deploy/docker-compose.yml restart prometheus
```

`db_password` and the Redis password must contain only URL-safe characters (the generator above does) because they are embedded in URLs.
Redis is only a wake-up channel for idle workers; if you do not want it, remove the `redis` service and the two `redis_*` secrets, and the workers poll the database instead.
Rotate any of these by replacing the file and recreating the container that reads it; revoke an API key with `platform apikey revoke <key id>`.
