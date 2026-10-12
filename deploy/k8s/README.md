# Kubernetes

A kustomization for the API, two workers, a single PostgreSQL, default-deny network policies and the restricted Pod Security Standard.
Not tested against a cluster: `tests/test_deploy.py` checks the manifests' structure and the security properties below, nothing has been applied to a real cluster.

```bash
kubectl apply -k deploy/k8s --dry-run=client          # needs only kubectl, no cluster
kubectl create namespace substrate
kubectl -n substrate create secret generic substrate-secrets \
  --from-file=db_password=deploy/secrets/db_password --from-file=db_url=deploy/secrets/db_url     # see deploy/secrets/README.md for how to make them
kubectl apply -k deploy/k8s
kubectl -n substrate exec -it deploy/substrate-api -- python -m substrate platform init --tenant lab      # prints the first administrator's key once
```

(The namespace is created by the kustomization too; creating it first only lets the Secret be placed before the pods start.)

What you get: pods that run as uid 10001 with a read-only root filesystem, no capabilities, no privilege escalation, a `RuntimeDefault` seccomp profile and no service-account
token; a network that is closed unless a policy opens it (see `networkpolicy.yaml` for the exact openings); resource requests and limits; readiness and liveness probes on the
API; a PodDisruptionBudget for it; secrets mounted as files, not environment variables.

What you must still provide: an image (`deploy/Dockerfile`; set it in `kustomization.yaml`), a ReadWriteMany storage class for the artifacts (or an S3 store), an ingress with
TLS in front of the API (the API speaks plain HTTP), a real PostgreSQL if the data matters (delete `postgres.yaml`), and the namespaces named in `api-ingress`.
`optional/servicemonitor.yaml` scrapes the API's metrics if you run the Prometheus Operator.
