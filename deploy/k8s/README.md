# Kubernetes

A kustomization for the API, two workers, a single PostgreSQL, default-deny network policies and the restricted Pod Security Standard.
`tests/test_deploy.py` checks the manifests' structure and the security properties below. They were also applied to a one-node minikube cluster (Kubernetes 1.30, Calico), where every pod ran under the
restricted Pod Security Standard, a run was submitted and reproduced, and the network policies were checked from inside the pods and from other namespaces; see [docs/platform.md](../../docs/platform.md) for that and for what was not run (a
managed cluster, more than one node, another network plugin, a real ReadWriteMany storage class, TLS). The Helm chart in `deploy/helm/substrate` renders the same workloads.

```bash
kubectl apply -k deploy/k8s --dry-run=client          # needs only kubectl, no cluster
kubectl create namespace substrate
# The database URL here names the Service "postgres" (docker-compose.yml's file names "db", which does not exist in a cluster). Make it from the password, in a file that is removed afterwards:
printf 'postgresql+psycopg2://substrate:%s@postgres:5432/substrate' "$(cat deploy/secrets/db_password)" > /tmp/k8s_db_url
kubectl -n substrate create secret generic substrate-secrets \
  --from-file=db_password=deploy/secrets/db_password --from-file=db_url=/tmp/k8s_db_url        # see deploy/secrets/README.md for how to make db_password
rm /tmp/k8s_db_url
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
