"""The control plane: experiment registry, run orchestration, artifact store, access control, reproducibility and the HTTP API.

Everything here is optional: the scientific core (engines, translators, pipeline) imports nothing from this package. Install the extras with
`pip install -e ".[platform]"`.
"""
