#!/usr/bin/env bash
# Create an isolated Python environment with PySCF inside WSL, for substrate's quantum-chemistry engine.
#
# PySCF has no Windows build, so on Windows substrate runs it through WSL. This script needs no sudo and installs
# nothing system-wide: everything lives in ~/.substrate-qc. Remove it with:  rm -rf ~/.substrate-qc
#
# Run it from Windows with:   wsl bash /mnt/c/<path to this project>/scripts/setup_qc_env.sh
set -euo pipefail
ENV="${SUBSTRATE_QC_ENV:-$HOME/.substrate-qc}"

if [ ! -x "$ENV/bin/python" ]; then
    # Debian/Ubuntu ship `venv` without `ensurepip`, so create the environment without pip and bootstrap pip into it.
    python3 -m venv --without-pip "$ENV"
fi
if ! "$ENV/bin/python" -m pip --version >/dev/null 2>&1; then
    curl -sS --max-time 120 https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$ENV/bin/python" /tmp/get-pip.py --quiet
    rm -f /tmp/get-pip.py
fi
"$ENV/bin/python" -m pip install --quiet --upgrade pip
"$ENV/bin/python" -m pip install --quiet pyscf numpy

"$ENV/bin/python" - <<'EOF'
import numpy, pyscf
print(f"ok: numpy {numpy.__version__}, pyscf {pyscf.__version__}")
EOF
echo "environment: $ENV ($(du -sh "$ENV" | cut -f1))"
