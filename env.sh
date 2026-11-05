# Run "source env.sh" from the repository root in each terminal.
export PYTHONPATH="$PWD:$PWD/partA:$PWD/partB:$PWD/partC:$PWD/partD${PYTHONPATH:+:$PYTHONPATH}"
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db"
export ETA_MODEL="data/eta_v1.joblib"

# Keep numerical libraries from creating excessive CPU threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -o pipefail
