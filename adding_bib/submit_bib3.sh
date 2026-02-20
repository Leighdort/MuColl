#!/bin/bash
#SBATCH -J mucoll_bib_reco
#SBATCH -p batch
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH --array=0-999
#SBATCH -o bib_reco_%A_%a.out
#SBATCH -e bib_reco_%A_%a.err

# ========= Array ID =========
CHUNK=$SLURM_ARRAY_TASK_ID

NEVENTS=10
OUTPUT_BASE_DIR="/users/rldohert/data/mucoll/rldohert"

# Container sees benchmarks at /work
APPTAINER_IMAGE=docker://ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:main
#APPTAINER_IMAGE="docker://ghcr.io/muoncollidersoft/mucoll-sim-alma9:full_gaudi_test"
#APPTAINER_IMAGE=docker://ghcr.io/muoncollidersoft/mucoll-sim-alma9:main
# Bind host paths → container paths
WORK_BIND="/users/rldohert/work/mucoll"
#DATA_BIND="/users/rldohert/data/mucoll/rldohert:/users/rldohert/data/mucoll/rldohert"
DATA_BIND="/oscar/data/mleblan6/mucoll/rldohert"
MUCOLL_BENCHMARKS_PATH="/users/rldohert/work/mucoll/mucoll-benchmarks"
#BIB_BIND="/users/rldohert/data/usmcc/BIB/Simulation:/BIB"
BIB_BIND="/oscar/data/mleblan6/usmcc/BIB/Simulation"
# Worker script path inside cntainer
SCRIPT_PATH="/users/rldohert/work/mucoll/mucoll-slurm/run_bib.sh"

apptainer exec --cleanenv \
    --bind ${WORK_BIND} \
    --bind ${DATA_BIND} \
    --bind ${BIB_BIND} \
    ${APPTAINER_IMAGE} \
    bash ${SCRIPT_PATH} \
        ${CHUNK} \
        ${NEVENTS} \
        ${OUTPUT_BASE_DIR} \
        ${MUCOLL_BENCHMARKS_PATH}
