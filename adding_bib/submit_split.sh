#!/bin/bash
#SBATCH -J mucoll_sim
#SBATCH -p batch
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --array=0-199
#SBATCH -o sim_%A_%a.out
#SBATCH -e sim_%A_%a.err

# ---- Job array ID ----
JOB_ID=$SLURM_ARRAY_TASK_ID

# ---- Simulation configuration ----
NEVENTS_PER_JOB=50
OUTPUT_BASE_DIR="/users/rldohert/data/mucoll/rldohert"
#OUTPUT_BASE_DIR="/oscar/data/mleblan6/mucoll/rldohert"
MUCOLL_BENCHMARKS_PATH="/users/rldohert/work/mucoll"
#MUCOLL_BENCHMARKS_PATH="/work"

# ---- Apptainer configuration ----
APPTAINER_IMAGE="docker://ghcr.io/muoncollidersoft/mucoll-sim-alma9:full_gaudi_test"
DATA_DIR_TO_BIND="/users/rldohert/data/mucoll/rldohert"
#DATA_DIR_TO_BIND="/oscar/data/mleblan6/mucoll/rldohert"
WORK_DIR="/users/rldohert/work/mucoll"
SCRIPT_PATH="/work/mucoll-slurm/split_generation.sh"

# ---- Run split_generation.sh inside the container ----
apptainer exec --cleanenv \
    --bind /users/rldohert/work/mucoll:/work \
    --bind /oscar/data/mleblan6/mucoll/rldohert:/oscar/data/mleblan6/mucoll/rldohert \
    ${APPTAINER_IMAGE} \
    bash ${SCRIPT_PATH} \
    ${JOB_ID} ${NEVENTS_PER_JOB} ${OUTPUT_BASE_DIR} ${MUCOLL_BENCHMARKS_PATH}
