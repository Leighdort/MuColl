#!/bin/bash
set -e

JOB_ID=$1

# ---- Configuration ----
NEVENTS=10

GEN_DIR="/users/rldohert/data/mucoll/rldohert/pdg_211_pt_5_theta_15-15"
#GEN_DIR="/oscar/data/mleblan6/mucoll/rldohert/pdg_211_pt_5_theta_15-15"
GEN_FILE="${GEN_DIR}/gen_pdg_211_pt_5_theta_15-15.root"

#SIM_OUT_DIR="/oscar/data/mleblan6/mucoll/rldohert/pdg_211_pt_5_theta_15-15_bib"
SIM_OUT_DIR="/users/rldohert/data/mucoll/rldohert/pdg_211_pt_5_theta_15-15_bib"
MUCOLL_BENCHMARKS_PATH="/users/rldohert/work/mucoll/mucoll-benchmarks"
#MUCOLL_BENCHMARKS_PATH="/work"

# Make if it doens't already exist
mkdir -p "$SIM_OUT_DIR"

START_EVENT=$(( JOB_ID * NEVENTS ))

echo "JOB_ID       = $JOB_ID"
echo "START_EVENT = $START_EVENT"
echo "NEVENTS     = $NEVENTS"
echo "GEN_FILE    = $GEN_FILE"
echo "OUT_DIR     = $SIM_OUT_DIR"

# ---- Environment ----
#source /users/rldohert/work/mucoll/setup.sh
source /work/setup.sh
cd $MUCOLL_BENCHMARKS_PATH/k4MuCPlayground
source setup_digireco.sh .. MAIA_v0

# ---- Work directory ----
WORKDIR=/tmp/mucoll-job_${JOB_ID}_${RANDOM}
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "Working in $WORKDIR"

cp -r $MUCOLL_BENCHMARKS_PATH/reconstruction/PandoraSettings/ ./

# ---- Simulation chunk ----
ddsim \
  --steeringFile "$MUCOLL_BENCHMARKS_PATH/simulation/steer_baseline.py" \
  --inputFiles "$GEN_FILE" \
  --skipNEvents "$START_EVENT" \
  --numberOfEvents "$NEVENTS" \
  --outputFile "${SIM_OUT_DIR}/sim_output_chunk_${JOB_ID}.edm4hep.root"

echo "Simulation chunk ${JOB_ID} completed"
