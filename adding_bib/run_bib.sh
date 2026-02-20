#!/bin/bash
SECONDS=0
set -e
# Arguments
CHUNK=$1
NEVENTS=$2
OUTPUT_DIR=$3
MUCOLL_BENCHMARKS_PATH=$4
echo "Starting chunk $CHUNK with $NEVENTS events"

#Source the main enviornment setup
#INPUT_BASE=/users/rldohert/data/mucoll/rldohert/pdg_211_pt_5_theta_15-15_bib
#INPUT_BASE=/users/rldohert/data/mucoll/rldohert/new_sim_pt_50_0
INPUT_BASE=/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_15-15_bib2/job_${CHUNK}
INPUT_FILE=${INPUT_BASE}/sim_output_p50_211_nobib${CHUNK}.edm4hep.root

#INPUT_FILE=${INPUT_BASE}/sim_output_chunk_${CHUNK}.edm4hep.root
#I am changing the path to see
#INPUT_BASE=/oscar/data/mleblan6/mucoll/rldohert/pdg_211_pt_5_theta_15-15_bib
#INPUT_FILE=${INPUT_BASE}/sim_output_chunk_${CHUNK}.edm4hep.root

echo "I got to here"
source /users/rldohert/work/mucoll/setup.sh
#I am also changing source
#source /work/setup.sh

# Setup for k4MuCPlayground
cd $MUCOLL_BENCHMARKS_PATH/k4MuCPlayground
# We need to source setup_digireco.sh. It expects to be in k4MuCPlayground or given a path.
# We are in k4MuCPlayground, so passing ".." works as it expects the root of benchmarks.
source setup_digireco.sh .. MAIA_v0
echo "I got to here"

# Create a temporary working directory
WORKDIR=/tmp/mucoll_chunk_${CHUNK}_$$

mkdir -p $WORKDIR
cd $WORKDIR
echo "Working in $WORKDIR"
# Copy PandoraSettings needed for reconstruction

cp -r $MUCOLL_BENCHMARKS_PATH/reconstruction/PandoraSettings/ ./
# -- We need to rename the file here though so its not problemtic 
# We also need to point to where the data is from

cp "$INPUT_FILE" sim_output.edm4hep.root


# -- 3. Digitization with BIB-- 

MUPLUS=/oscar/data/mleblan6/usmcc/BIB/Simulation/MUPLUS
MUMINUS=/oscar/data/mleblan6/usmcc/BIB/Simulation/MUMINUS

k4run $MUCOLL_BENCHMARKS_PATH/digitization/digi_steer.py -n $NEVENTS \
    --RandSeed ${CHUNK} \
    --doOverlayFull \
    --OverlayFullNumberBackground 17 \
    --OverlayFullPathToMuPlus ${MUPLUS} \
    --OverlayFullPathToMuMinus ${MUMINUS} \
    --IOSvc.Input sim_output.edm4hep.root \
    --IOSvc.Output digi_output.edm4hep.root

# -- 4. Reconsutrction with BIB --
#cp -r ../reconstruction/PandoraSettings/ ./
k4run $MUCOLL_BENCHMARKS_PATH/reconstruction/reco_steer.py -n $NEVENTS \
    --IOSvc.Input digi_output.edm4hep.root \
    --IOSvc.Output reco_output.edm4hep.root

#Final output
#undo this
#FINAL_OUT_DIR=$OUTPUT_DIR/chunk_${CHUNK}
#undo this 
#mkdir -p $FINAL_OUT_DIR

echo Moving Files to $INPUT_BASE
mv digi_output.edm4hep.root $INPUT_BASE/digi_output_p50_211_bib${CHUNK}.edm4hep.root
mv reco_output.edm4hep.root $INPUT_BASE/reco_output_p50_211_bib${CHUNK}.edm4hep.root
#Rename files

echo "Chunk $CHUNK finished successfully"

#Cleanup
cd /
rm -rf $WORKDIR
echo "Total runtime: ${SECONDS} seconds"
