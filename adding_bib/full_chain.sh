#!/bin/bash
set -e
# Arguments
CHUNK=$1
NEVENTS=$2
OUTPUT_DIR=$3
MUCOLL_BENCHMARKS_PATH=$4
PDG=$5
PT=$6
THETA_MIN=$7
#Need to set the variables up here too 
SECONDS=0
echo "Starting chunk $CHUNK with $NEVENTS events"
INPUT_BASE=/users/rldohert/data/mucoll/rldohert/pdg_211_pt_10_theta_15-15_bib

LOGDIR=${OUTPUT_DIR}/logs2
mkdir -p ${LOGDIR}

LOGFILE=${LOGDIR}/chunk_${CHUNK}.log
ERRFILE=${LOGDIR}/chunk_${CHUNK}.err

# Redirect stdout and stderr
exec > >(tee -a ${LOGFILE}) 2> >(tee -a ${ERRFILE} >&2)

source /users/rldohert/work/mucoll/setup.sh
cd $MUCOLL_BENCHMARKS_PATH/k4MuCPlayground
source setup_digireco.sh .. MAIA_v0

#I got all fo the things running
#Create Temporary working directory
WORKDIR=/tmp/mucoll_chunk_${CHUNK}_$$

mkdir -p $WORKDIR
cd $WORKDIR
echo "Working in $WORKDIR"

cp -r $MUCOLL_BENCHMARKS_PATH/reconstruction/PandoraSettings/ ./

# -- Generation
echo "Running Generation..."
python  $MUCOLL_BENCHMARKS_PATH/generation/pgun/pgun_edm4hep.py \
    -p 1 -e $NEVENTS --pdg $PDG --pt $PT --theta $THETA_MIN -- gen_output.edm4hep.root

# -- Simulation
echo "Running Simulation..."
ddsim --steeringFile $MUCOLL_BENCHMARKS_PATH/simulation/steer_baseline.py \
    --numberOfEvents $NEVENTS \
    --inputFiles gen_output.edm4hep.root \
    --outputFile sim_output.edm4hep.root
echo "---------------------------"
echo "Simulation Finished: ${SECONDS} seconds"
echo "---------------------------"
# -- Digitization with Bib
echo "Running Digitization..."
k4run $MUCOLL_BENCHMARKS_PATH/digitization/digi_steer.py -n $NEVENTS \
    --IOSvc.Input sim_output.edm4hep.root \
    --IOSvc.Output digi_output.edm4hep.root

# -- Reconstruction
echo "Running Reconstruction..."
k4run $MUCOLL_BENCHMARKS_PATH/reconstruction/reco_steer.py -n $NEVENTS \
    --IOSvc.Input digi_output.edm4hep.root \
    --IOSvc.Output reco_output.edm4hep.root

#FINAL_OUT_DIR=$OUTPUT_DIR/nobib_${CHUNK}
# -- Move Outputs
#FINAL_OUT_DIR=$OUTPUT_DIR/nobib_${CHUNK}
FINAL_OUT_DIR=$OUTPUT_DIR/pdg_211_pt_10_theta_15-15_bib
#/users/rldohert/data/mucoll/rldohert/pdg_211_pt_2_theta_15-15_bib
mkdir -p $FINAL_OUT_DIR
echo "Moving files to $FINAL_OUT_DIR"

# -- Rename Files
mv gen_output.edm4hep.root $FINAL_OUT_DIR/gen_output_p10_211_nobib${CHUNK}.edm4hep.root
mv sim_output.edm4hep.root $FINAL_OUT_DIR/sim_output_p10_211_nobib${CHUNK}.edm4hep.root
mv digi_output.edm4hep.root $FINAL_OUT_DIR/digi_output_p10_211_nobib${CHUNK}.edm4hep.root
mv reco_output.edm4hep.root $FINAL_OUT_DIR/reco_output_p10_211_nobib${CHUNK}.edm4hep.root

# Cleanup
cd ..
rm -rf $WORKDIR
echo "Job $CHUNK finished successfully"
echo "Everything finished, total runtime: ${SECONDS} seconds"
