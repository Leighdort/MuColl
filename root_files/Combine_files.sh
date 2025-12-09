#!/bin/bash
#This will be how I combine all the scripts

BASE="/users/rldohert/data/mucoll/rldohert"
cd "$BASE" || exit 1

# Loop over all pdg directories
module load root 
for DIR in pdg_*; do
    [ -d "$DIR" ] || continue

    echo "Processing $DIR ..."

    # Extract labels from folder name for output filenames
    # Examples: pdg_11_pt_100_theta_15-15
    PDG=$(echo "$DIR" | cut -d'_' -f2)
    PT=$(echo "$DIR" | cut -d'_' -f4)
    THETA=$(echo "$DIR" | cut -d'_' -f6)

    # Output names
    SIM_OUT="sim_pdg_${PDG}_pt_${PT}_theta_${THETA}.root"
    RECO_OUT="reco_pdg_${PDG}_pt_${PT}_theta_${THETA}.root"
    GEN_OUT="gen_pdg_${PDG}_pt_${PT}_theta_${THETA}.root"
    DIGI_OUT="digi_pdg_${PDG}_pt_${PT}_theta_${THETA}.root"

    # Collect input file lists
    SIM_LIST=$(ls "$DIR"/job_*/sim_output_*.root 2>/dev/null)
    RECO_LIST=$(ls "$DIR"/job_*/reco_output_*.root 2>/dev/null)
    GEN_LIST=$(ls "$DIR"/job_*/gen_output_*.root 2>/dev/null)
    DIGI_LIST=$(ls "$DIR"/job_*/digi_output_*.root 2>/dev/null)

    # Run hadd safely only if files exist
    if [ -n "$SIM_LIST" ]; then
        echo "  Combining SIM → $SIM_OUT"
        hadd -f "$DIR/$SIM_OUT" $SIM_LIST
    else
        echo "  No SIM files found in $DIR"
    fi

    if [ -n "$RECO_LIST" ]; then
        echo "  Combining RECO → $RECO_OUT"
        hadd -f "$DIR/$RECO_OUT" $RECO_LIST
    else
        echo "  No RECO files found in $DIR"
    fi

    if [ -n "$GEN_LIST" ]; then
        echo "  Combining GEN → $GEN_OUT"
        hadd -f "$DIR/$GEN_OUT" $GEN_LIST
    else
        echo "  No GEN files found in $DIR"
    fi

    if [ -n "$DIGI_LIST" ]; then
        echo "  Combining DIGI → $DIGI_OUT"
        hadd -f "$DIR/$DIGI_OUT" $DIGI_LIST
    else
        echo "  No DIGI files found in $DIR"
    fi

    echo "Finished $DIR"
    echo
done