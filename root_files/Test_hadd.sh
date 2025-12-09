#!/bin/bash
# DRY RUN — does not run hadd

BASE="/users/rldohert/data/mucoll/rldohert"
cd "$BASE" || exit 1

for DIR in pdg_*; do
    [ -d "$DIR" ] || continue

    echo "Processing $DIR ..."

    PDG=$(echo "$DIR" | cut -d'_' -f2)
    PT=$(echo "$DIR" | cut -d'_' -f4)
    THETA=$(echo "$DIR" | cut -d'_' -f6)

    SIM_OUT="sim_pdg_${PDG}_pt_${PT}_theta_${THETA}.root"
    RECO_OUT="reco_pdg_${PDG}_pt_${PT}_theta_${THETA}.root"
    GEN_OUT="gen_pdg_${PDG}_pt_${PT}_theta_${THETA}.root"
    DIGI_OUT="digi_pdg_${PDG}_pt_${PT}_theta_${THETA}.root"

    SIM_LIST=$(ls "$DIR"/job_*/sim_output_*.root 2>/dev/null)
    RECO_LIST=$(ls "$DIR"/job_*/reco_output_*.root 2>/dev/null)
    GEN_LIST=$(ls "$DIR"/job_*/gen_output_*.root 2>/dev/null)
    DIGI_LIST=$(ls "$DIR"/job_*/digi_output_*.root 2>/dev/null)

    if [ -n "$SIM_LIST" ]; then
        echo "  Would combine SIM → $DIR/$SIM_OUT"
        echo "    hadd -f $DIR/$SIM_OUT $SIM_LIST"
    else
        echo "  No SIM files found in $DIR"
    fi

    if [ -n "$RECO_LIST" ]; then
        echo "  Would combine RECO → $DIR/$RECO_OUT"
        echo "    hadd -f $DIR/$RECO_OUT $RECO_LIST"
    else
        echo "  No RECO files found in $DIR"
    fi

    if [ -n "$GEN_LIST" ]; then
        echo "  Would combine GEN → $DIR/$GEN_OUT"
        echo "    hadd -f $DIR/$GEN_OUT $GEN_LIST"
    else
        echo "  No GEN files found in $DIR"
    fi

    if [ -n "$DIGI_LIST" ]; then
        echo "  Would combine DIGI → $DIR/$DIGI_OUT"
        echo "    hadd -f $DIR/$DIGI_OUT $DIGIN_LIST"
    else
        echo "  No DIGI files found in $DIR"
    fi

    echo "Finished (dry run) $DIR"
    echo
done
