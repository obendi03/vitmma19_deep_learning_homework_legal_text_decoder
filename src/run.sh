#!/bin/bash
#!/bin/bash
set -e

# Create logs directory if missing
mkdir -p logs

# Timestamped logfile
LOGFILE="logs/run.log"

echo "Logging to: $LOGFILE"

{
    echo "=== Starting pipeline at $(date) ==="

    echo "Running script: 00_download_data.py"
    python3 00_download_data.py

    echo "Running script: 01_data_preprocessing.py"
    python3 01_data_preprocessing.py

    echo "Running script: 02_train.py"
    python3 02_train.py

#    echo "Running script: 02_train_ray_tune.py"
#    python3 02_train_ray_tune.py


    echo "Running script: 03_1_evaluation_test_set.py"
    python3 03_1_evaluation_test_set.py

    echo "Running script: 03_2_evaluation_consensus.py"
    python3 03_2_evaluation_consensus.py

    echo "Running script: 04_inference.py"
    python3 04_inference.py

    echo "=== Finished at $(date) ==="
} 2>&1 | tee "$LOGFILE"
