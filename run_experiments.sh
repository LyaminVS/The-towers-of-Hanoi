#!/bin/bash
# FILE: run.sh

echo "===================================================="
echo "   Tower of Hanoi RL: Running Full Experiment Set   "
echo "===================================================="

# Check if we are inside Docker or have Python
if ! command -v python3 &> /dev/null
then
    echo "Error: Python3 could not be found."
    exit
fi

# Run the master Python script
python3 run_experiments.py

echo "===================================================="
echo "   Experiments Completed Successfully!              "
echo "===================================================="