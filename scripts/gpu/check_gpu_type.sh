#!/bin/bash

OUTPUT_FILE="/vol/bitbucket/al1624/projects/bayesian-moe-router/scripts/gpu/gpu_info.txt"
> "$OUTPUT_FILE"  # Clear output file

HOST_PATTERNS=(
    "texel%02d.doc.ic.ac.uk:1:44"
    "oak%02d.doc.ic.ac.uk:1:38"
    "perry%02d.doc.ic.ac.uk:1:6"
    "cedar%02d.doc.ic.ac.uk:1:7"
    "gpu%02d.doc.ic.ac.uk:1:36"
    "ash%02d.doc.ic.ac.uk:1:41"
    "beech%02d.doc.ic.ac.uk:1:20"
    "willow%02d.doc.ic.ac.uk:1:20"
    "vertex%02d.doc.ic.ac.uk:1:22"
    "ray%02d.doc.ic.ac.uk:1:26"
    "maple%d.doc.ic.ac.uk:1:10"
    "curve%02d.doc.ic.ac.uk:1:12"
    "pixel%02d.doc.ic.ac.uk:1:40"
)

for pattern in "${HOST_PATTERNS[@]}"; do
    IFS=":" read -r fmt start end <<< "$pattern"
    for ((i=start; i<=end; i++)); do
        HOST=$(printf "$fmt" "$i")
        echo "Checking $HOST..."

        GPU_INFO=$(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes "$HOST" "nvidia-smi --query-gpu=memory.total --format=csv,noheader" 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$GPU_INFO" ]; then
            while read -r MEM; do
                echo "$HOST $MEM" >> "$OUTPUT_FILE"
            done <<< "$GPU_INFO"
        else
            echo "$HOST: SSH failed or nvidia-smi not available."
        fi
    done
done

# Get GPU info on the original machine
LOCAL_GPU_INFO=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null)
if [ $? -eq 0 ] && [ -n "$LOCAL_GPU_INFO" ]; then
    while read -r MEM; do
        echo "localhost $MEM" >> "$OUTPUT_FILE"
    done <<< "$LOCAL_GPU_INFO"
else
    echo "localhost: nvidia-smi not available."
fi