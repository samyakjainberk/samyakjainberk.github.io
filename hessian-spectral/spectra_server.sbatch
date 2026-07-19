#!/bin/bash
#SBATCH --job-name=spectra-server
#SBATCH --partition=main
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/nas/ucb/samsj/blog/captures/server_%j.log

# GPU server for the Training-time spectra card in the blogpost.
# After it starts, the log below prints the node and the exact ssh tunnel
# command; then open http://127.0.0.1:8974/hessian-spectral-estimation.html
echo "node: $(hostname)   job: $SLURM_JOB_ID   gpu: $CUDA_VISIBLE_DEVICES"
cd /nas/ucb/samsj/blog
exec python3 spectra_server.py --port 8974 --bind 0.0.0.0
