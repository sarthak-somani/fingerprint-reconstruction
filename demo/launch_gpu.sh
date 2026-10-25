#!/bin/bash
# Serve the demo from a GPU node for up to 4 h and print the ssh tunnel to use. From the repo root:
#     mkdir -p logs && sbatch demo/launch_gpu.sh                      (default partition below; or -p <partition>)
# Then: tail -f logs/demo-<jobid>.log, wait for the local URL line, and run the printed ssh command on your laptop.
# Set PY to your Python (default: python on PATH) and LOGIN to the cluster's login host.
#SBATCH -J fpr-demo
#SBATCH -p interactive
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -o logs/demo-%j.log

PORT=${PORT:-7860}
LOGIN=${LOGIN:-prajna.iitb.ac.in}
echo "node $(hostname), port $PORT. On your laptop run:"
echo "    ssh -N -L 8888:$(hostname):$PORT $USER@$LOGIN"
echo "then open http://127.0.0.1:8888 (8888 avoids the ports Windows reserves) when this log prints the local URL (start-up takes 3-4 min)."
exec ${PY:-python} -u demo/gradio_app.py --host 0.0.0.0 --port "$PORT"
