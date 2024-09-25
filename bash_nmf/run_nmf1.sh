#!/bin/bash
#SBATCH --partition=compute
#SBATCH --output=/dssg/home/ai2010815068/yan/gproject/scripts/logs/job.%j.out

singularity exec --bind /dssg/home/ai2010815068/yan/gproject:/data  /dssg/home/ai2010815068/docker_images/mcc\
/bin/bash /usr/local/nmf/run_nmf.sh \
/usr/local/R2023b /data/adhd/stability_splits /data
