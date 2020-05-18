#!/bin/sh
dir="RUN/bc4/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
ram="32G"        # change these
time="0-5:00"    # accordingly
job_name="SGL"$1
CONFIG="configs/bc4.yml"
sbatch -t $time -J $job_name -o $dir$job_name'.out' -e $dir$job_name'.err' --mem=$ram RUN/bc4/submit_job.sh sh RUN/single.sh $1
