# mkdir outputs    # comment this out if directory already made
dir="RUN/bc4/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
ram="16G"        # change these
time="0-2:00"    # accordingly
job_name="DEPTH"
CONFIG="configs/bc4.yml"
sbatch -t $time -J $job_name -o $dir$'depth.out' -e $dir'depth.err' --mem=$ram RUN/bc4/submit_job.sh sh RUN/all_depth.sh
