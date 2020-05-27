# mkdir outputs    # comment this out if directory already made
dir="RUN/bc4/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
ram="128G"        # change these
time="0-1:00"    # accordingly
job_name="d-VID"
CONFIG="configs/bc4.yml"
# VID="depth-bear"
# sbatch -t $time -J $job_name$VID -o $dir$VID'.out' -e $dir$VID'.err' --mem=$ram RUN/bc4/submit_cpu.sh sh RUN/vid.sh $CONFIG $VID
VID="depth-child"
sbatch -t $time -J $job_name$VID -o $dir$VID'.out' -e $dir$VID'.err' --mem=$ram RUN/bc4/submit_cpu.sh sh RUN/vid.sh $CONFIG $VID
VID="depth-cup"
sbatch -t $time -J $job_name$VID -o $dir$VID'.out' -e $dir$VID'.err' --mem=$ram RUN/bc4/submit_cpu.sh sh RUN/vid.sh $CONFIG $VID
VID="depth-face"
sbatch -t $time -J $job_name$VID -o $dir$VID'.out' -e $dir$VID'.err' --mem=$ram RUN/bc4/submit_cpu.sh sh RUN/vid.sh $CONFIG $VID
VID="depth-walking"
sbatch -t $time -J $job_name$VID -o $dir$VID'.out' -e $dir$VID'.err' --mem=$ram RUN/bc4/submit_cpu.sh sh RUN/vid.sh $CONFIG $VID
