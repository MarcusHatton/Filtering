#!/bin/bash
####################################
# JOB INFO: serial with array
####################################

#SBATCH --output=slurm_outputs/slurm-%A.out        
#SBATCH --nodes=1                
#SBATCH --ntasks=1              
##SBATCH --array=1-5
#SBATCH --time=01:00:00         


#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=fail
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=mjh1n20@soton.ac.uk


####################################
# LOADING THE CONDA ENVIRONMENT
####################################

module load conda/py3-latest
# I get warnings to use 'conda deactivate' instead but that doesn't work
source deactivate 
conda activate first_try

####################################
# RETRIEVING THE PARAMETERS FOR THE RUNS
####################################

## ET=$(awk -v var=${SLURM_ARRAY_TASK_ID} '$1==var {print $2}' config.txt)
# response=$(conda list | grep h5py) 
# echo "Response to conda list | grep h5py ${response}"
# echo "End time of task with ID ${SLURM_ARRAY_TASK_ID} is ${ET}."


####################################
# LAUNCHING THE JOBS
#####################################

python3 -u main_file.py config.txt
# python -u RootVsMin.py
#python3 -u pickling_micro.py ${ET}
# python3 -u visualizing_micro.py ${ET}
# python3 -u pickling_meso_obs.py ${ET} 
# python3 -u visualizing_obs.py ${ET}
# python3 -u multiple_filtering_meso.py ${ET}
# python3 -u visualizing_meso.py ${ET}
# python3 -u visualizing_meso_manyFW.py ${ET}
