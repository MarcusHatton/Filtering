#!/bin/bash
####################################

#SBATCH --output=outputs/slurm-%A.out        
#SBATCH --nodes=1                
#SBATCH --ntasks=1              
#SBATCH --time=01:00:00         

####################################
# LOADING THE CONDA ENVIRONMENT
####################################

module load conda/py3-latest
# I get warnings to use 'conda deactivate' instead but that doesn't work
source deactivate 
conda activate first_try

####################################
# LAUNCHING THE JOBS
#####################################

python3 main_file.py config.txt
