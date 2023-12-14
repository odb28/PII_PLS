#!/bin/bash

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J odb28

#! Which project should be charged:
#SBATCH -A CUNNIFFE-SL3-CPU
#SBATCH -p cclake

#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! SBATCH --qos=INTR

#! How many (MPI) tasks will there be in total? (<= nodes*56)
#! The Cascade Lake (cclake) nodes have 56 CPUs (cores) each and
#! 3420 MiB of memory per CPU.
#SBATCH --ntasks=1

#! How much wallclock time will be required?
#SBATCH --time=01:15:00

#! Submit a job array with index values between 0 and 31
#! NOTE: This must be a range, not a single number (i.e. specifying '32' here would only run one job, with index 32)
#SBATCH --array=18-197

#! What types of email messages do you wish to receive?
#SBATCH --mail-type=ALL

#! This to prevents the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
#SBATCH --no-requeue

#SBATCH -o ../slurm_logs/output.%j.out # STDOUT

#! sbatch directives end here

#! ############################################################
#! Module setup

#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                
module purge                              
module load rhel7/default-ccl            

#! Insert additional module load commands after this line if needed:
module load python/3.8

#! ############################################################
#! Python setup
source fitting_env/bin/activate

#! Work directory (i.e. where the job will run):
workdir="/home/odb28/rds/hpc-work/PII_PLS/Cluster/scripts"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! ############################################################
#! Command line construction
application="fit1000.py"
options=""

CMD="$application $options"

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > ../slurm_logs/machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat ../slurm_logs/machine.file.$JOBID | sed -e 's/\..*$//g'`
fi
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval python $CMD
