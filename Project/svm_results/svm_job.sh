#!/bin/bash  
#$ -q gpu
#$ -N TestSVM
#$ -m beas

module load anaconda/3.5-2.4.0


python /data/users/anuragm/svm.py

