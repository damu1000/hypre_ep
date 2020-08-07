#!/bin/bash

export CUDA_AWARE_MPI=0

if [ 1 -eq 1 ]
then
rm -rf ./output/*

mpirun -np 4 ./1_ref_gpu gpu 4 4 4 2 2 1 2

mpirun -np 4 ./2_gpu gpu 4 4 4 2 2 1 2 1 1 1


rm -rf ./output/*

mpirun -np 4 ./1_ref_gpu gpu 16 16 16 2 2 1 4
mpirun -np 4 ./2_gpu gpu 16 16 16 2 2 1 4 16 4 4


rm -rf ./output/*

mpirun -np 16 ./1_ref_gpu gpu 16 16 16 4 2 2 4
mpirun -np 16 ./2_gpu gpu 16 16 16 4 2 2 4 16 4 4



rm -rf ./output/*

mpirun -np 16 ./1_ref_gpu gpu 44 44 44 4 2 2 8
mpirun -np 16 ./2_gpu gpu 44 44 44 4 2 2 8 16 4 4


fi

#rm -rf ./output/*

#mpirun -np 16 ./2_gpu gpu 64 64 64 4 2 2 8 16 4 4

