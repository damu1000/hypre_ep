#!/bin/bash


CC=$1

if [ -z $CC ]
then
	CC=0
fi

export NVCC_WRAPPER_DEFAULT_COMPILER=mpic++



#nvcc_wrapper hypre_ref.cc -lineinfo -std=c++11 -I$HOME/hypre_project/hypre_cuda/build/include/ -L$HOME/hypre_project/hypre_cuda/build/lib -lHYPRE -g -I$KOKKOS_PATH/build/include -L$KOKKOS_PATH/build/lib -lkokkos --expt-extended-lambda -arch=sm_52 -lcusparse -lcudart -lcublas -lnvToolsExt -DCC=$CC -fopenmp  -o 1_ref_gpu -O2 -pg &


#nvcc_wrapper hypre_gpu.cc -lineinfo -std=c++11 -I$HOME/uintah_kokkos_dev/hypre/hypre_cuda/src/build/include/ -L$HOME/uintah_kokkos_dev/hypre/hypre_cuda/src/build/lib -lHYPRE -g -I$KOKKOS_PATH/build/include -L$KOKKOS_PATH/build/lib -lkokkos --expt-extended-lambda -o 2_gpu -arch=sm_52 -lcusparse -lcudart -lcublas -lnvToolsExt -DCC=$CC -fopenmp -O2 -pg


nvcc -ccbin=mpicxx hypre_ref.cc -lineinfo -std=c++11 -I$HOME/hypre_project/hypre_cuda/build/include/ -L$HOME/hypre_project/hypre_cuda/build/lib -lHYPRE -g --expt-extended-lambda -arch=sm_52 -lcusparse -lcudart -lcublas -lnvToolsExt -DCC=$CC -o 1_ref_gpu -O2 -pg -x cu -Xcompiler -fopenmp &


nvcc -ccbin=mpicxx hypre_gpu.cc -lineinfo -std=c++11 -I$HOME/uintah_kokkos_dev/hypre/hypre_cuda/src/build/include/ -L$HOME/uintah_kokkos_dev/hypre/hypre_cuda/src/build/lib -lHYPRE -g --expt-extended-lambda -o 2_gpu -arch=sm_52 -lcusparse -lcudart -lcublas -lnvToolsExt -DCC=$CC -x cu -Xcompiler -fopenmp -O0 

