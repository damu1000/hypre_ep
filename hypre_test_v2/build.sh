#!/bin/bash
export MPICH_CXX=nvcc_wrapper
mpicxx hypre_cpu.cc -std=c++11 -I/home/damodars/uintah_kokkos_dev/hypre/hypre_cpu/build/include/ -L/home/damodars/uintah_kokkos_dev/hypre/hypre_cpu/build/lib -lHYPRE -g -O3 -I$KOKKOS_PATH/build/include -L$KOKKOS_PATH/build/lib -lkokkos --expt-extended-lambda -o 1_cpu -fopenmp &

#mpicxx hypre_gpu.cc -std=c++11 -I/home/damodars/uintah_kokkos_dev/hypre/hypre_kokkos/src/build/include/ -L/home/damodars/uintah_kokkos_dev/hypre/hypre_kokkos/src/build/lib -lHYPRE -g -O3 -I$KOKKOS_PATH/build/include -L$KOKKOS_PATH/build/lib -lkokkos --expt-extended-lambda -o gpu_kokkos -arch=sm_52 -fopenmp &

mpicxx hypre_gpu.cc -std=c++11 -I/home/damodars/hypre_project/hypre_cuda/build/include/ -L/home/damodars/hypre_project/hypre_cuda/build/lib -lHYPRE -g -O3 -I$KOKKOS_PATH/build/include -L$KOKKOS_PATH/build/lib -lkokkos --expt-extended-lambda -o 2_cuda_orig -arch=sm_52 -lcusparse -lcudart -lcublas -lnvToolsExt -fopenmp &

mpicxx hypre_gpu_ep.cc -std=c++11 -I/home/damodars/uintah_kokkos_dev/hypre/hypre_cuda/src/build/include/ -I/home/damodars/uintah_kokkos_dev/hypre/hypre_cuda/src -L/home/damodars/uintah_kokkos_dev/hypre/hypre_cuda/src/build/lib -lHYPRE -g -O3 -I$KOKKOS_PATH/build/include -L$KOKKOS_PATH/build/lib -lkokkos --expt-extended-lambda -o 3_cuda_ep -arch=sm_52 -lcusparse -lcudart -lcublas -lnvToolsExt -fopenmp --default-stream per-thread
