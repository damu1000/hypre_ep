# hypre_ep

## Hypre using MPI End Points:
This is the multi-threaded version of the Hypre linear solver library from LLNL. This version treats each thread as one MPI End Point instead of using openmp to parallelize data-parallel loops to achive faster performance. More details are presented in "Sahasrabudhe D., Berzins M. (2020) Improving Performance of the Hypre Iterative Solver for Uintah Combustion Codes on Manycore Architectures Using MPI Endpoints and Kernel Consolidation. In: Krzhizhanovskaya V. et al. (eds) Computational Science – ICCS 2020. ICCS 2020. Lecture Notes in Computer Science, vol 12137. Springer, Cham. https://doi.org/10.1007/978-3-030-50371-0_13 https://link.springer.com/chapter/10.1007/978-3-030-50371-0_13"

Hypre v2.15.1 is modified to use MPI EP.

## Directory structure and compilation instructions:
### hypre_ep/src: 
This is the updated Endpoint version of Hypre 2.15.1. Configure using:

cd hypre_ep/src

mkdir build

./configure --prefix=`pwd`/build CC=mpicxx CXX=mpicxx F77=mpif77 CFLAGS="-fPIC -O3-g" CXXFLAGS="-fPIC -O3 -g"

make -j32 install

### hypre_cpu/src:
This is the baseline CPU version with timing code added in MPI wrappers to measure communication time. 

Configure line for MPI everywhere (1 thread per mpi rank): ./configure --prefix=`pwd`/build CC=mpicxx CXX=mpicxx F77=mpif77 CFLAGS="-fPIC -O3-g" CXXFLAGS="-fPIC -O3 -g"

Configure line for MPI + OpenMP: ./configure --prefix=`pwd`/build CC=mpicxx CXX=mpicxx F77=mpif77 CFLAGS="-fPIC -O3-g" CXXFLAGS="-fPIC -O3 -g" --with-openmp

make -j32 install

### hypre_test_v2/src: 
Contains mini-apps to construct a 3d grid and Laplace equation, call Hypre to get the solution and measure the performance.
