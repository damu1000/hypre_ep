# hypre_ep

## Hypre using MPI End Points:
This is the modified multi-threaded version of the Hypre linear solver library developed by LLNL (https://github.com/hypre-space/hypre). This version treats each thread as one MPI End Point instead of using openmp to parallelize data-parallel loops to achive faster performance. More details are presented in "Sahasrabudhe D., Berzins M. (2020) Improving Performance of the Hypre Iterative Solver for Uintah Combustion Codes on Manycore Architectures Using MPI Endpoints and Kernel Consolidation. In: Krzhizhanovskaya V. et al. (eds) Computational Science – ICCS 2020. ICCS 2020. Lecture Notes in Computer Science, vol 12137. Springer, Cham. https://doi.org/10.1007/978-3-030-50371-0_13"

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

### hypre_test_v2: 

Contains mini-apps to construct a 3d grid and Laplace equation, call Hypre to get the solution and measure the performance.
1. hypre_cpu.cc: MPI everywhere version (uses single cpu thread per rank). Should be linked to hypre_cpu. Requires libxml2. Compile using:

mpicxx hypre_cpu.cc -std=c++11 -I<...>/hypre_cpu/src/build/include/ -L<...>/hypre_cpu/src/build/lib -lHYPRE -I<libxml2 path>/include/libxml2 -L<libxml2 path>/lib -lxml2 -g -O3 -fopenmp -ldl -o 1_cpu

Run as:

mpirun -np <#rank> ./1_cpu <id> <input_xml_name>
  
e.g. mpirun -np 16 ./1_cpu cpu_run input.xml

The command spawns 16 MPI ranks and uses sample hypre_test_v2/input.xml. It creates a grid of total 16 patches arranged as 4x2x2 patches along x, y and z dimensions. Each patch is of size 32 cubed. "cpu_run" identifies the output - "cpu_run" can be any string. Ensure total number of patches are divisible by number of ranks. 

2. hypre_cpu_ep.cc:  hypre_cpu.cc converted into hypre_cpu_ep.cc by creating extra openmp threads and a calling couple of additional functions. Link with hypre_ep. Each openmp thread is assigned with one or more patches calls hypre (contrary to hypre_cpu) for its own patches. Requires libxml2. Compile using:

mpicxx hypre_cpu_ep.cc -std=c++11 -I<...>/hypre_ep/src/build/include/ -I<...>/hypre_ep/src/ -L<...>/hypre_ep/src/build/lib -lHYPRE -I<libxml2 path>/include/libxml2 -L<libxml2 path>/lib -lxml2 -g -O3 -o 2_ep -fopenmp

Run similar to 1_cpu, except now there will some threads instead of ranks. There could be different combinations of #threads and #ranks. Ensure that the number of patches is divisible by (#ranks*#threads) e.g. Following lines will spawn 4 mpi ranks with 4 threads per rank. Each thread (or EP) will get one patch:

export OMP_NUM_THREADS=4
  
mpirun -np 4 ./2_ep ep_run input.xml

To compare the accuracy of ep: Uncomment calls to functions write_to_file and verifyX in files hypre_cpu.cc and hypre_cpu_ep.cc respectively. Create a folder "output" in hypre_test_v2 and run 1_cpu and 2_ep using same number of patches and same number of ranks/EndPoints. 1_cpu will write results into files and 2_ep will compare its own results with those dumped in the file.

3. hypre_gpu.cc: To be used for hypre ep cuda code. Kokkos needed. In progress.

