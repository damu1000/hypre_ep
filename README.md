# hypre_ep

## Hypre using MPI End Points:
This is the modified multi-threaded version of the Hypre linear solver library developed by LLNL (https://github.com/hypre-space/hypre). This version treats each thread as one MPI End Point instead of using openmp to parallelize data-parallel loops to achive faster performance. More details are presented in "Sahasrabudhe D., Berzins M. (2020) Improving Performance of the Hypre Iterative Solver for Uintah Combustion Codes on Manycore Architectures Using MPI Endpoints and Kernel Consolidation. In: Krzhizhanovskaya V. et al. (eds) Computational Science – ICCS 2020. ICCS 2020. Lecture Notes in Computer Science, vol 12137. Springer, Cham. https://doi.org/10.1007/978-3-030-50371-0_13"

Hypre v2.15.1 is modified to use MPI EP.

## Directory structure and compilation instructions:
### hypre_ep/src: 
This is the updated Endpoint version of Hypre 2.15.1. Configure using:

cd hypre_ep/src

mkdir build

./configure --prefix=`pwd`/build CC=mpicxx CXX=mpicxx F77=mpif77 CFLAGS="-fPIC -O3 -g" CXXFLAGS="-fPIC -O3 -g"

Configure on with intel compiler on AVX512 supported architectures: Add "-fp-model precise -xMIC-AVX512" to CFLAGS and CXXFLAGS.

make -j32 install

### hypre_cpu/src:
This is the baseline CPU version with timing code added in MPI wrappers to measure communication time. 

Configure line for MPI everywhere (1 thread per mpi rank): ./configure --prefix=`pwd`/build CC=mpicxx CXX=mpicxx F77=mpif77 CFLAGS="-fPIC -O3 -g" CXXFLAGS="-fPIC -O3 -g"

Configure line for MPI + OpenMP: ./configure --prefix=`pwd`/build CC=mpicxx CXX=mpicxx F77=mpif77 CFLAGS="-fPIC -O3 -g" CXXFLAGS="-fPIC -O3 -g" --with-openmp

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

export OMP_PROC_BIND=spread,spread
export OMP_PLACES=threads
export OMP_STACKSIZE=1G
export OMP_NESTED=true
  
mpirun -np 4 ./2_ep ep_run input.xml

To compare the accuracy of ep: Uncomment calls to functions write_to_file and verifyX in files hypre_cpu.cc and hypre_cpu_ep.cc respectively. Create a folder "output" in hypre_test_v2 and run 1_cpu and 2_ep using same number of patches and same number of ranks/EndPoints. 1_cpu will write results into files and 2_ep will compare its own results with those dumped in the file.

3. hypre_gpu.cc: To be used for hypre ep cuda code. Kokkos needed. In progress.


### Optimization switches:
Enable/disable following optimizations.

- Inter-thread comm: Exchanges data among threads (EPs) of the same rank without using MPI. Enable/disable with macro USE_INTER_THREAD_COMM in /src/utilities/hypre_mpi_ep_helper.h

- Multiple MPI comms: Use multiple MPI communicators to avoid waiting for locks within MPI and also exploite network parallelism in MPI 3.1. Uses a unique communicator in each direction. Enable/disable with macro USE_INTER_THREAD_COMM in /src/utilities/hypre_mpi_ep_helper.h

- Odd-even comms: Uses odd-even assignment of MPI communicator (similar to Rohit Zambre's work).  Enable/disable with macro USE_ODD_EVEN_COMMS in /src/utilities/hypre_mpi_ep_helper.h

- Efficient patch assignment: Inter-rank communication can be reduced by using block-wise patch assignment in 3 dimensions rather than assigning patches linearly. xthreads, ythreads, zthreads parameters in the input file can be used to do so.

- Adaptive hierarchical parallelism: The main problem with out-of-box OpenMP implementation of parallelizing data parallel loops using OpenMP is not having enough work to justify OpenMP threads synchronization. The multi-grid algorithm in Hypre goes on coarsening the mesh. As a result, the number of cells reduce at every multi-grid level and OpenMP synchronization dominates than the actual computations at coarser levels. Yet such data parallel loops can be benifitial at scale because they can reduce number of ranks (or number of EPs) and reduce global reduction costs (MPI_allreduce), provided OpenMP synchronization overheads are reduced. (Global reduction takes up-to 30% of execution time on 512 nodes of Theta.) The new mechanism uses two level OpenMP parallelism. Each EP now spawns its own worker threads (thus each EP is a team of threads now rather than a single thread), but "adaptive" nature uses OpenMP pragma only if number of loop iterations are more than the value set in environment variable "HYPRE_MIN_WORKLOAD". Uses parameter "team_size" in input.xml to set number of worker threads. The downside is CPU cores remain unused when the number of cells are less than HYPRE_MIN_WORKLOAD. Enable by adding "--with-openmp" in configure line of hypre_ep. Can be disabled at runtime my setting very high value for HYPRE_MIN_WORKLOAD or by setting team_size in inputs.xml to 1. Need to find a sweet spot with number of EPs per rank and team_size so as to keep the cost of global reductions and local openmp synchronization overheads minimal.

- Vectorization: Enable vectorization of few key kernels (which is not done by default): Add "-DUSE_SIMD" to CFLAGS and CXXFLAGS. Do this as the last optimization while checking incremental performance of different optimizations. Depending on problem size and communication pattern vectorization can give upto 20% speed-up on KNL. (Of corse the same can be done on CPU only version also)


