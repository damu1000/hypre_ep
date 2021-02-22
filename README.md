# hypre_ep

## Hypre using MPI End Points:
This is the modified multi-threaded version of the Hypre linear solver library developed by LLNL (https://github.com/hypre-space/hypre). This version treats each thread as one MPI End Point instead of using openmp to parallelize data-parallel loops to achive faster performance. Experiments use two main components of combustion simulation done in Uintah (https://github.com/Uintah). More details are presented in "D. Sahasrabudhe, R. Zambre, A. Chandramowlishwaran, M. Berzins. In Journal of Computational Science, Springer International Publishing, pp. 101279. 2020. DOI: https://doi.org/10.1016/j.jocs.2020.101279"

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
export OMP_NESTED=true
  
mpirun -np 4 ./2_ep ep_run input.xml

To compare the accuracy of ep, set to 1 in input.xml. Create a folder "output" in hypre_test_v2 and run 1_cpu and 2_ep using same number of patches. 1_cpu will write results into files and 2_ep will compare its own results with those dumped in the file.


3. hypre_cpu_custom_ep.cc: This version uses lightweight custom_parallel_for instead of openmp - runs faster than openmp by 20 to 30%. This is similar to ICCS version. Set BOXLOOP_VER to 5 in hypre/src/struct_mv/boxloop.h and run headers. Optionaly directly update hypre/src/struct_mv/_hypre_struct_mv.h, which is bad but works for now. Compile same as hypre_cpu_ep.cc. Thread binding to the correct cores is absolutely critical for this version and is tricky while using optimizations such as hierarchical parallelism and funneled comm. Do not use OMP_PROC_BIND and OMP_PLACES. Instead use environment variables HYPRE_BINDING (used for EP threads and worker threads) and HYPRE_BINDING_COMM (used for comm threads) to give list of cores. The list will be equally divided among the ranks on a node and further among threads of the rank. Ensure the total number of threads (excluding comm thread) less than or equal to the number of cores specified in the list.

e.g. On a 16 core node with 2 hw threads per core, if 2 ranks are spawned with 2 EPs/rank with 4 worker threads/EP, then total 16 threads will do the computation and 2 threads (1 per rank) will funnel the comm. Use export HYPRE_BINDING=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 and export HYPRE_BINDING_COMM=31,32. This will give the following binding:

rank 0: cores 0-7 and 31, rank 1: cores 8-15 and 32

EP 0: 0-3, EP 1: 4-7, EP2: 8-11, and EP3: 12-15 with 4 worker threads of every EP mapped to individual cores with EP's cores.

Comm threads of ranks 0 and 1 are mapped to cores 31 and 32 respectively.

Compilation, inputs and execution is same as hypre_cpu_ep.cc

4. hypre_gpu.cc: To be used for hypre ep cuda code. Kokkos needed. In progress.


### Optimization switches:
Enable/disable following optimizations.

- Inter-thread comm: Exchanges data among threads (EPs) of the same rank without using MPI. Enable/disable with macro USE_INTER_THREAD_COMM in /src/utilities/hypre_mpi_ep_helper.h

- Multiple MPI comms: Use multiple MPI communicators to avoid waiting for locks within MPI and also exploite network parallelism in MPI 3.1. Uses a unique communicator in each direction. Enable/disable with macro USE_INTER_THREAD_COMM in /src/utilities/hypre_mpi_ep_helper.h

- Odd-even comms: Uses odd-even assignment of MPI communicator (similar to Rohit Zambre's work).  Enable/disable with macro USE_ODD_EVEN_COMMS in /src/utilities/hypre_mpi_ep_helper.h

- Tag-based parallelism: Specify hints to the MPI library to use information encoded in MPI tags to map to multiple network resources. Enable/disable with macro USE_TAG_PAR_COMM in /src/utilities/hypre_mpi_ep_helper.h. Only USE_TAG_PAR_COMM or USE_MULTIPLE_COMMS at a time. Both enabled will lead to erroneous behavior.

- Efficient patch assignment: Inter-rank communication can be reduced by using block-wise patch assignment in 3 dimensions rather than assigning patches linearly. xthreads, ythreads, zthreads parameters in the input file can be used to do so.

- Hierarchical parallelism - The main problem with out-of-box OpenMP implementation of parallelizing data parallel loops using OpenMP is not having enough work to justify OpenMP threads synchronization. The multi-grid algorithm in Hypre goes on coarsening the mesh. As a result, the number of cells reduce at every multi-grid level and OpenMP synchronization dominates than the actual computations at coarser levels resulting into slowdowns. That's where the EP model helps where each thread acts as a rank and no thread syncrhonization (except MPI_reduce) is needed. Yet using small number of threads (2 to 4) can be helpful to run data parallel loops for thoughput and also reduce communication - "a sweet spot" between both the models. That's when upto 1.4x speedup can be seen. Use parameter "team_size" in input.xml to set number of worker threads.

- Adaptive hierarchical parallelism (In Progress): Each EP now spawns its own worker threads (thus each EP is a team of threads now rather than a single thread), but "adaptive" nature uses OpenMP pragma only if number of loop iterations are more than the value set in environment variable "HYPRE_MIN_WORKLOAD". The downside is CPU cores remain unused when the number of cells are less than HYPRE_MIN_WORKLOAD. Need to find a sweet spot with number of EPs per rank and team_size so as to keep the cost of global reductions and local openmp synchronization overheads minimal. Need to see whether changing number of threads at run-time based on the loop count helps.

- Vectorization: Enable vectorization of few key kernels (which is not done by default): Add "-DUSE_SIMD" to CFLAGS and CXXFLAGS. Do this as the last optimization while checking incremental performance of different optimizations. Depending on problem size and communication pattern vectorization can give upto 20% speed-up on KNL. (Of corse the same can be done on CPU only version also)





## Install Uintah:

*** Do not get confused with the directory name kokkos_src. It is Uintah source code. Named kokkos_src by mistake. ***

cd uintah

mkdir work_dir

### Install cpu version

mkdir 1_cpu

cd 1_cpu

../kokkos_src_original/configure --enable-64bit --enable-optimize="-std=c++11 -O2 -g -fp-model precise -xMIC-AVX512" --enable-assertion-level=0 --with-mpi=built-in --with-hypre=../../hypre_cpu/src/build/ CC=mpiicc CXX=mpiicpc F77=ifort --no-create --no-recursion

make -j32 sus

cp ./StandAlone/sus ../work_dir/1_cpu


### Install kokkos OpenMP

// warning: the kokkos installation script is not tested with the latest version, but should work.

git clone https://github.com/kokkos/kokkos.git

cd kokkos

mkdir build

cd build

// --with-hwloc is optional. Provid if hwloc is available

../generate_makefile.bash --kokkos-disable-warnings --prefix=`pwd` --kokkos-path=../ --with-openmp --with-serial --with-options=disable_profiling --arch=KNL --with-hwloc=$HOME/installs/hwloc/install

make 

make install

### Install EP version

cd ..

mkdir 2_ep

cd 2_ep

//Provide kokkos path of earlier installation. "-L$HOME/installs/hwloc/install/lib -lhwloc -ldl" is optional. Provide if given during kokkos installation 

../kokkos_src/configure --enable-64bit --enable-optimize="-std=c++11 -O2 -g -fp-model precise -xMIC-AVX512" --enable-assertion-level=0 --with-mpi=built-in --with-hypre=../../hypre_ep/src/build/ LDFLAGS="-L$HOME/installs/kokkos-knl/build/kokkos/lib -lkokkos -L$HOME/installs/hwloc/install/lib -lhwloc -ldl" CXXFLAGS="-I$HOME/installs/kokkos-knl/build/kokkos/include -DUINTAH_ENABLE_KOKKOS -I../../hypre_ep/src/"

make -j32 sus

cp ./StandAlone/sus ../work_dir/2_ep


### to clean Uintah installations:

make clean

make cleanreally

make reallyclean






















