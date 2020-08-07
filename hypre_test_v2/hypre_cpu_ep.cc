//Standalone code similar to Uintah "Hypre Solver" task invoked from Uintah example "solver test 1"
//Compile line:
//  mpicxx hypre_test.cc -std=c++11 -fopenmp -I/home/sci/damodars/installs/2_hypre-2.11.0_omp_original/src/hypre/include/ -L/home/sci/damodars/installs/2_hypre-2.11.0_omp_original/src/hypre/lib -lHYPRE -g -O2
//run as:
//  export OMP_NUM_THREADS=16
//	mpirun -np 4 ./a.out 64 4
//  This will create a grid of 64 cube cell divided among 4 cube number of patches. Thus each patch will be of size 16 cube. 16 omp threads per rank
//  For simplicity assuming cubical domain with same cells and patches in every dimension.
/*

Albion:
CPU Only
export MPICH_CXX=g++
export HYPRE_PATH=/home/damodars/uintah_kokkos_dev/hypre_kokkos/src/build

mpicxx hypre_test.cc -std=c++11 -I$(HYPRE_PATH)/include/ -L$(HYPRE_PATH)/lib -lHYPRE -g -O3

mpicxx 2_hypre_gpu.cc -std=c++11 -I/home/damodars/uintah_kokkos_dev/hypre_kokkos/src/build/include/ -L/home/damodars/uintah_kokkos_dev/hypre_kokkos/src/build/lib -lHYPRE -g -O3 -I$KOKKOS_PATH/build/include -L$KOKKOS_PATH/build/lib -lkokkos --expt-extended-lambda -o gpu -arch=sm_52 


mpicxx hypre_cpu_ep.cc -std=c++11 -I/home/damodars/hypre_ep/hypre_ep/src/build/include/ -I/home/damodars/hypre_ep/hypre_ep/src/ -L/home/damodars/hypre_ep/hypre_ep/src/build/lib -lHYPRE -g -O3 -o 2_ep -fopenmp

 */

//#include <cuda_profiler_api.h>


#include<chrono>
#include <ctime>
#include<cmath>
#include<vector>
#include<mpi.h>
#include<stdio.h>
#include<chrono>
#include <ctime>
#include<iostream>
#include<string.h>
// hypre includes
#include <_hypre_struct_mv.h>
#include <_hypre_utilities.h>
#include <HYPRE_struct_ls.h>
#include <krylov.h>
//#include <Kokkos_Core.hpp>
#include <omp.h>
#include <sched.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include "io.h"


//typedef Kokkos::Serial KernelSpace;
//typedef Kokkos::Cuda KernelSpace;

struct Stencil4{
	double p, n, e, t;
};

//typedef Kokkos::View<double*, Kokkos::LayoutRight, KernelSpace::memory_space> ViewDouble;
//typedef Kokkos::View<Stencil4*, Kokkos::LayoutRight, KernelSpace::memory_space> ViewStencil4;

double tolerance = 1.e-25;
char **argv;
int argc;

typedef struct IntVector
{
	int value[3]; //x: 0, y:1, z:2. Loop order z, y, x. x changes fastest.
} IntVector;

int get_affinity() {
	int core=-1;
	cpu_set_t mask;
	long nproc, i;

	if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
		perror("sched_getaffinity");
		assert(false);
	} else {
		nproc = sysconf(_SC_NPROCESSORS_ONLN);
		for (i = 0; i < nproc; i++) {
			core++;
			//printf("%d ", CPU_ISSET(i, &mask));
			if(CPU_ISSET(i, &mask))
				break;
		}
		//printf("affinity: %d\n", core);
	}
	return core;
}

//patches numbers are always assigned serially. Proc assignment logic may vary.
//index into g_patch_proc indicates patch id, values indicates assigned proc.
std::vector<int> g_patch_proc;

//as of now using simple logic of serial assignment
void assignPatchToProc(int x_patches, int y_patches, int z_patches, int num_of_ranks){
	int patch=0, rank=0;
	int number_of_patches = x_patches * y_patches * z_patches; 
	int patches_per_rank = number_of_patches / num_of_ranks;

	g_patch_proc.resize(x_patches * y_patches * z_patches);

	for(int i=0; i<z_patches; i++)
	for(int j=0; j<y_patches; j++)
	for(int k=0; k<x_patches; k++){
		g_patch_proc[patch++] = rank;
		if(patch % patches_per_rank == 0) rank++;
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	/*if(rank==0)
		for(int i=0; i<g_patch_proc.size(); i++)
			printf("patch assignment: %d %d\n", i, g_patch_proc[i]);*/
}

//Quick  and really really dirty. Try improving later. should be called after assignPatchToProc.
void getMyPatches( std::vector<int>& my_patches, int my_rank)
{
	for(int i=0; i<g_patch_proc.size(); i++){
		if(g_patch_proc[i] == my_rank)	my_patches.push_back(i);
	}
}

void findNeighbors(std::vector<RankDir> & neighbors, std::vector<int>& my_patches, int x_patches, int y_patches, int z_patches, int rank){
/*
1. Loop over my patches. For every patch:
2. Loop over (-1, -1, -1) to (1, 1, 1). Calculate offset for every direction (i, j, k) and add it to the current patch to get neighboring patch
3. Find proc of neighbor patch. Add it if its not same as self.
*/
	int tot_num_patches = g_patch_proc.size();

	for(int p : my_patches){	
		int iid = p / (y_patches*x_patches);
		int jid = (p % (y_patches*x_patches)) / x_patches;
		int kid = p % x_patches;

		for(int i=iid-1; i<iid+2; i++)
		for(int j=jid-1; j<jid+2; j++)
		for(int k=kid-1; k<kid+2; k++){
			if(i > -1 && i < z_patches && j > -1 && j < y_patches && k > -1 && k < x_patches){
				int neighbor = i * y_patches * x_patches + j * x_patches + k;
				int proc = g_patch_proc[neighbor];
				if(proc != rank){
					//printf("pushing neighbours %d %d %d %d %d\n",k-kid, j-jid, i-iid, rank, proc);
					neighbors.push_back(RankDir(k-kid, j-jid, i-iid, rank, proc)); //for send
					neighbors.push_back(RankDir(k-kid, j-jid, i-iid, proc, rank)); //for recv
					//if(rank==4) 	printf("neighbors %d %d %d %d %d\n", neighbor, i, j, k, p);
				}
			}
		}		
	}


	
}

void hypre_solve(const char * exp_type, int patch_dim, int x_patches, int y_patches, int z_patches)
{

	//--------------------------------------------------------- init ----------------------------------------------------------------------------------------------

	int number_of_patches = x_patches * y_patches * z_patches; 

	
	HYPRE_Init(argc, argv);

	int rank, size;
	hypre_MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	hypre_MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(number_of_patches % size != 0){
		printf("Ensure total number of patches (patches cube) is divisible number of ranks. exiting\n");
		exit(1);
	}

	int num_cells = number_of_patches*patch_dim*patch_dim*patch_dim;

	thread_local extern double _hypre_comm_time;
	//----------------------------------------------------------------------------------------------------------------------------------------------------------


	//----------------------------------------------- patch set up----------------------------------------------------------------------------------------------

	std::vector<int> my_patches;
	getMyPatches( my_patches, rank );

	std::vector<RankDir> neighbors;
  findNeighbors(neighbors, my_patches, x_patches, y_patches, z_patches, rank); //assuming EP.

	createCommMap(neighbors.data(), neighbors.size());	//in hypre_mpi_ep_helper.h

	int patches_per_rank = my_patches.size();
	int x_dim = patch_dim, y_dim = patch_dim, z_dim = patch_dim;

	std::vector<IntVector> low(patches_per_rank), high(patches_per_rank);	//store boundaries of every patch

	for(int i=0; i<patches_per_rank; i++)
	{
		//patch id based on local patch number (i) + starting patch assigned to this rank.
		int patch_id = my_patches[i];

		// convert patch_id into 3d patch co-cordinates. Multiply by patch_dim to get starting cell of patch.
		low[i].value[0] = x_dim * (patch_id % x_patches);
		low[i].value[1] = y_dim * ((patch_id / x_patches) % y_patches);
		low[i].value[2] = z_dim * ((patch_id / x_patches) / y_patches);

		// add patch_dim to low to get end cell of patch. subtract 1 as hypre uses inclusive boundaries... as per Uintah code
		high[i].value[0] = low[i].value[0] + x_dim - 1; //including high cell. hypre needs so.. i guess
		high[i].value[1] = low[i].value[1] + y_dim - 1;
		high[i].value[2] = low[i].value[2] + z_dim - 1;
		printf("%d/%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", rank, size, patch_id, low[i].value[0], low[i].value[1], low[i].value[2], high[i].value[0], high[i].value[1], high[i].value[2]);
	}

	//All patches will have same values. Hence just create 1 patch for A and 1 for B. Pass same values again and again for every patch.
	//ViewDouble X("X", x_dim * y_dim * z_dim), B("B", x_dim * y_dim * z_dim);
	//ViewStencil4 A("A", x_dim * y_dim * z_dim);
	Stencil4 A[x_dim * y_dim * z_dim];
	double X[x_dim * y_dim * z_dim], B[x_dim * y_dim * z_dim];

//	Kokkos::parallel_for(Kokkos::RangePolicy<KernelSpace>(0, x_dim * y_dim * z_dim), KOKKOS_LAMBDA(int i){
	for(int i=0; i<x_dim * y_dim * z_dim; i++){
			A[i].p = 6; A[i].n=-1; A[i].e=-1; A[i].t=-1;
			B[i] = 1;
			X[i] = 0;

	}
//	});

	//----------------------------------------------------------------------------------------------------------------------------------------------------------



	//----------------------------------------------------- hypre -----------------------------------------------------------------------------------------------

	extern thread_local bool do_setup;

	HYPRE_StructSolver * solver = new HYPRE_StructSolver;
	HYPRE_StructSolver * precond_solver = new HYPRE_StructSolver;
	HYPRE_StructMatrix * HA = new HYPRE_StructMatrix;
	HYPRE_StructVector * HB = new HYPRE_StructVector;
	HYPRE_StructVector * HX = new HYPRE_StructVector;
	HYPRE_StructGrid grid;
	int periodic[]={0,0,0};
	HYPRE_StructStencil stencil;
	int offsets[4][3] = {{0,0,0},
			{-1,0,0},
			{0,-1,0},
			{0,0,-1}};
	bool HB_created = false;
	bool HA_created = false;
	bool HX_created = false;

	for(int timestep=0; timestep<6; timestep++)
	{
		_hypre_comm_time = 0.0;
		auto start = std::chrono::system_clock::now();

		if(do_setup)
		{
			// Setup grid
			HYPRE_StructGridCreate(MPI_COMM_WORLD, 3, &grid);
			for(int i=0; i<patches_per_rank; i++)
				HYPRE_StructGridSetExtents(grid, low[i].value, high[i].value);

			HYPRE_StructGridSetPeriodic(grid, periodic);	// No Periodic boundaries

			HYPRE_StructGridAssemble(grid);	// Assemble the grid

			HYPRE_StructStencilCreate(3, 4, &stencil);	// Create the stencil
			for(int i=0;i<4;i++)
				HYPRE_StructStencilSetElement(stencil, i, offsets[i]);

			// Create the matrix
			if(HA_created)	//just in case if we have to loop over
				HYPRE_StructMatrixDestroy(*HA);
			HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, HA);
			HA_created = true;
			HYPRE_StructMatrixSetSymmetric(*HA, true);
			int ghost[] = {1,1,1,1,1,1};
			HYPRE_StructMatrixSetNumGhost(*HA, ghost);
			HYPRE_StructMatrixInitialize(*HA);

			// Create the RHS
			if(HB_created)	//just in case if we have to loop over
				HYPRE_StructVectorDestroy(*HB);
			HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, HB);
			HB_created = true;
			HYPRE_StructVectorInitialize(*HB);

			// setup the coefficient matrix
			int stencil_indices[] = {0,1,2,3};

			for(int i=0; i<patches_per_rank; i++){
				double *values = reinterpret_cast<double *>(A);
				HYPRE_StructMatrixSetBoxValues(*HA, low[i].value, high[i].value,
						4, stencil_indices,
						values);
			}

			HYPRE_StructMatrixAssemble(*HA);
		}//if(do_setup)

		//set up RHS

		for(int i=0; i<patches_per_rank; i++)
			HYPRE_StructVectorSetBoxValues(*HB, low[i].value, high[i].value, B);

		if(do_setup)
			HYPRE_StructVectorAssemble(*HB);



		if(do_setup)		// Create the solution vector
		{
			if(HX_created)	//just in case if we have to loop over
				HYPRE_StructVectorDestroy(*HX);
			HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, HX);
			HX_created=true;
			HYPRE_StructVectorInitialize(*HX);
			HYPRE_StructVectorAssemble(*HX);
		}

		//solve the system
		bool solver_created = false;

		if (do_setup)
		{
			if(solver_created) //just in case if we have to loop over
				HYPRE_StructPCGDestroy(*solver);
			HYPRE_StructPCGCreate(MPI_COMM_WORLD,solver);
			solver_created = true;
		}

		HYPRE_StructPCGSetMaxIter(*solver, 100);
		HYPRE_StructPCGSetTol(*solver, tolerance);
		HYPRE_StructPCGSetTwoNorm(*solver,  1);
		HYPRE_StructPCGSetRelChange(*solver,  0);
		HYPRE_StructPCGSetLogging(*solver,  0);

		HYPRE_PtrToStructSolverFcn precond;
		HYPRE_PtrToStructSolverFcn precond_setup;
		auto solve_only_start = std::chrono::system_clock::now();
		if (do_setup)
		{
			HYPRE_StructPFMGCreate        (MPI_COMM_WORLD,    precond_solver);
			HYPRE_StructPFMGSetMaxIter    (*precond_solver,   1);
			HYPRE_StructPFMGSetTol        (*precond_solver,   0.0);
			HYPRE_StructPFMGSetZeroGuess  (*precond_solver);

			/* weighted Jacobi = 1; red-black GS = 2 */
			HYPRE_StructPFMGSetRelaxType   (*precond_solver,  1);
			HYPRE_StructPFMGSetNumPreRelax (*precond_solver,  1);
			HYPRE_StructPFMGSetNumPostRelax(*precond_solver,  1);
			HYPRE_StructPFMGSetSkipRelax   (*precond_solver,  0);
			HYPRE_StructPFMGSetLogging     (*precond_solver,  0);

			precond = HYPRE_StructPFMGSolve;
			precond_setup = HYPRE_StructPFMGSetup;
			HYPRE_StructPCGSetPrecond(*solver, precond,precond_setup, *precond_solver);

			HYPRE_StructPCGSetup(*solver, *HA, *HB, *HX);

		}
		do_setup = false;
		//printf("%d do_setup false\n", omp_get_thread_num());
		//sleep(2);
		HYPRE_StructPCGSolve(*solver, *HA, *HB, *HX);	//main solve

		int num_iterations;
		HYPRE_StructPCGGetNumIterations(*solver, &num_iterations);

		double final_res_norm;
		HYPRE_StructPCGGetFinalRelativeResidualNorm(*solver,&final_res_norm);

		auto solve_only_end = std::chrono::system_clock::now();

		if(rank ==0 && (final_res_norm > tolerance || std::isfinite(final_res_norm) == 0))
		{
			std::cout << "HypreSolver not converged in " << num_iterations << " iterations, final residual= " << final_res_norm << "\n";
			exit(1);
		}

		for(int i=0; i<patches_per_rank; i++)
			HYPRE_StructVectorGetBoxValues(*HX, low[i].value, high[i].value, X);

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> comp_time = end - start;
		std::chrono::duration<double> solve_time = solve_only_end - solve_only_start;

		double t_comp_time, avg_comp_time=0.0, t_solve_time, avg_solve_time=0.0, avg_comm_time=0.0;
		t_comp_time = comp_time.count();
		t_solve_time = solve_time.count();

		hypre_MPI_Allreduce(&_hypre_comm_time, &avg_comm_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);			
		hypre_MPI_Allreduce(&t_comp_time, &avg_comp_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		hypre_MPI_Allreduce(&t_solve_time, &avg_solve_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		avg_comm_time /= size;
		avg_comp_time /= size;
		avg_solve_time /= size;

		std::cout.precision(2);
		//exp_type			//size			num_cells   		patch_dim   		x_patches    		y_patches   		z_patches
		//timestep			 total_time				solve_only_time			avg_comm_time				num_iterations			final_res_norm
		if(rank==0 && omp_get_thread_num()==0)
			std::cout <<
			exp_type << "\t" << size << "\t" << num_cells << "\t" << patch_dim << "\t" << x_patches << "\t" << y_patches << "\t" << z_patches << "\t" << 
			timestep << "\t" << avg_comp_time << "\t" << avg_solve_time << "\t" << avg_comm_time << "\t" << num_iterations << "\t" << final_res_norm << 
			"\n" ;

		//verifyX(X, x_dim * y_dim * z_dim);
	}//for(int timestep=0; timestep<11; timestep++)


	//----------------------------------------------------------------------------------------------------------------------------------------------------------

	//clean up
	HYPRE_StructStencilDestroy(stencil);
	HYPRE_StructGridDestroy(grid);
	hypre_EndTiming (tHypreAll_);

	fflush(stdout);
	HYPRE_Finalize();
}	//end of hypre_solve


int main(int argc1, char **argv1)
{
	if(argc1 != 6){
		printf("Enter arguments patch size and number of patches. exiting\n");
		exit(1);
	}
	argc = argc1;
	argv = argv1;
	
	const char * exp_type = argv[1];	//cpu / gpu / gpu-mps / gpu-superpatch. Only to print in output.
	int patch_dim = atoi(argv[2]);	//patch size
	int x_patches = atoi(argv[3]);	//number of patches in X dimension
	int y_patches = atoi(argv[4]);	//number of patches in Y dimension
	int z_patches = atoi(argv[5]);	//number of patches in Z dimension

//	Kokkos::initialize(argc, argv1);
	{
		int required=MPI_THREAD_MULTIPLE, provided;

		MPI_Init_thread(&argc, &argv, required, &provided);
		
		int partitions = omp_get_max_threads();
		//Kokkos::OpenMP::partition_master( hypre_solve, omp_get_num_threads(), 1);
		
//		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0, partitions), [&](int i){
//			hypre_solve(partitions, 1);
//		});

		int size;
		MPI_Comm_size(MPI_COMM_WORLD, &size);

		assignPatchToProc(x_patches, y_patches, z_patches, size*partitions); //assuming EP.

		hypre_set_num_threads(partitions, omp_get_thread_num);
		//cudaProfilerStart();
#pragma omp parallel
		{
			/*int cpu_affinity = get_affinity();
			int device_id=-1;
			cudaGetDevice(&device_id);
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, device_id);
			printf("Device id: %d,  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d, cpu_affinity: %d\n", 
					device_id, deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID, cpu_affinity);*/
			
			hypre_solve(exp_type, patch_dim, x_patches, y_patches, z_patches);
		}
		//cudaProfilerStop();
		
		MPI_Finalize();			


	}
//	Kokkos::finalize();
	return 0;

}
