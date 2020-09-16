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


 mpicxx hypre_cpu_ep.cc -std=c++11 -I/home/damodars/hypre_ep/hypre_ep/src/build/include/ -I/home/damodars/hypre_ep/hypre_ep/src/ -L/home/damodars/hypre_ep/hypre_ep/src/build/lib -lHYPRE -I/home/damodars/install/libxml2-2.9.7/build/include/libxml2 -L/home/damodars/install/libxml2-2.9.7/build/lib -lxml2  -g -O3 -o 2_ep -fopenmp

CC hypre_cpu_ep.cc -std=c++11 -fp-model precise -xMIC-AVX512 -I/home/damodars/hypre_ep/hypre_ep/src/build/include/ -I/home/damodars/hypre_ep/hypre_ep/src -L/home/damodars/hypre_ep/hypre_ep/src/build/lib -lHYPRE -lxml2 -g -O3 -fopenmp -ldl -o 2_ep -dynamic

mpicxx hypre_cpu_ep.cc -std=c++11 -fp-model precise -xMIC-AVX512 -I/home/sci/damodars/hypre_ep/hypre_ep/src -I/home/sci/damodars/hypre_ep/hypre_ep/src/build/include/ -I/home/sci/damodars/installs/libxml2-2.9.7/build/include/libxml2/ -L/home/sci/damodars/hypre_ep/hypre_ep/src/build/lib -lHYPRE  -L/home/sci/damodars/installs/libxml2-2.9.7/build/lib/ -lxml2 -g -O3 -fopenmp -ldl -o 2_ep


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

//#define USE_FUNNELLED_COMM
/*do not define USE_FUNNELLED_COMM here. use -DUSE_FUNNELLED_COMM compiler option instead*/

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

void set_affinity( const int proc_unit )
{
  cpu_set_t mask;
  unsigned int len = sizeof(mask);
  CPU_ZERO(&mask);
  CPU_SET(proc_unit, &mask);
  sched_setaffinity(0, len, &mask);
}

//patches numbers are always assigned serially. EP assignment logic may vary.
//index into g_patch_ep indicates patch id, values indicates assigned EP.
std::vector<int> g_patch_ep;
std::vector<int> g_ep_superpatch; //ep to superpatch mapping
int g_superpatches[3];
int g_num_of_ranks3d[3];

#define OneDtoThreeD(t, p, x_patches, y_patches, z_patches)	\
	t[2] = p / (y_patches*x_patches);						\
	t[1] = (p % (y_patches*x_patches)) / x_patches;			\
	t[0] = p % x_patches;

//as of now using simple logic of serial assignment
void assignPatchToEP(xmlInput input, int rank, int size, int num_of_threads){

	int num_of_eps = size * num_of_threads; // this number of end points
	int num_of_patches = input.xpatches * input.ypatches * input.zpatches;
	int patches_per_ep = num_of_patches / num_of_eps;
	int num_of_psets = num_of_patches / num_of_threads; //number of patch sets
	int num_of_psets_per_rank = num_of_psets / size;
	int xpsets = input.xpatches / input.xthreads,  ypsets = input.ypatches / input.ythreads,  zpsets = input.zpatches / input.zthreads;

	if(num_of_eps > num_of_patches){
		printf("Ensure that num of end points <= num of patches\n"); exit(1);
	}

	//bulletproofing: Ensure num_of_psets_per_rank is aligned with xpsets, ypsets and zpsets
	g_num_of_ranks3d[0] = xpsets, g_num_of_ranks3d[1] = ypsets, g_num_of_ranks3d[2] = zpsets;
	int temp = num_of_psets_per_rank;

	for(int i=0; i<3; i++){
		if(g_num_of_ranks3d[i] >= temp){ //patches fit within the given dimension. recompute and break
			if(g_num_of_ranks3d[i] % temp != 0){ //patch dim should be multiple of patches_per_ep
				printf("Part 1: number of psets per rank should be aligned with patch dimensions at %s:%d\n", __FILE__, __LINE__);exit(1);
			}
			g_num_of_ranks3d[i] = g_num_of_ranks3d[i] / temp;
			temp = 1;
			break;
		}
		else{//there can be multiple rows or planes assigned to one EP
			if(temp % g_num_of_ranks3d[i] != 0){ //now patches_per_ep should be multiple of patch_dim
				printf("Part 2: number of psets per rank should be aligned with patch dimensions at %s:%d\n", __FILE__, __LINE__);exit(1);
			}
			temp = temp / g_num_of_ranks3d[i];
			g_num_of_ranks3d[i] = 1; //current dimension will be 1 because no division across this dimension
		}
	}

	if(temp !=1 || size != g_num_of_ranks3d[0]*g_num_of_ranks3d[1]*g_num_of_ranks3d[2]){
		printf("Part 3: Error in superpatch generation logic at %s:%d\n", __FILE__, __LINE__);exit(1);
	}

	g_patch_ep.resize(input.xpatches * input.ypatches * input.zpatches);
	g_ep_superpatch.resize(num_of_eps);

	//Each rank will hold patchsets from rank*num_of_psets_per_rank to rank*num_of_psets_per_rank + num_of_psets_per_rank - 1
	//bulletproofing ensures patchset assignment among ranks is aligned and continuous. So ranks can be converted directly to 3D coordinates for odd-even logic
	//patch assignment
	for(int rank=0; rank<size; rank++){
		//convert low and high patchset ids to 3d and multiply by xthreads, ythreads, zthreads to find out first and the last patch ids of the rank
		int plow[3], phigh[3], pslownum = rank*num_of_psets_per_rank, pshighnum = rank*num_of_psets_per_rank + num_of_psets_per_rank-1;
		OneDtoThreeD(plow, pslownum, xpsets, ypsets, zpsets);	//plow has 3d patch set id of the first patch of the rank;
		OneDtoThreeD(phigh, pshighnum, xpsets, ypsets, zpsets);	//phigh has 3d patch set id of the last patch of the rank;
		//Now multiply by xthreads, ythreads, zthreads to find out first and the last patch ids of the rank
		plow[0] *= input.xthreads; plow[1] *= input.ythreads; plow[2] *= input.zthreads;
		phigh[0] *= input.xthreads; phigh[1] *= input.ythreads; phigh[2] *= input.zthreads;
		phigh[0] += input.xthreads; phigh[1] += input.ythreads; phigh[2] += input.zthreads; //Add xthreads, ythreads, zthreads to take high to the last patch within the pset
		int tid = 0, count=0;
		//now iterate over patches and assign EPs.
		for(int k=plow[2]; k<phigh[2]; k++)
		for(int j=plow[1]; j<phigh[1]; j++)
		for(int i=plow[0]; i<phigh[0]; i++){
			int patch = k * input.xpatches * input.ypatches + j * input.xpatches + i;
			int ep = rank * num_of_threads + tid;
			g_patch_ep[patch] = ep;
			g_ep_superpatch[ep] = patch / patches_per_ep;
			count++;
			if(count % patches_per_ep == 0) tid++;
			//calculate super patches within ep to find out directions
		}
	}

//	if(rank==0){
//		printf("ranks shape: %d %d %d\n", g_num_of_ranks3d[0], g_num_of_ranks3d[1], g_num_of_ranks3d[2]);
//		for(int i=0; i<g_patch_ep.size(); i++)
//			printf("patch %d EP %d\n", i, g_patch_ep[i]);
//
//		for(int i=0; i<g_ep_superpatch.size(); i++)
//			printf("EP %d superpatch %d\n", i, g_ep_superpatch[i]);
//	}


	//compute number of number of superpatches in each dimension. Needed to convert super patch number to 3d during comm mapping.
	//same logic as used earlier to compute number of ranks in each direction
	g_superpatches[0] = input.xpatches, g_superpatches[1] = input.ypatches, g_superpatches[2] = input.zpatches;

	int spatches_per_ep = patches_per_ep;
	for(int i=0; i<3; i++){
		if(g_superpatches[i] >= spatches_per_ep){ //patches fit within the given dimension. recompute and break
			if(g_superpatches[i] % spatches_per_ep != 0){ //patch dim should be multiple of patches_per_ep
				printf("Part 1: number of patches per EP should be aligned with patch dimensions at %s:%d\n", __FILE__, __LINE__);exit(1);
			}
			g_superpatches[i] = g_superpatches[i] / spatches_per_ep;
			spatches_per_ep = 1;
			break;
		}
		else{//there can be multiple rows or planes assigned to one EP
			if(spatches_per_ep % g_superpatches[i] != 0){ //now patches_per_ep should be multiple of patch_dim
				printf("Part 2: number of patches per EP should be aligned with patch dimensions at %s:%d\n", __FILE__, __LINE__);exit(1);
			}
			spatches_per_ep = spatches_per_ep / g_superpatches[i];
			g_superpatches[i] = 1; //current dimension will be 1 because no division across this dimension
		}
	}

	if(spatches_per_ep !=1 || num_of_eps != g_superpatches[0]*g_superpatches[1]*g_superpatches[2]){
		printf("Part 3: Error in superpatch generation logic at %s:%d\n", __FILE__, __LINE__);exit(1);
	}
}

//Quick  and really really dirty. Try improving later. should be called after assignPatchToEP.
void getMyPatches( std::vector<int>& my_patches, int my_rank)
{
	for(int i=0; i<g_patch_ep.size(); i++){
		if(g_patch_ep[i] == my_rank)	my_patches.push_back(i);
	}
}


void hypre_solve(const char * exp_type, xmlInput input)
{

	//--------------------------------------------------------- init ----------------------------------------------------------------------------------------------

	int number_of_patches = input.xpatches * input.ypatches * input.zpatches;
	int num_of_threads = input.xthreads * input.ythreads * input.zthreads;

	
	HYPRE_Init(argc, argv);

	if(omp_get_thread_num()==num_of_threads) //this is comm thread so dont call hypre, just return
		return;

	int rank, size;
	hypre_MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	hypre_MPI_Comm_size(MPI_COMM_WORLD, &size);
//	set_affinity(rank);

	if(number_of_patches % size != 0){
		printf("Ensure total number of patches (patches cube) is divisible number of ranks at %s:%d. exiting\n", __FILE__, __LINE__);
		exit(1);
	}

	int num_cells = number_of_patches*input.patch_size*input.patch_size*input.patch_size;

	thread_local extern double _hypre_comm_time;
	//----------------------------------------------------------------------------------------------------------------------------------------------------------


	//----------------------------------------------- patch set up----------------------------------------------------------------------------------------------

	std::vector<int> my_patches;
	getMyPatches( my_patches, rank );

	createCommMap(g_ep_superpatch.data(), g_superpatches, g_num_of_ranks3d, input.xthreads, input.ythreads, input.zthreads);	//in hypre_mpi_ep_helper.h

	int patches_per_rank = my_patches.size();
	int x_dim = input.patch_size, y_dim = input.patch_size, z_dim = input.patch_size;

	std::vector<IntVector> low(patches_per_rank), high(patches_per_rank);	//store boundaries of every patch

	for(int i=0; i<patches_per_rank; i++)
	{
		//patch id based on local patch number (i) + starting patch assigned to this rank.
		int patch_id = my_patches[i];

		// convert patch_id into 3d patch co-cordinates. Multiply by patch_size to get starting cell of patch.
		low[i].value[0] = x_dim * (patch_id % input.xpatches);
		low[i].value[1] = y_dim * ((patch_id / input.xpatches) % input.ypatches);
		low[i].value[2] = z_dim * ((patch_id / input.xpatches) / input.ypatches);

		// add patch_size to low to get end cell of patch. subtract 1 as hypre uses inclusive boundaries... as per Uintah code
		high[i].value[0] = low[i].value[0] + x_dim - 1; //including high cell. hypre needs so.. i guess
		high[i].value[1] = low[i].value[1] + y_dim - 1;
		high[i].value[2] = low[i].value[2] + z_dim - 1;
		//printf("%d/%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", rank, size, patch_id, low[i].value[0], low[i].value[1], low[i].value[2], high[i].value[0], high[i].value[1], high[i].value[2]);
	}

	//ViewDouble X("X", x_dim * y_dim * z_dim), B("B", x_dim * y_dim * z_dim);
	//ViewStencil4 A("A", x_dim * y_dim * z_dim);
	Stencil4 *A = new Stencil4[x_dim * y_dim * z_dim * patches_per_rank];
	double *X = new double[x_dim * y_dim * z_dim * patches_per_rank];
	double *B = new double[x_dim * y_dim * z_dim * patches_per_rank];

	//	Kokkos::parallel_for(Kokkos::RangePolicy<KernelSpace>(0, x_dim * y_dim * z_dim), KOKKOS_LAMBDA(int i){
	for(int i=0; i<x_dim * y_dim * z_dim * patches_per_rank; i++){
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

	for(int timestep=0; timestep<input.timesteps; timestep++)
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
				double *values = reinterpret_cast<double *>(&A[i * x_dim * y_dim * z_dim]);
				HYPRE_StructMatrixSetBoxValues(*HA, low[i].value, high[i].value,
						4, stencil_indices,
						values);
			}

			HYPRE_StructMatrixAssemble(*HA);
		}//if(do_setup)

		//set up RHS

		for(int i=0; i<patches_per_rank; i++)
			HYPRE_StructVectorSetBoxValues(*HB, low[i].value, high[i].value, &B[i * x_dim * y_dim * z_dim]);

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
			std::cout << "HypreSolver not converged in " << num_iterations << " iterations, final residual= " << final_res_norm << " at " <<  __FILE__  << ":" << __LINE__ << "\n";
			fflush(stdout);
			exit(1);
		}

		for(int i=0; i<patches_per_rank; i++)
			HYPRE_StructVectorGetBoxValues(*HX, low[i].value, high[i].value, &X[i * x_dim * y_dim * z_dim]);

		if((timestep % input.output_interval == 0 || timestep == input.timesteps-1)){

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


			if(rank==0 && omp_get_thread_num()==0){
				std::cout.precision(2);
			//exp_type			//size			num_cells   		patch_size   		x_patches    		y_patches   		z_patches
			//timestep			 total_time				solve_only_time			avg_comm_time				num_iterations			final_res_norm
				std::cout <<
				exp_type << "\t" << size << "\t" << num_cells << "\t" << input.patch_size << "\t" << input.xpatches << "\t" << input.ypatches << "\t" << input.zpatches << "\t" <<
				timestep << "\t" << avg_comp_time << "\t" << avg_solve_time << "\t" << avg_comm_time << "\t" << num_iterations << "\t" << final_res_norm <<
				"\n" ;
			}
		}

		if(input.verify==1)
			verifyX(X, x_dim * y_dim * z_dim, my_patches);
	}//for(int timestep=0; timestep<11; timestep++)


	//----------------------------------------------------------------------------------------------------------------------------------------------------------

	//clean up
	HYPRE_StructStencilDestroy(stencil);
	HYPRE_StructGridDestroy(grid);
	hypre_EndTiming (tHypreAll_);

	delete []A;
	delete []X;
	delete []B;

	fflush(stdout);
	HYPRE_Finalize();
}	//end of hypre_solve


int main(int argc1, char **argv1)
{
	if(argc1 != 3){
		printf("Enter arguments id_string and input file name at %s:%d. exiting\n", __FILE__, __LINE__);
		exit(1);
	}
	argc = argc1;
	argv = argv1;
	
	const char * exp_type = argv[1];	//cpu / gpu / gpu-mps / gpu-superpatch. Only to print in output.

//	Kokkos::initialize(argc, argv1);
	{
		int required=MPI_THREAD_MULTIPLE, provided;
		MPI_Init_thread(&argc, &argv, required, &provided);
		
		//Kokkos::OpenMP::partition_master( hypre_solve, omp_get_num_threads(), 1);
		
//		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0, partitions), [&](int i){
//			hypre_solve(partitions, 1);
//		});

		int rank, size;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);

		xmlInput input = parseInput(argv[2], rank);
		int num_of_threads = input.xthreads * input.ythreads * input.zthreads;

		if(input.xthreads>0 && input.ythreads>0 && input.zthreads>0){	//override #threads

#ifdef USE_FUNNELLED_COMM
			omp_set_num_threads(num_of_threads+1);
#else
			omp_set_num_threads(num_of_threads);
#endif

			if(rank ==0) printf("number of threads %d, %d, %d\n", input.xthreads, input.ythreads, input.zthreads);
		}

		if(rank ==0) printf("Number of threads %d\n", num_of_threads);

		assignPatchToEP(input, rank, size, num_of_threads); //assuming EP.

		hypre_set_num_threads(num_of_threads, omp_get_thread_num);
		//cudaProfilerStart();
#pragma omp parallel
		{
			int teamid = omp_get_thread_num();
		  if(teamid<num_of_threads){
			omp_set_num_threads(input.team_size); //second level of parallelism
#pragma omp parallel
			{
				int threadid = omp_get_thread_num();
				int cpu = rank * num_of_threads * input.team_size + teamid * input.team_size + threadid;
//				set_affinity(cpu);
				printf("expected_rank %d team %d thread %d cpu %d\n",rank, teamid, threadid, cpu);

				int cpu_affinity = get_affinity();
				printf("rank %d team %d thread %d cpu %d\n",rank, teamid, threadid, cpu_affinity);
			}


			}
			else{
//				set_affinity(rank * num_of_threads * input.team_size + 16);
				int cpu_affinity = get_affinity();
				int threadid = 0;
				printf("rank %d team %d thread %d cpu %d\n",rank, teamid, threadid, cpu_affinity);
			}

			/*int device_id=-1;
			cudaGetDevice(&device_id);
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, device_id);
			printf("Device id: %d,  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d, cpu_affinity: %d\n", 
					device_id, deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID, cpu_affinity);*/
			
			hypre_solve(exp_type, input);
		}
		//cudaProfilerStop();
		if(rank ==0) printf("Solved successfully\n");
		MPI_Finalize();			


	}
//	Kokkos::finalize();
	return 0;

}
