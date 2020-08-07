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
export MPICH_CXX=nvcc_wrapper
export HYPRE_PATH=/home/damodars/uintah_kokkos_dev/hypre_cpu/build

mpicxx hypre_test.cc -std=c++11 -I$(HYPRE_PATH)/include/ -L$(HYPRE_PATH)/lib -lHYPRE -g -O3

mpicxx 1_hypre_mpi_only.cc -std=c++11 -I$HYPRE_PATH/include/ -L$HYPRE_PATH/lib -lHYPRE -g -O3 -I$KOKKOS_PATH/build/include -L$KOKKOS_PATH/build/lib -lkokkos --expt-extended-lambda
 */

//#define CC 0
//#define USE_PAPI
/*
#ifndef __host__
#define __host__
#define __device__
#endif
*/

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

#include <sched.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include "io.h"
//#include "papi_util.h"


//typedef Kokkos::Serial KernelSpace;
//typedef Kokkos::Cuda KernelSpace;


struct Stencil4{
	double p, n, e, t;
};

typedef cudaView<double> ViewDouble;
typedef cudaView<Stencil4> ViewStencil4;

double tolerance = 1.e-25;

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

int g_rank=-1;

int main(int argc, char **argv)
{
	//papi_init(argc, argv);
/*#pragma omp parallel
{
	int cpu_affinity = get_affinity();
	printf("cpu_affinity: %d\n", cpu_affinity);
#pragma omp barrier
}*/
	//Kokkos::initialize(argc, argv);
	{
		//--------------------------------------------------------- init ----------------------------------------------------------------------------------------------
		if(argc != 8){
			printf("Enter arguments patch size and number of patches. exiting\n");
			exit(1);
		}

		const char * exp_type = argv[1];	//cpu / gpu / gpu-mps / gpu-superpatch. Only to print in output.
		//int patch_dim = atoi(argv[2]);	//patch size
		int x_dim = atoi(argv[2]);
		int y_dim = atoi(argv[3]);
		int z_dim = atoi(argv[4]);
		int x_patches = atoi(argv[5]);	//number of patches in X dimension
		int y_patches = atoi(argv[6]);	//number of patches in Y dimension
		int z_patches = atoi(argv[7]);	//number of patches in Z dimension
		//int device_levels = atoi(argv[8]);	//number of levels to be run on device
		int number_of_patches = x_patches * y_patches * z_patches; 

		MPI_Init(&argc, &argv);
		HYPRE_Init(argc, argv);

		int rank, size;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		g_rank = rank;

		if(number_of_patches % size != 0){
			printf("Ensure total number of patches (patches cube) is divisible number of ranks. exiting\n");
			exit(1);
		}

		//int num_cells = number_of_patches*patch_dim*patch_dim*patch_dim;
		int num_cells = number_of_patches*x_dim*y_dim*z_dim;

		extern double _hypre_comm_time;
		//----------------------------------------------------------------------------------------------------------------------------------------------------------


		//----------------------------------------------- patch set up----------------------------------------------------------------------------------------------

		int patches_per_rank = number_of_patches / size;


		int patch_start = patches_per_rank*rank;	//first patch assigned to rank. next patches_per_rank will be assigned to same rank
		std::vector<IntVector> low(patches_per_rank), high(patches_per_rank);	//store boundaries of every patch

		for(int i=0; i<patches_per_rank; i++)
		{
			//patch id based on local patch number (i) + starting patch assigned to this rank.
			int patch_id = patch_start + i;

			// convert patch_id into 3d patch co-cordinates. Multiply by patch_dim to get starting cell of patch.
			low[i].value[0] = x_dim * (patch_id % x_patches);
			low[i].value[1] = y_dim * ((patch_id / x_patches) % y_patches);
			low[i].value[2] = z_dim * ((patch_id / x_patches) / y_patches);

			// add patch_dim to low to get end cell of patch. subtract 1 as hypre uses inclusive boundaries... as per Uintah code
			high[i].value[0] = low[i].value[0] + x_dim - 1; //including high cell. hypre needs so.. i guess
			high[i].value[1] = low[i].value[1] + y_dim - 1;
			high[i].value[2] = low[i].value[2] + z_dim - 1;
			//printf("%d/%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", rank, size, patch_id, low[i].value[0], low[i].value[1], low[i].value[2], high[i].value[0], high[i].value[1], high[i].value[2]);
		}

		//All patches will have same values. Hence just create 1 patch for A and 1 for B. Pass same values again and again for every patch.
		ViewDouble X(x_dim * y_dim * z_dim), B(x_dim * y_dim * z_dim);

#if CC==0
		ViewStencil4 A(x_dim * y_dim * z_dim);
#endif
#if CC==2
		ViewDouble A(x_dim * y_dim * z_dim);
#endif


		parallel_for(x_dim * y_dim * z_dim, [=] __device__ (int i){
#if CC==0
			A(i).p = 6; A(i).n=-1; A(i).e=-1; A(i).t=-1;
#endif
#if CC==2
			A(i) = 6;
#endif
			B(i) = 1;
			X(i) = 0;
		});

		//----------------------------------------------------------------------------------------------------------------------------------------------------------



		//----------------------------------------------------- hypre -----------------------------------------------------------------------------------------------

		bool do_setup=true;

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

		//papi_c pp(1, papi_event);
		if(rank==0)
std::cout <<
				"exp_type \t size \t num_cells \t x_dim \t y_dim \t z_dim \t x_patches \t y_patches \t z_patches \t" <<
				"timestep \t avg_comp_time \t avg_solve_time \t avg_comm_time \t num_iterations \t final_res_norm \n" ;

		for(int timestep=0; timestep<5; timestep++)
		{
			//pp.start();
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

				// setup the coefficient matrix
#if CC==0
				int stencil_indices[] = {0,1,2,3};
				HYPRE_StructMatrixInitialize(*HA);
				for(int i=0; i<patches_per_rank; i++){
					double *values = reinterpret_cast<double *>(A.data());
					HYPRE_StructMatrixSetBoxValues(*HA, low[i].value, high[i].value, 4, stencil_indices, values);
				}
#endif


#if CC==1
				int stencil_indices[] = {0,1,2,3};
				HYPRE_StructMatrixSetConstantEntries(*HA, 4, stencil_indices);
				HYPRE_StructMatrixInitialize(*HA);
				double values[] = {6, -1, -1, -1};
				HYPRE_StructMatrixSetConstantValues(*HA, 4, stencil_indices, values );
#endif

#if CC==2
				int stencil_indices[] = {1,2,3};

				HYPRE_StructMatrixSetConstantEntries(*HA, 3, stencil_indices);
				HYPRE_StructMatrixInitialize(*HA);

				double values[] = {-1, -1, -1};
				HYPRE_StructMatrixSetConstantValues(*HA, 3, stencil_indices, values );

				int dia_stencil_index[] = {0};
				for(int i=0; i<patches_per_rank; i++){
					double *values = reinterpret_cast<double *>(A.data());
					HYPRE_StructMatrixSetBoxValues(*HA, low[i].value, high[i].value, 1, dia_stencil_index, values);
				}
#endif


				HYPRE_StructMatrixAssemble(*HA);

				// Create the RHS
				if(HB_created)	//just in case if we have to loop over
					HYPRE_StructVectorDestroy(*HB);
				HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, HB);
				HB_created = true;
				HYPRE_StructVectorInitialize(*HB);

			}//if(do_setup)

			//set up RHS

			for(int i=0; i<patches_per_rank; i++)
				HYPRE_StructVectorSetBoxValues(*HB, low[i].value, high[i].value, B.data());

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
				//HYPRE_StructPFMGSetDeviceLevel(*precond_solver, device_levels);

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
				HYPRE_StructVectorGetBoxValues(*HX, low[i].value, high[i].value, X.data());
			
			//pp.stop("");

			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> comp_time = end - start;
			std::chrono::duration<double> solve_time = solve_only_end - solve_only_start;
			
			double t_comp_time, avg_comp_time=0.0, t_solve_time, avg_solve_time=0.0, avg_comm_time=0.0;
			t_comp_time = comp_time.count();
			t_solve_time = solve_time.count();

			MPI_Reduce(&_hypre_comm_time, &avg_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);			
			MPI_Reduce(&t_comp_time, &avg_comp_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&t_solve_time, &avg_solve_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			
			avg_comm_time /= size;
			avg_comp_time /= size;
			avg_solve_time /= size;
			
			std::cout.precision(2);
				//exp_type			//size			num_cells   		x_dim	x_dim	x_dim   		x_patches    		y_patches   		z_patches
				//timestep			 total_time				solve_only_time			avg_comm_time				num_iterations			final_res_norm
			if(rank==0)
				std::cout <<
				exp_type << "\t" << size << "\t" << num_cells << "\t" << x_dim << "\t" << y_dim << "\t" << z_dim << "\t" << x_patches << "\t" << y_patches << "\t" << z_patches << "\t" <<
				timestep << "\t" << avg_comp_time << "\t" << avg_solve_time << "\t" << avg_comm_time << "\t" << num_iterations << "\t" << final_res_norm << 
				"\n" ;

			write_to_file(X, rank);
		}//for(int timestep=0; timestep<11; timestep++)


		//----------------------------------------------------------------------------------------------------------------------------------------------------------

		//clean up
		HYPRE_StructStencilDestroy(stencil);
		HYPRE_StructGridDestroy(grid);
		hypre_EndTiming (tHypreAll_);

		HYPRE_Finalize();

		MPI_Finalize();
	}
	//Kokkos::finalize();

	return 0;

}

