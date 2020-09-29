/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Solvers/HypreSolver.h>
#include <CCA/Components/Solvers/MatrixUtil.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Stencil4.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/ConvergenceFailure.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancerPort.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/Timers/Timers.hpp>
#include <iomanip>
#include <omp.h>
#include <CCA/Components/Schedulers/custom_thread.h>

// hypre includes
#include <_hypre_struct_mv.h>
#include <_hypre_utilities.h>
#include <HYPRE_struct_ls.h>
#include <krylov.h>


#ifdef INTEL_ITTNOTIFY_API
#include <ittnotify.h>
__itt_domain* my_itt_domain = __itt_domain_create("MyTraces.MyDomain");
__itt_string_handle* shMyTask = __itt_string_handle_create("HypreSolve");
#endif


//#define PRINTSYSTEM

#ifndef HYPRE_TIMING
#ifndef hypre_ClearTiming
// This isn't in utilities.h for some reason...
#define hypre_ClearTiming()
#endif
#endif

__thread int do_setup=1;
extern thread_local double _hypre_comm_time;
thread_local size_t g_hypre_buff_size{0};
thread_local double *g_hypre_buff;

using namespace std;
using namespace Uintah;

//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

namespace Uintah {

  //==============================================================================
  //
  // Class HypreStencil7
  //
  //==============================================================================
  template<class Types>
  class HypreStencil7 : public RefCounted {
  public:
    HypreStencil7(const Level              * level,
                  const MaterialSet        * matlset,
                  const VarLabel           * A, 
                        Task::WhichDW        which_A_dw,
                  const VarLabel           * x, 
                        bool                 modifies_x,
                  const VarLabel           * b, 
                        Task::WhichDW        which_b_dw,
                  const VarLabel           * guess,
                        Task::WhichDW        which_guess_dw,
                  const HypreSolver2Params * params,
                        bool                 modifies_hypre)
      : level(level), matlset(matlset),
        A_label(A), which_A_dw(which_A_dw),
        X_label(x), 
        B_label(b), which_b_dw(which_b_dw),
        modifies_x(modifies_x),
        guess_label(guess), which_guess_dw(which_guess_dw), params(params),
        modifies_hypre(modifies_hypre)
    {
	  	const char* hypre_num_of_threads_str = std::getenv("HYPRE_THREADS"); //use diff env variable if it conflicts with OMP. but using same will be consistent.
		if(hypre_num_of_threads_str)
		{
		  	char temp_str[16];
			strcpy(temp_str, hypre_num_of_threads_str);
			const char s[2] = ",";
			char *token;
			token = strtok(temp_str, s);	/* get the first token */
			m_hypre_num_of_threads = atoi(token);
			token = strtok(NULL, s);
			m_partition_size =  atoi(token);
		}
		else
		{
			m_hypre_num_of_threads = std::max(1, Uintah::Parallel::getNumPartitions());
			m_partition_size = std::max(1, Uintah::Parallel::getThreadsPerPartition());
		}


      	//printf("m_hypre_num_of_threads : %d at %s %d %s\n",m_hypre_num_of_threads, __FILE__, __LINE__, __FUNCTION__);
    	hypre_solver_label.resize(m_hypre_num_of_threads);
    	d_hypre_solverP_.resize(m_hypre_num_of_threads);
    	for(int i=0; i<m_hypre_num_of_threads ; i++)
    	{
    		std::string label_name = "hypre_solver_label" + std::to_string(i);
    		hypre_solver_label[i] = VarLabel::create(label_name, SoleVariable<hypre_solver_structP>::getTypeDescription());
    		//std::cout << "created label " << label_name << " , " << hypre_solver_label[i]->getName() << "\n";
    	}
                   
      firstPassThrough_ = true;
      movingAverage_    = 0.0;
    }

    //---------------------------------------------------------------------------------------------
    
    virtual ~HypreStencil7() {
      	   	for(int i=0; i<m_hypre_num_of_threads ; i++)
    	    		VarLabel::destroy(hypre_solver_label[i]);
    }

    //---------------------------------------------------------------------------------------------

    double * getBuffer( size_t buff_size )
    {
      if (g_hypre_buff_size < buff_size) {
        g_hypre_buff_size = buff_size;

#if defined(HYPRE_USING_CUDA) || (defined(HYPRE_USING_KOKKOS) && defined(KOKKOS_ENABLE_CUDA))
        if (g_hypre_buff) {
          cudaErrorCheck(cudaFree((void*)g_hypre_buff));
        }

        cudaErrorCheck(cudaMalloc((void**)&g_hypre_buff, buff_size));
#else
        if (g_hypre_buff) {
          free(g_hypre_buff);
        }

        g_hypre_buff = (double *)malloc(buff_size);
#endif
      }

      return g_hypre_buff; // although g_hypre_buff is a member of the class and can be accessed inside task, it can not be directly
                     // accessed inside parallel_for on device (even though its a device pointer, value is not passed by reference)
                     // So return explicitly to a local variable. The local variable gets passed by copy.
    }

    void solve(const ProcessorGroup* pg, 
               const PatchSubset* ps,
               const MaterialSubset* matls,
               DataWarehouse* old_dw, 
               DataWarehouse* new_dw,
               Handle<HypreStencil7<Types> > stencil)
    {
    	int g_nodal_rank=-1, g_nodal_size=-1, g_global_rank=-1;
    	MPI_Comm shmcomm;
    	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
    	MPI_Comm_rank(shmcomm, &g_nodal_rank);
    	MPI_Comm_size(shmcomm, &g_nodal_size);
    	MPI_Comm_free(&shmcomm);

    	const char* hypre_binding_str = std::getenv("HYPRE_BINDING"); //use diff env variable if it conflicts with OMP. but using same will be consistent.
		char temp_str[1024];
		const char s[2] = ",";
		char *token;
		int affinity[m_hypre_num_of_threads*m_partition_size*g_nodal_size];
#ifdef FUNNELED_COMM
		int comm_affinity[g_nodal_size];
#else
		int *comm_affinity = affinity;
#endif

		if(hypre_binding_str)
		{
			strcpy(temp_str, hypre_binding_str);
			token = strtok(temp_str, s);	/* get the first token */
			int i=0;

			while( token != NULL && i < m_hypre_num_of_threads*m_partition_size*g_nodal_size) {
				affinity[i] = atoi(token);
				//printf("%d %d %d \n",my_rank, i, affinity[i]);
				token = strtok(NULL, s);
				i++;
			}
		}
		else
		{
			printf("Error: set environment variable HYPRE_BINDING\n");
			exit(1);
		}

#ifdef FUNNELED_COMM
		//read HYPRE_BINDING_COMM
		const char* hypre_binding_comm_str = std::getenv("HYPRE_BINDING_COMM");
		if(hypre_binding_comm_str)
		{
			strcpy(temp_str, hypre_binding_comm_str);
			token = strtok(temp_str, s);
			int i=0;

			while( token != NULL && i < g_nodal_size) {
				comm_affinity[i] = atoi(token);
				//printf("%d %d %d \n",my_rank, i, affinity[i]);
				token = strtok(NULL, s);
				i++;
			}
		}
		else
		{
			printf("Error: set environment variable HYPRE_BINDING_COMM\n");
			exit(1);
		}
#endif




		int curr_threads = omp_get_max_threads();
    	omp_set_num_threads(2);
    #pragma omp parallel sections num_threads(2)
    	{

    #pragma omp section
    		{
    			int flag;
    			MPI_Is_thread_main( &flag );

    			if(flag)	// main thread should call hypre_solve and worker should init further worker threads. This main thread to act as master thread for funneled comm
    			{
    				//printf("main thread waiting for init\n");
    				wait_for_init(m_hypre_num_of_threads, m_partition_size, comm_affinity, g_nodal_rank);
    				//printf("calling Hypre solve \n");
    				HypreSolve(pg, ps, matls, old_dw, new_dw, stencil);
    				destroy();
    			}
    			else
    			{
    				//printf("worker thread init\n");
    				thread_init(m_hypre_num_of_threads, m_partition_size, affinity, g_nodal_rank);
    			}

    		}

    #pragma omp section
    		{
    			int flag;
    			MPI_Is_thread_main( &flag );

    			if(flag)	// main thread should call hypre_solve and worker should init further worker threads. This main thread to act as master thread for funneled comm
    			{
    				//printf("main thread waiting for init\n");
    				wait_for_init(m_hypre_num_of_threads, m_partition_size, comm_affinity, g_nodal_rank);
    				//printf("calling Hypre solve \n");
    				HypreSolve(pg, ps, matls, old_dw, new_dw, stencil);
    				destroy();
    			}
    			else
    			{
    				//printf("worker thread init\n");
    				thread_init(m_hypre_num_of_threads, m_partition_size, affinity, g_nodal_rank);
    			}

    		}


    	}
    	//printf("omp section end\n");
    	omp_set_num_threads(curr_threads);

    }

    void HypreSolve(const ProcessorGroup* pg,
                   const PatchSubset* ps,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw,
                   Handle<HypreStencil7<Types> >)
    {

    	//create patch subsets equal to number of partitions. in case of scheduler other than kokkos omp, all patches will be in a same patch subset - same as before
      	PatchSubset ** new_patches = new PatchSubset *[m_hypre_num_of_threads];
		for(int part_id = 0; part_id < m_hypre_num_of_threads; part_id++)
		{
			new_patches[part_id] = new PatchSubset();
		}

		int patch_per_part = ps->size() / m_hypre_num_of_threads, remainder = ps->size() % m_hypre_num_of_threads;
		//patch_per_part = (ps->size() % m_hypre_num_of_threads > 0) ? patch_per_part + 1 : patch_per_part;
		int thread_id = 0, count=0;


		for(int p=0;p<ps->size();p++)	//splitting patches among patch subsets
		{
			//printf("%d patch: %d to thread %d\n",pg->myrank(), p, thread_id);
			new_patches[thread_id]->add(ps->get(p));
			count++;

			if(count >= patch_per_part)
			{
				if(thread_id < remainder)	//if number of patches are not perfectly divisible by threads, add 1 extra patch to every thread till remainder patches are not over
				{
					p++;	//dont forget p++
					//printf("%d patch: %d to thread %d\n",pg->myrank(), p, thread_id);
					new_patches[thread_id]->add(ps->get(p));
					count++;

				}
				count = 0;
				thread_id++;
			}
		}



      typedef typename Types::sol_type sol_type;
      if(m_hypre_num_of_threads > ps->size())
      {
    	  std::string a = "******************************* Attention patch count: " + to_string(ps->size()) +  ", hypre threads: " + to_string(m_hypre_num_of_threads)  + "****************************************\n";
    	  std::cout << a;
      }

      tHypreAll_ = hypre_InitializeTiming("Total Hypre time");
      hypre_BeginTiming(tHypreAll_);
      
      tMatVecSetup_ = hypre_InitializeTiming("Matrix + Vector setup");
      tSolveOnly_   = hypre_InitializeTiming("Solve time");
      
      int timestep = params->state->getCurrentTopLevelTimeStep();

      //________________________________________________________
      // Solve frequency
      const int solvFreq = params->solveFrequency;
      // note - the first timestep in hypre is timestep 1
      if (solvFreq == 0 || timestep % solvFreq )
      {
        //new_dw->transferFrom(old_dw,X_label,patches,matls,true);
    	new_dw->transferFrom(old_dw,X_label,ps,matls,true);
        return;
      }

      DataWarehouse *A_dw,*b_dw, *guess_dw;	//damodar: took these out of loop, they caused crash
	  A_dw = new_dw->getOtherDataWarehouse(which_A_dw);
	  b_dw = new_dw->getOtherDataWarehouse(which_b_dw);
	  guess_dw = new_dw->getOtherDataWarehouse(which_guess_dw);

	  int num_threads = std::min(m_hypre_num_of_threads, ps->size());
      hypre_set_num_threads(m_hypre_num_of_threads, m_partition_size, get_custom_team_id);

      double hypre_avg_comm_time=0.0;
      std::atomic<int> hypre_comm_threads_added{0};

#ifdef FUNNELED_COMM
	//for(int t = 0; t < hypre_num_of_threads+1; t++)
	custom_partition_master(0, m_hypre_num_of_threads+1, [&](int t)
#else
	//for(int t = 0; t < hypre_num_of_threads; t++)
	custom_partition_master(0, m_hypre_num_of_threads, [&](int t)
#endif
//#pragma omp parallel for num_threads(num_threads) schedule(static, 1)
    	//for(int part_id = 0; part_id < num_threads; part_id++)	//iterate over number of partitions
    	{
		//printf("calling hypre_init_thread\n");
    	int thread_id = hypre_init_thread();

  		if(thread_id>=0)	//main thread manages comm, others execute hypre code.
  		{
    		int part_id = thread_id;
    		_hypre_comm_time = 0;	//defined in mpistubs.c. Used in strct_communication.c, mpistubs.c and printed in HypreSolve.cc (Uintah) for every timestep

    		const PatchSubset * patches = dynamic_cast<const PatchSubset*>(new_patches[part_id]);

		  //________________________________________________________
		  // Matrix setup frequency - this will destroy and recreate a new Hypre matrix at the specified setupFrequency
		  int suFreq = params->getSetupFrequency();
		  bool mod_setup = true;
		  if (suFreq != 0)
			mod_setup = (timestep % suFreq);
		  do_setup =  ((timestep == 1) || ! mod_setup) ? 1 : 0;

		  // always setup on first pass through
		  if( firstPassThrough_){
			do_setup = 1;
			if(thread_id==0)
				firstPassThrough_ = false;
		  }

		  //________________________________________________________
		  // update coefficient frequency - This will ONLY UPDATE the matrix coefficients without destroying/recreating the Hypre Matrix
		  const int updateCoefFreq = params->getUpdateCoefFrequency();
		  bool modUpdateCoefs = true;
		  if (updateCoefFreq != 0) modUpdateCoefs = (timestep % updateCoefFreq);
		  bool updateCoefs = ( (timestep == 1) || !modUpdateCoefs );
		  //________________________________________________________
		  struct hypre_solver_struct* hypre_solver_s = 0;
		  bool restart = false;

		  if (new_dw->exists(hypre_solver_label[part_id])) {

			new_dw->get(d_hypre_solverP_[part_id],hypre_solver_label[part_id]);
			hypre_solver_s =  d_hypre_solverP_[part_id].get().get_rep();

		  }
		  else if (old_dw->exists(hypre_solver_label[part_id])) {

			old_dw->get(d_hypre_solverP_[part_id],hypre_solver_label[part_id]);

			hypre_solver_s =  d_hypre_solverP_[part_id].get().get_rep();
			new_dw->put(d_hypre_solverP_[part_id], hypre_solver_label[part_id]);

		  }
		  else {
			SoleVariable<hypre_solver_structP> hypre_solverP_;
			hypre_solver_struct* hypre_solver_ = scinew hypre_solver_struct;

			hypre_solver_->solver = scinew HYPRE_StructSolver;
			hypre_solver_->precond_solver = scinew HYPRE_StructSolver;
			hypre_solver_->HA = scinew HYPRE_StructMatrix;
			hypre_solver_->HX = scinew HYPRE_StructVector;
			hypre_solver_->HB = scinew HYPRE_StructVector;

			hypre_solverP_.setData(hypre_solver_);
			hypre_solver_s =  hypre_solverP_.get().get_rep();
			new_dw->put(hypre_solverP_,hypre_solver_label[part_id]);
			restart = true;
		  }
		  ASSERTEQ(sizeof(Stencil7), 7*sizeof(double));


		  Timers::Simple timer;
		  if(pg->myrank() == 0 && thread_id==0) {
			  timer.start();
		  }

#ifdef INTEL_ITTNOTIFY_API
			 //__itt_resume();
			 //__itt_id itt_id;
			 //__itt_frame_begin_v3(my_itt_domain, &itt_id);
			 __itt_task_begin(my_itt_domain, __itt_null, __itt_null, shMyTask);

#endif

		  for(int m = 0;m<matls->size();m++){
			int matl = matls->get(m);

			hypre_BeginTiming(tMatVecSetup_);
			//__________________________________
			// Setup grid
			HYPRE_StructGrid grid;
			if (timestep == 1 || do_setup==1 || restart) {

			  HYPRE_StructGridCreate(pg->getComm(), 3, &grid);

			  for(int p=0;p<patches->size();p++){
				const Patch* patch = patches->get(p);
				Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);

				IntVector l,h1;
				if(params->getSolveOnExtraCells()) {
				  l  = patch->getExtraLowIndex(basis, IntVector(0,0,0));
				  h1 = patch->getExtraHighIndex(basis, IntVector(0,0,0))-IntVector(1,1,1);
				} else {
				  l = patch->getLowIndex(basis);
				  h1 = patch->getHighIndex(basis)-IntVector(1,1,1);
				}
				HYPRE_StructGridSetExtents(grid, l.get_pointer(), h1.get_pointer());
			  }

			  // Periodic boundaries
			  //const Level* level = getLevel(patches);
			  const Level* level = getLevel(ps);
			  IntVector periodic_vector = level->getPeriodicBoundaries();

			  IntVector low, high;
			  level->findCellIndexRange(low, high);
			  IntVector range = high-low;

			  int periodic[3];
			  periodic[0] = periodic_vector.x() * range.x();
			  periodic[1] = periodic_vector.y() * range.y();
			  periodic[2] = periodic_vector.z() * range.z();
			  HYPRE_StructGridSetPeriodic(grid, periodic);
			  // Assemble the grid
			  HYPRE_StructGridAssemble(grid);
			}

			//__________________________________
			// Create the stencil
			HYPRE_StructStencil stencil;
			if ( timestep == 1 || do_setup==1 || restart) {
			  if(params->getSymmetric()){
				HYPRE_StructStencilCreate(3, 4, &stencil);
				int offsets[4][3] = {{0,0,0},
				  {-1,0,0},
				  {0,-1,0},
				  {0,0,-1}};
				for(int i=0;i<4;i++) {
				  HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
				}

			  } else {
				HYPRE_StructStencilCreate(3, 7, &stencil);
				int offsets[7][3] = {{0,0,0},
				  {1,0,0}, {-1,0,0},
				  {0,1,0}, {0,-1,0},
				  {0,0,1}, {0,0,-1}};

				for(int i=0;i<7;i++){
				  HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
				}
			  }
			}

			//__________________________________
			// Create the matrix
			HYPRE_StructMatrix* HA = hypre_solver_s->HA;

			if (timestep == 1 || restart) {
			  HYPRE_StructMatrixCreate(pg->getComm(), grid, stencil, HA);
			  HYPRE_StructMatrixSetSymmetric(*HA, params->getSymmetric());
			  int ghost[] = {1,1,1,1,1,1};
			  HYPRE_StructMatrixSetNumGhost(*HA, ghost);
			  HYPRE_StructMatrixInitialize(*HA);
			} else if (do_setup==1) {
			  HYPRE_StructMatrixDestroy(*HA);
			  HYPRE_StructMatrixCreate(pg->getComm(), grid, stencil, HA);
			  HYPRE_StructMatrixSetSymmetric(*HA, params->getSymmetric());
			  int ghost[] = {1,1,1,1,1,1};
			  HYPRE_StructMatrixSetNumGhost(*HA, ghost);
			  HYPRE_StructMatrixInitialize(*HA);
			}

			// setup the coefficient matrix ONLY on the first timestep, if we are doing a restart, or if we set setupFrequency != 0, or if UpdateCoefFrequency != 0
			if (timestep == 1 || restart || do_setup==1 || updateCoefs) {
			  for(int p=0;p<patches->size();p++) {
				const Patch* patch = patches->get(p);
				printTask( patches, patch, cout_doing, "HypreSolver:solve: Create Matrix" );
				//__________________________________
				// Get A matrix from the DW
				typename Types::symmetric_matrix_type AStencil4;
				typename Types::matrix_type A;
				if (params->getUseStencil4())
				  A_dw->get( AStencil4, A_label, matl, patch, Ghost::None, 0);
				else
				  A_dw->get( A, A_label, matl, patch, Ghost::None, 0);

				Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);

				IntVector l,h;
				if(params->getSolveOnExtraCells()){
				  l = patch->getExtraLowIndex(basis, IntVector(0,0,0));
				  h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
				} else {
				  l = patch->getLowIndex(basis);
				  h = patch->getHighIndex(basis);
				}

	            IntVector hh(h.x()-1, h.y()-1, h.z()-1);
	            int stencil_point = ( params->getSymmetric()) ? 4 : 7;
	            unsigned long Nx = abs(h.x()-l.x()), Ny = abs(h.y()-l.y()), Nz = abs(h.z()-l.z());
	            int start_offset = l.x() + l.y()*Nx + l.z()*Nx*Ny; //ensure starting point is 0 while indexing d_buff
	            size_t buff_size = Nx*Ny*Nz*sizeof(double)*stencil_point;
	            double * d_buff = getBuffer( buff_size );	//allocate / reallocate d_buff;
	            int stencil_indices[] = {0,1,2,3,4,5,6};

				//__________________________________
				// Feed it to Hypre
	            if( params->getSymmetric()){

	              // use stencil4 as coefficient matrix. NOTE: This should be templated
	              // on the stencil type. This workaround is to get things moving
	              // until we convince component developers to move to stencil4. You must
	              // set m_params->setUseStencil4(true) when you setup your linear solver
	              // if you want to use stencil4. You must also provide a matrix of type
	              // stencil4 otherwise this will crash.
	              if ( params->getUseStencil4()) {

					 custom_parallel_for(l.z(), h.z(), [&](int k){
					  for(int j=l.y();j<h.y();j++){
						for(int i=l.x();i<h.x();i++){
						  int id = (i + j*Nx + k*Nx*Ny - start_offset)*stencil_point;
						  d_buff[id + 0] = AStencil4(i, j, k).p;
						  d_buff[id + 1] = AStencil4(i, j, k).w;
						  d_buff[id + 2] = AStencil4(i, j, k).s;
						  d_buff[id + 3] = AStencil4(i, j, k).b;
						}
					  } // y loop
					}, m_partition_size);  // z loop

	              } else { // use stencil7

	            	  custom_parallel_for(l.z(), h.z(), [&](int k){
	            		for(int j=l.y();j<h.y();j++){
	            		  for(int i=l.x();i<h.x();i++){
	            		    int id = (i + j*Nx + k*Nx*Ny - start_offset)*stencil_point;
	            			  d_buff[id + 0] = A(i, j, k).p;
	            			  d_buff[id + 1] = A(i, j, k).w;
	            			  d_buff[id + 2] = A(i, j, k).s;
	            			  d_buff[id + 3] = A(i, j, k).b;

	            			}
	            		} // y loop
	            	  }, m_partition_size);  // z loop

	              }
	            } else { // if( m_params->getSymmetric())

				   custom_parallel_for(l.z(), h.z(), [&](int k){
				    for(int j=l.y();j<h.y();j++){
					  for(int i=l.x();i<h.x();i++){
	                  int id = (i + j*Nx + k*Nx*Ny - start_offset)*stencil_point;
	                  d_buff[id + 0] = A(i, j, k).p;
	                  d_buff[id + 1] = A(i, j, k).e;
	                  d_buff[id + 2] = A(i, j, k).w;
	                  d_buff[id + 3] = A(i, j, k).n;
	                  d_buff[id + 4] = A(i, j, k).s;
	                  d_buff[id + 5] = A(i, j, k).t;
	                  d_buff[id + 6] = A(i, j, k).b;

					  }
				    } // y loop
				  }, m_partition_size);  // z loop

	            }

	            HYPRE_StructMatrixSetBoxValues(*HA,
	                                           l.get_pointer(), hh.get_pointer(),
	                                           stencil_point, stencil_indices,
	                                           d_buff);

			  }
			  if (timestep == 1 || restart || do_setup==1) {
				  HYPRE_StructMatrixAssemble(*HA);
			  }
			}

			//__________________________________
			// Create the RHS
			HYPRE_StructVector* HB = hypre_solver_s->HB;

			if (timestep == 1 || restart) {
			  HYPRE_StructVectorCreate(pg->getComm(), grid, HB);
			  HYPRE_StructVectorInitialize(*HB);
			} else if (do_setup) {
			  HYPRE_StructVectorDestroy(*HB);
			  HYPRE_StructVectorCreate(pg->getComm(), grid, HB);
			  HYPRE_StructVectorInitialize(*HB);
			}

			for(int p=0;p<patches->size();p++){
			  const Patch* patch = patches->get(p);
			  printTask( patches, patch, cout_doing, "HypreSolver:solve: Create RHS" );

			  //__________________________________
			  // Get the B vector from the DW
			  typename Types::const_type B;
			  b_dw->get(B, B_label, matl, patch, Ghost::None, 0);

			  Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);

			  IntVector l,h;
			  if(params->getSolveOnExtraCells()) {
				l = patch->getExtraLowIndex(basis,  IntVector(0,0,0));
				h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
			  } else {
				l = patch->getLowIndex(basis);
				h = patch->getHighIndex(basis);
			  }

			  //__________________________________
			  // Feed it to Hypre

			  unsigned long Nx = abs(h.x()-l.x()), Ny = abs(h.y()-l.y()), Nz = abs(h.z()-l.z());
			  int start_offset = l.x() + l.y()*Nx + l.z()*Nx*Ny; //ensure starting point is 0 while indexing d_buff
			  size_t buff_size = Nx*Ny*Nz*sizeof(double);
			  double * d_buff = getBuffer( buff_size );	//allocate / reallocate d_buff;

				 custom_parallel_for(l.z(), h.z(), [&](int k){
				  for(int j=l.y();j<h.y();j++){
					for(int i=l.x();i<h.x();i++){
				      int id = (i + j*Nx + k*Nx*Ny - start_offset);
				      d_buff[id] = B(i, j, k);
					}
				  } // y loop
				}, m_partition_size);  // z loop

			  IntVector hh(h.x()-1, h.y()-1, h.z()-1);
			  HYPRE_StructVectorSetBoxValues( *HB,
											 l.get_pointer(), hh.get_pointer(),
											 d_buff);

			}
			if (timestep == 1 || restart || do_setup) {
				HYPRE_StructVectorAssemble(*HB);
			}


			//__________________________________
			// Create the solution vector
			HYPRE_StructVector* HX = hypre_solver_s->HX;

			if (timestep == 1 || restart) {
			  HYPRE_StructVectorCreate(pg->getComm(), grid, HX);
			  HYPRE_StructVectorInitialize(*HX);
			} else if (do_setup) {
			  HYPRE_StructVectorDestroy(*HX);
			  HYPRE_StructVectorCreate(pg->getComm(), grid, HX);
			  HYPRE_StructVectorInitialize(*HX);
			}

			for(int p=0;p<patches->size();p++){
			  const Patch* patch = patches->get(p);
			  printTask( patches, patch, cout_doing, "HypreSolver:solve: Create X" );

			  //__________________________________
			  // Get the initial guess
			  if(guess_label){
				typename Types::const_type X;
				guess_dw->get(X, guess_label, matl, patch, Ghost::None, 0);

				Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);

				IntVector l,h;
				if(params->getSolveOnExtraCells()){
				  l = patch->getExtraLowIndex(basis, IntVector(0,0,0));
				  h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
				}
				else{
				  l = patch->getLowIndex(basis);
				  h = patch->getHighIndex(basis);
				}

				//__________________________________
				// Feed it to Hypre

			    IntVector hh(h.x()-1, h.y()-1, h.z()-1);

			    unsigned long Nx = abs(h.x()-l.x()), Ny = abs(h.y()-l.y()), Nz = abs(h.z()-l.z());
			    int start_offset = l.x() + l.y()*Nx + l.z()*Nx*Ny; //ensure starting point is 0 while indexing d_buff
			    size_t buff_size = Nx*Ny*Nz*sizeof(double);
			    double * d_buff = getBuffer( buff_size );	//allocate / reallocate d_buff;

				 custom_parallel_for(l.z(), h.z(), [&](int k){
				  for(int j=l.y();j<h.y();j++){
					for(int i=l.x();i<h.x();i++){
					  int id = (i + j*Nx + k*Nx*Ny - start_offset);
					  d_buff[id] = X(i, j, k);
					}
				  } // y loop
				  }, m_partition_size);  // z loop


			    HYPRE_StructVectorSetBoxValues( *HX,
					  						   l.get_pointer(), hh.get_pointer(),
											   d_buff);

			  }  // initialGuess
			} // patch loop
			if (timestep == 1 || restart || do_setup) {
				HYPRE_StructVectorAssemble(*HX);
			}

			hypre_EndTiming(tMatVecSetup_);
			//__________________________________
			//  Dynamic tolerances  Arches uses this
			double precond_tolerance = 0.0;

		Timers::Simple solve_timer;
		solve_timer.start();

			hypre_BeginTiming(tSolveOnly_);

			int num_iterations;
			double final_res_norm;

			//______________________________________________________________________
			// Solve the system
			if (params->solvertype == "SMG" || params->solvertype == "smg"){

			  HYPRE_StructSolver* solver =  hypre_solver_s->solver;
			  if (timestep == 1 || restart) {
				HYPRE_StructSMGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = smg;
				hypre_solver_s->created_solver=true;
			  } else if (do_setup==1) {
				HYPRE_StructSMGDestroy(*solver);
				HYPRE_StructSMGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = smg;
				hypre_solver_s->created_solver=true;
			  }
			  HYPRE_StructSMGSetMemoryUse   (*solver,  0);
			  HYPRE_StructSMGSetMaxIter     (*solver,  params->maxiterations);
			  HYPRE_StructSMGSetTol         (*solver,  params->tolerance);
			  HYPRE_StructSMGSetRelChange   (*solver,  0);
			  HYPRE_StructSMGSetNumPreRelax (*solver,  params->npre);
			  HYPRE_StructSMGSetNumPostRelax(*solver,  params->npost);
			  HYPRE_StructSMGSetLogging     (*solver,  params->logging);

			  if (do_setup==1){
				  HYPRE_StructSMGSetup          (*solver,  *HA, *HB, *HX);
			  }
			  HYPRE_StructSMGSolve(*solver, *HA, *HB, *HX);
			  HYPRE_StructSMGGetNumIterations(*solver, &num_iterations);
			  HYPRE_StructSMGGetFinalRelativeResidualNorm(*solver, &final_res_norm);

			} else if(params->solvertype == "PFMG" || params->solvertype == "pfmg"){

			  HYPRE_StructSolver* solver =  hypre_solver_s->solver;

			  if (timestep == 1 || restart) {
				HYPRE_StructPFMGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = pfmg;
				hypre_solver_s->created_solver=true;
			  } else if (do_setup==1) {
				HYPRE_StructPFMGDestroy(*solver);
				HYPRE_StructPFMGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = pfmg;
				hypre_solver_s->created_solver=true;
			  }

			  HYPRE_StructPFMGSetMaxIter    (*solver,      params->maxiterations);
			  HYPRE_StructPFMGSetTol        (*solver,      params->tolerance);
			  HYPRE_StructPFMGSetRelChange  (*solver,      0);

			  /* weighted Jacobi = 1; red-black GS = 2 */
			  HYPRE_StructPFMGSetRelaxType   (*solver,  params->relax_type);
			  HYPRE_StructPFMGSetNumPreRelax (*solver,  params->npre);
			  HYPRE_StructPFMGSetNumPostRelax(*solver,  params->npost);
			  HYPRE_StructPFMGSetSkipRelax   (*solver,  params->skip);
			  HYPRE_StructPFMGSetLogging     (*solver,  params->logging);

			  if (do_setup==1){
				HYPRE_StructPFMGSetup          (*solver,  *HA, *HB,  *HX);
			  }

			  HYPRE_StructPFMGSolve(*solver, *HA, *HB, *HX);
			  HYPRE_StructPFMGGetNumIterations(*solver, &num_iterations);
			  HYPRE_StructPFMGGetFinalRelativeResidualNorm(*solver,
														   &final_res_norm);

			} else if(params->solvertype == "SparseMSG" || params->solvertype == "sparsemsg"){

			  HYPRE_StructSolver* solver = hypre_solver_s->solver;
			  if (timestep == 1 || restart) {
				HYPRE_StructSparseMSGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = sparsemsg;
				hypre_solver_s->created_solver=true;
			  } else if (do_setup==1) {
				HYPRE_StructSparseMSGDestroy(*solver);
				HYPRE_StructSparseMSGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = sparsemsg;
				hypre_solver_s->created_solver=true;
			  }

			  HYPRE_StructSparseMSGSetMaxIter  (*solver, params->maxiterations);
			  HYPRE_StructSparseMSGSetJump     (*solver, params->jump);
			  HYPRE_StructSparseMSGSetTol      (*solver, params->tolerance);
			  HYPRE_StructSparseMSGSetRelChange(*solver, 0);

			  /* weighted Jacobi = 1; red-black GS = 2 */
			  HYPRE_StructSparseMSGSetRelaxType(*solver,  params->relax_type);
			  HYPRE_StructSparseMSGSetNumPreRelax(*solver,  params->npre);
			  HYPRE_StructSparseMSGSetNumPostRelax(*solver,  params->npost);
			  HYPRE_StructSparseMSGSetLogging(*solver,  params->logging);
			  if (do_setup==1){
				HYPRE_StructSparseMSGSetup(*solver, *HA, *HB,  *HX);
			  }

			  HYPRE_StructSparseMSGSolve(*solver, *HA, *HB, *HX);
			  HYPRE_StructSparseMSGGetNumIterations(*solver, &num_iterations);
			  HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(*solver,
																&final_res_norm);

			  //__________________________________
			  //
			} else if(params->solvertype == "CG" || params->solvertype == "cg"
					  || params->solvertype == "conjugategradient"
					  || params->solvertype == "PCG"
					  || params->solvertype == "cg"){

			  HYPRE_StructSolver* solver =  hypre_solver_s->solver;

			  if (timestep == 1 || restart) {
				HYPRE_StructPCGCreate(pg->getComm(),solver);
				hypre_solver_s->solver_type = pcg;
				hypre_solver_s->created_solver=true;
			  } else if (do_setup==1) {
				HYPRE_StructPCGDestroy(*solver);
				HYPRE_StructPCGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = pcg;
				hypre_solver_s->created_solver=true;
			  }

			  HYPRE_StructPCGSetMaxIter(*solver, params->maxiterations);
			  HYPRE_StructPCGSetTol(*solver, params->tolerance);
			  HYPRE_StructPCGSetTwoNorm(*solver,  1);
			  HYPRE_StructPCGSetRelChange(*solver,  0);
			  HYPRE_StructPCGSetLogging(*solver,  params->logging);


			  HYPRE_PtrToStructSolverFcn precond;
			  HYPRE_PtrToStructSolverFcn precond_setup;
			  HYPRE_StructSolver* precond_solver = hypre_solver_s->precond_solver;
			  SolverType precond_solver_type;

			  if (timestep == 1 || restart) {
				setupPrecond(pg, precond, precond_setup, *precond_solver,
							 precond_tolerance,precond_solver_type);
				hypre_solver_s->precond_solver_type = precond_solver_type;
				hypre_solver_s->created_precond_solver=true;
				HYPRE_StructPCGSetPrecond(*solver, precond,precond_setup,
										  *precond_solver);

			  } else if (do_setup==1) {

				destroyPrecond(*precond_solver);
				setupPrecond(pg, precond, precond_setup, *precond_solver,
							 precond_tolerance,precond_solver_type);
				hypre_solver_s->precond_solver_type = precond_solver_type;
				hypre_solver_s->created_precond_solver=true;

				HYPRE_StructPCGSetPrecond(*solver, precond,precond_setup,
										  *precond_solver);
			  }


			  if (do_setup==1) {
				HYPRE_StructPCGSetup(*solver, *HA,*HB, *HX);
			  }
			  HYPRE_StructPCGSolve(*solver, *HA, *HB, *HX);
			  HYPRE_StructPCGGetNumIterations(*solver, &num_iterations);
			  HYPRE_StructPCGGetFinalRelativeResidualNorm(*solver,&final_res_norm);

			} else if(params->solvertype == "Hybrid"
					  || params->solvertype == "hybrid"){
			  /*-----------------------------------------------------------
			   * Solve the system using Hybrid
			   *-----------------------------------------------------------*/
			  HYPRE_StructSolver* solver =  hypre_solver_s->solver;

			  if (timestep == 1 || restart) {
				HYPRE_StructHybridCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = hybrid;
				hypre_solver_s->created_solver=true;
			  } else if (do_setup==1) {
				HYPRE_StructHybridDestroy(*solver);
				HYPRE_StructHybridCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = hybrid;
				hypre_solver_s->created_solver=true;
			  }

			  HYPRE_StructHybridSetDSCGMaxIter(*solver, 100);
			  HYPRE_StructHybridSetPCGMaxIter(*solver, params->maxiterations);
			  HYPRE_StructHybridSetTol(*solver, params->tolerance);
			  HYPRE_StructHybridSetConvergenceTol(*solver, 0.90);
			  HYPRE_StructHybridSetTwoNorm(*solver, 1);
			  HYPRE_StructHybridSetRelChange(*solver, 0);
			  HYPRE_StructHybridSetLogging(*solver, params->logging);


			  HYPRE_PtrToStructSolverFcn precond;
			  HYPRE_PtrToStructSolverFcn precond_setup;
			  HYPRE_StructSolver*  precond_solver = hypre_solver_s->precond_solver;
			  SolverType precond_solver_type;

			  if (timestep == 1 || restart) {
				setupPrecond(pg, precond, precond_setup, *precond_solver,
							 precond_tolerance,precond_solver_type);
				hypre_solver_s->precond_solver_type = precond_solver_type;
				hypre_solver_s->created_precond_solver=true;
				HYPRE_StructHybridSetPrecond(*solver,
										   (HYPRE_PtrToStructSolverFcn)precond,
										   (HYPRE_PtrToStructSolverFcn)precond_setup,
										   (HYPRE_StructSolver)precond_solver);

			  } else if (do_setup==1) {
				destroyPrecond(*precond_solver);
				setupPrecond(pg, precond, precond_setup, *precond_solver,
							 precond_tolerance,precond_solver_type);
				hypre_solver_s->precond_solver_type = precond_solver_type;
				hypre_solver_s->created_precond_solver=true;
				HYPRE_StructHybridSetPrecond(*solver,
										   (HYPRE_PtrToStructSolverFcn)precond,
										   (HYPRE_PtrToStructSolverFcn)precond_setup,
										   (HYPRE_StructSolver)precond_solver);
			  }

			  if (do_setup==1) {
				HYPRE_StructHybridSetup(*solver, *HA, *HB, *HX);
			  }

			  HYPRE_StructHybridSolve(*solver, *HA, *HB, *HX);
			  HYPRE_StructHybridGetNumIterations(*solver,&num_iterations);
			  HYPRE_StructHybridGetFinalRelativeResidualNorm(*solver,
															 &final_res_norm);
			  //__________________________________
			  //
			} else if(params->solvertype == "GMRES"
					  || params->solvertype == "gmres"){

			  HYPRE_StructSolver* solver =  hypre_solver_s->solver;

			  if (timestep == 1 || restart) {
				HYPRE_StructGMRESCreate(pg->getComm(),solver);
				hypre_solver_s->solver_type = gmres;
				hypre_solver_s->created_solver=true;
			  } else if (do_setup==1) {
				HYPRE_StructGMRESDestroy(*solver);
				HYPRE_StructGMRESCreate(pg->getComm(),solver);
				hypre_solver_s->solver_type = gmres;
				hypre_solver_s->created_solver=true;
			  }

			  HYPRE_StructGMRESSetMaxIter(*solver,params->maxiterations);
			  HYPRE_StructGMRESSetTol(*solver, params->tolerance);
			  HYPRE_GMRESSetRelChange((HYPRE_Solver)solver,  0);
			  HYPRE_StructGMRESSetLogging  (*solver, params->logging);

			  HYPRE_PtrToStructSolverFcn precond;
			  HYPRE_PtrToStructSolverFcn precond_setup;
			  HYPRE_StructSolver*   precond_solver = hypre_solver_s->precond_solver;
			  SolverType precond_solver_type;

			  if (timestep == 1 || restart) {
				setupPrecond(pg, precond, precond_setup, *precond_solver,
							 precond_tolerance,precond_solver_type);
				hypre_solver_s->precond_solver_type = precond_solver_type;
				hypre_solver_s->created_precond_solver=true;
				HYPRE_StructGMRESSetPrecond(*solver, precond, precond_setup,
											*precond_solver);
			  }  else if (do_setup==1) {
				destroyPrecond(*precond_solver);
				setupPrecond(pg, precond, precond_setup, *precond_solver,
							 precond_tolerance,precond_solver_type);
				hypre_solver_s->precond_solver_type = precond_solver_type;
				hypre_solver_s->created_precond_solver=true;
				HYPRE_StructGMRESSetPrecond(*solver,precond,precond_setup,
											*precond_solver);
			  }

			  if (do_setup==1) {
				HYPRE_StructGMRESSetup(*solver,*HA,*HB,*HX);
			  }

			  HYPRE_StructGMRESSolve(*solver,*HA,*HB,*HX);
			  HYPRE_StructGMRESGetNumIterations(*solver, &num_iterations);
			  HYPRE_StructGMRESGetFinalRelativeResidualNorm(*solver,
															&final_res_norm);

			} else {
			  throw InternalError("Unknown solver type: "+params->solvertype, __FILE__, __LINE__);
			}


	#ifdef PRINTSYSTEM
			//__________________________________
			//   Debugging
			vector<string> fname;
			params->getOutputFileName(fname);
			HYPRE_StructMatrixPrint(fname[0].c_str(), *HA, 0);
			HYPRE_StructVectorPrint(fname[1].c_str(), *HB, 0);
			HYPRE_StructVectorPrint(fname[2].c_str(), *HX, 0);
	#endif

			printTask( patches, patches->get(0), cout_doing, "HypreSolver:solve: testConvergence" );
			//__________________________________
			// Test for convergence
//			if(final_res_norm > params->tolerance || std::isfinite(final_res_norm) == 0){
//			  if( params->getRestartTimestepOnFailure() ){
//				if(pg->myrank() == 0)
//				  cout << "HypreSolver not converged in " << num_iterations
//					   << "iterations, final residual= " << final_res_norm
//					   << ", requesting smaller timestep\n";
//				//new_dw->abortTimestep();
//				//new_dw->restartTimestep();
//			  } else {
//				throw ConvergenceFailure("HypreSolver variable: "+X_label->getName()+", solver: "+params->solvertype+", preconditioner: "+params->precondtype,
//										 num_iterations, final_res_norm,
//										 params->tolerance,__FILE__,__LINE__);
//			  }
//			}

			solve_timer.stop();

			hypre_EndTiming (tSolveOnly_);

			//__________________________________
			// Push the solution into Uintah data structure
			for(int p=0;p<patches->size();p++){
			//custom_parallel_for(0, patches->size(), [&](int p){
			  const Patch* patch = patches->get(p);
			  printTask( patches, patch, cout_doing, "HypreSolver:solve: copy solution" );
			  Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);

			  IntVector l,h;
			  if(params->getSolveOnExtraCells()){
				l = patch->getExtraLowIndex(basis,  IntVector(0,0,0));
				h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
			  }else{
				l = patch->getLowIndex(basis);
				h = patch->getHighIndex(basis);
			  }
			  CellIterator iter(l, h);

			  typename Types::sol_type Xnew;
			  if(modifies_x){
				new_dw->getModifiable(Xnew, X_label, matl, patch);
			  }else{
				new_dw->allocateAndPut(Xnew, X_label, matl, patch);
			  }

	          IntVector hh(h.x()-1, h.y()-1, h.z()-1);
	          unsigned long Nx = abs(h.x()-l.x()), Ny = abs(h.y()-l.y()), Nz = abs(h.z()-l.z());
	          int start_offset = l.x() + l.y()*Nx + l.z()*Nx*Ny; //ensure starting point is 0 while indexing d_buff
	          size_t buff_size = Nx*Ny*Nz*sizeof(double);
	          double * d_buff = getBuffer( buff_size );	//allocate / reallocate d_buff;

	          // Get the solution back from hypre
	          HYPRE_StructVectorGetBoxValues(*HX,
	              l.get_pointer(), hh.get_pointer(),
	              d_buff);

			  custom_parallel_for(l.z(), h.z(), [&](int k){
			    for(int j=l.y();j<h.y();j++){
				  for(int i=l.x();i<h.x();i++){

	              int id = (i + j*Nx + k*Nx*Ny - start_offset);
	              Xnew(i, j, k) = d_buff[id];
				  }
			    } // y loop
			  }, m_partition_size);  // z loop

			 // printf("end %d - %d: %d\n", pg->myrank(), cust_g_team_id, patch->getID() );
			}//);
			//__________________________________
			// clean up
			if ( timestep == 1 || do_setup==1 || restart) {
			  HYPRE_StructStencilDestroy(stencil);
			  HYPRE_StructGridDestroy(grid);
			}
			hypre_EndTiming (tHypreAll_);
			hypre_PrintTiming   ("Hypre Timings:", pg->getComm());
			hypre_FinalizeTiming(tMatVecSetup_);
			hypre_FinalizeTiming(tSolveOnly_);
			hypre_FinalizeTiming(tHypreAll_);
			hypre_ClearTiming();

#ifdef INTEL_ITTNOTIFY_API
		     __itt_task_end(my_itt_domain);
		     //__itt_frame_end_v3(my_itt_domain, &itt_id);
		     //__itt_pause();
#endif

			timer.stop();


			if(pg->myrank() == 0)
			{
#pragma omp critical
				{
				hypre_avg_comm_time += _hypre_comm_time;
				}
				hypre_comm_threads_added++;
			}
			//wait for all threads to finish

			if(pg->myrank() == 0 && thread_id==0) {

				while(hypre_comm_threads_added < m_hypre_num_of_threads);

				hypre_avg_comm_time /= m_hypre_num_of_threads;
			  cout << "Solve of " << X_label->getName()
				   << " on level " << level->getIndex()
				   << " completed in: " << timer().seconds()
				   << " s comm wait time:" << hypre_avg_comm_time << " s (solve only: " << solve_timer().seconds() << " s, ";

			  if (timestep > 2) {
				// alpha = 2/(N+1)
				// averaging window is 10 timesteps.
				double alpha = 2.0/(std::min(timestep - 2, 10) + 1);
				movingAverage_ =
			  alpha*solve_timer().seconds() + (1-alpha)*movingAverage_;

				cout << "mean: " <<  movingAverage_ << " s, ";
			  }

		  cout << num_iterations << " iterations, residual = "
			   << final_res_norm << ")." << std::endl;
			}

			timer.reset( true );
		  }//for(int m = 0;m<matls->size();m++){

#ifdef FUNNELED_COMM
		  if(thread_id==0)
		  {
			  //MPI_Barrier(MPI_COMM_WORLD);
		  	deallocate_funneled_comm();
		  }
#endif
    }//if(!flag)

    });
	//MPI_Barrier(MPI_COMM_WORLD);
  hypre_destroy_thread();
  }
    
    //---------------------------------------------------------------------------------------------
    
    void setupPrecond(const ProcessorGroup* pg,
                      HYPRE_PtrToStructSolverFcn& precond,
                      HYPRE_PtrToStructSolverFcn& pcsetup,
                      HYPRE_StructSolver& precond_solver,
                      const double precond_tolerance,
                      SolverType &precond_solver_type){
                      
      if(params->precondtype == "SMG" || params->precondtype == "smg"){
        /* use symmetric SMG as preconditioner */
        
        precond_solver_type = smg;
        HYPRE_StructSMGCreate         (pg->getComm(),    &precond_solver);  
        HYPRE_StructSMGSetMemoryUse   (precond_solver,   0);                                 
        HYPRE_StructSMGSetMaxIter     (precond_solver,   1);                                 
        HYPRE_StructSMGSetTol         (precond_solver,   precond_tolerance);                               
        HYPRE_StructSMGSetZeroGuess   (precond_solver);                                      
        HYPRE_StructSMGSetNumPreRelax (precond_solver,   params->npre);                      
        HYPRE_StructSMGSetNumPostRelax(precond_solver,   params->npost);                     
        HYPRE_StructSMGSetLogging     (precond_solver,   0);                                 

        
        precond = HYPRE_StructSMGSolve;
        pcsetup = HYPRE_StructSMGSetup;
      //__________________________________
      //
      } else if(params->precondtype == "PFMG" || params->precondtype == "pfmg"){
        /* use symmetric PFMG as preconditioner */
        precond_solver_type = pfmg;
        HYPRE_StructPFMGCreate        (pg->getComm(),    &precond_solver);
        HYPRE_StructPFMGSetMaxIter    (precond_solver,   1);
        HYPRE_StructPFMGSetTol        (precond_solver,   precond_tolerance); 
        HYPRE_StructPFMGSetZeroGuess  (precond_solver);

        /* weighted Jacobi = 1; red-black GS = 2 */
        HYPRE_StructPFMGSetRelaxType   (precond_solver,  params->relax_type);   
        HYPRE_StructPFMGSetNumPreRelax (precond_solver,  params->npre);   
        HYPRE_StructPFMGSetNumPostRelax(precond_solver,  params->npost);  
        HYPRE_StructPFMGSetSkipRelax   (precond_solver,  params->skip);   
        HYPRE_StructPFMGSetLogging     (precond_solver,  0);              

        precond = HYPRE_StructPFMGSolve;
        pcsetup = HYPRE_StructPFMGSetup;
      //__________________________________
      //
      } else if(params->precondtype == "SparseMSG" || params->precondtype == "sparsemsg"){
        precond_solver_type = sparsemsg;
        /* use symmetric SparseMSG as preconditioner */
        HYPRE_StructSparseMSGCreate       (pg->getComm(),   &precond_solver);                  
        HYPRE_StructSparseMSGSetMaxIter   (precond_solver,  1);                                
        HYPRE_StructSparseMSGSetJump      (precond_solver,  params->jump);                     
        HYPRE_StructSparseMSGSetTol       (precond_solver,  precond_tolerance);                              
        HYPRE_StructSparseMSGSetZeroGuess (precond_solver);                                    

        /* weighted Jacobi = 1; red-black GS = 2 */
        HYPRE_StructSparseMSGSetRelaxType (precond_solver, params->relax_type); 
        HYPRE_StructSparseMSGSetNumPreRelax (precond_solver,  params->npre);   
        HYPRE_StructSparseMSGSetNumPostRelax(precond_solver,  params->npost);  
        HYPRE_StructSparseMSGSetLogging     (precond_solver,  0);              
        
        precond = HYPRE_StructSparseMSGSolve;
        pcsetup = HYPRE_StructSparseMSGSetup;
      //__________________________________
      //
      } else if(params->precondtype == "Jacobi" || params->precondtype == "jacobi"){
        /* use two-step Jacobi as preconditioner */
        precond_solver_type = jacobi;
        HYPRE_StructJacobiCreate      (pg->getComm(),    &precond_solver);  
        HYPRE_StructJacobiSetMaxIter  (precond_solver,   2);                       
        HYPRE_StructJacobiSetTol      (precond_solver,   precond_tolerance);                     
        HYPRE_StructJacobiSetZeroGuess(precond_solver);                            
        
        precond = HYPRE_StructJacobiSolve;
        pcsetup = HYPRE_StructJacobiSetup;
      //__________________________________
      //
      } else if(params->precondtype == "Diagonal" || params->precondtype == "diagonal"){
        /* use diagonal scaling as preconditioner */
        precond_solver_type = diagonal;
        precond = HYPRE_StructDiagScale;
        pcsetup = HYPRE_StructDiagScaleSetup;
      } else {
        // This should have been caught in readParameters...
        throw InternalError("Unknown preconditionertype: "+params->precondtype, __FILE__, __LINE__);
      }
    }
    
    //---------------------------------------------------------------------------------------------
    
    void destroyPrecond(HYPRE_StructSolver precond_solver){
      if(params->precondtype        == "SMG"       || params->precondtype == "smg"){
        HYPRE_StructSMGDestroy(precond_solver);
      } else if(params->precondtype == "PFMG"      || params->precondtype == "pfmg"){
        HYPRE_StructPFMGDestroy(precond_solver);
      } else if(params->precondtype == "SparseMSG" || params->precondtype == "sparsemsg"){
        HYPRE_StructSparseMSGDestroy(precond_solver);
      } else if(params->precondtype == "Jacobi"    || params->precondtype == "jacobi"){
        HYPRE_StructJacobiDestroy(precond_solver);
      } else if(params->precondtype == "Diagonal"  || params->precondtype == "diagonal"){
      } else {
        // This should have been caught in readParameters...
        throw InternalError("Unknown preconditionertype in destroyPrecond: "+params->precondtype, __FILE__, __LINE__);
      }
    }

    //---------------------------------------------------------------------------------------------

  private:

    const Level*       level;
    const MaterialSet* matlset;
    const VarLabel*    A_label;
    Task::WhichDW      which_A_dw;
    const VarLabel*    X_label;
    const VarLabel*    B_label;
    Task::WhichDW      which_b_dw;
    bool               modifies_x;
    const VarLabel*    guess_label;
    Task::WhichDW      which_guess_dw;
    const HypreSolver2Params* params;
    bool               modifies_hypre;
    int 			   m_hypre_num_of_threads{0}, m_partition_size{0};
    std::vector<const VarLabel*> hypre_solver_label;
    std::vector<SoleVariable<hypre_solver_structP>> d_hypre_solverP_;
    bool   firstPassThrough_;
    double movingAverage_;
    
    // hypre timers - note that these variables do NOT store timings - rather, each corresponds to
    // a different timer index that is managed by Hypre. To enable the use and reporting of these
    // hypre timings, #define HYPRE_TIMING in HypreSolver.h
    int tHypreAll_;    // Tracks overall time spent in Hypre = matrix/vector setup & assembly + solve time.
    int tSolveOnly_;   // Tracks time taken by hypre to solve the system of equations
    int tMatVecSetup_; // Tracks the time taken by uintah/hypre to allocate and set matrix and vector box vaules
    
  }; // class HypreStencil7
  
  //==============================================================================
  //
  // HypreSolver2 Implementation
  //
  //==============================================================================

  HypreSolver2::HypreSolver2(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
  {
	  	const char* hypre_num_of_threads_str = std::getenv("HYPRE_THREADS"); //use diff env variable if it conflicts with OMP. but using same will be consistent.
		if(hypre_num_of_threads_str)
		{
		  	char temp_str[16];
			strcpy(temp_str, hypre_num_of_threads_str);
			const char s[2] = ",";
			char *token;
			token = strtok(temp_str, s);	/* get the first token */
			m_hypre_num_of_threads = atoi(token);
			token = strtok(NULL, s);
			m_partition_size =  atoi(token);
		}
		else
		{
			m_hypre_num_of_threads = std::max(1, Uintah::Parallel::getNumPartitions());
			m_partition_size = std::max(1, Uintah::Parallel::getThreadsPerPartition());
		}


	  	hypre_solver_label.resize(m_hypre_num_of_threads);
		for(int i=0; i<m_hypre_num_of_threads ; i++)
		{
			std::string label_name = "hypre_solver_label" + std::to_string(i);
			hypre_solver_label[i] = VarLabel::create(label_name,
											  SoleVariable<hypre_solver_structP>::getTypeDescription());
		}

  }

  //---------------------------------------------------------------------------------------------
  
  HypreSolver2::~HypreSolver2()
  {
	  	for(int i=0; i<m_hypre_num_of_threads ; i++)
	  		VarLabel::destroy(hypre_solver_label[i]);
  }

  //---------------------------------------------------------------------------------------------
  
  SolverParameters* HypreSolver2::readParameters(ProblemSpecP& params,
                                                 const string& varname,
                                                 SimulationStateP& state)
  {
    HypreSolver2Params* p = scinew HypreSolver2Params();
    p->state = state;
    bool found=false;
    if(params){
      for( ProblemSpecP param = params->findBlock("Parameters"); param != nullptr; param = param->findNextBlock("Parameters")) {
        string variable;
        if( param->getAttribute("variable", variable) && variable != varname ) {
          continue;
        }
        int sFreq;
        int coefFreq;
        param->getWithDefault ("solver",          p->solvertype,     "smg");      
        param->getWithDefault ("preconditioner",  p->precondtype,    "diagonal"); 
        param->getWithDefault ("tolerance",       p->tolerance,      1.e-10);     
        param->getWithDefault ("maxiterations",   p->maxiterations,  75);         
        param->getWithDefault ("npre",            p->npre,           1);          
        param->getWithDefault ("npost",           p->npost,          1);          
        param->getWithDefault ("skip",            p->skip,           0);          
        param->getWithDefault ("jump",            p->jump,           0);          
        param->getWithDefault ("logging",         p->logging,        0);
        param->getWithDefault ("setupFrequency",  sFreq,             1);
        param->getWithDefault ("updateCoefFrequency",  coefFreq,             1);
        param->getWithDefault ("solveFrequency",  p->solveFrequency, 1);
        param->getWithDefault ("relax_type",      p->relax_type,     1); 
        p->setSetupFrequency(sFreq);
        p->setUpdateCoefFrequency(coefFreq);
        // Options from the HYPRE_ref_manual 2.8
        // npre:   Number of relaxation sweeps before coarse grid correction
        // npost:  Number of relaxation sweeps after coarse grid correction
        // skip:   Skip relaxation on certain grids for isotropic 
        //         problems. This can greatly improve effciency by eliminating
        //         unnecessary relaxations when the underlying problem is isotropic.
        // jump:   not in manual
        //
        // relax_type
        // 0 : Jacobi                                                                   
        // 1 : Weighted Jacobi (default)                                                
        // 2 : Red/Black Gauss-Seidel (symmetric: RB pre-relaxation, BR post-relaxation)
        // 3 : Red/Black Gauss-Seidel (nonsymmetric: RB pre- and post-relaxation)       

        found=true;
      }
    }
    if(!found){
      p->solvertype    = "smg";
      p->precondtype   = "diagonal";
      p->tolerance     = 1.e-10;
      p->maxiterations = 75;
      p->npre    = 1;
      p->npost   = 1;
      p->skip    = 0;
      p->jump    = 0;
      p->logging = 0;
      p->setSetupFrequency(1);
      p->setUpdateCoefFrequency(1);
      p->solveFrequency = 1;
      p->relax_type = 1;
    }
    p->restart   = true;
    return p;
  }

  //---------------------------------------------------------------------------------------------
  
  void HypreSolver2::scheduleInitialize(const LevelP& level,SchedulerP& sched,
                                        const MaterialSet* matls)
  {
    Task* task = scinew Task("initialize_hypre", this,
                             &HypreSolver2::initialize);
  	for(int i=0; i<m_hypre_num_of_threads ; i++)
  		task->computes(hypre_solver_label[i]);

    sched->addTask(task, 
                   sched->getLoadBalancer()->getPerProcessorPatchSet(level), 
                   matls);

  }

  //---------------------------------------------------------------------------------------------
  
  void HypreSolver2::allocateHypreMatrices(DataWarehouse* new_dw)
  {
    //cout << "Doing HypreSolver2::allocateHypreMatrices" << endl;
	  	for(int i=0; i<m_hypre_num_of_threads ; i++)
		{
			SoleVariable<hypre_solver_structP> hypre_solverP_;
			hypre_solver_struct* hypre_solver_ = scinew hypre_solver_struct;

			hypre_solver_->solver = scinew HYPRE_StructSolver;
			hypre_solver_->precond_solver = scinew HYPRE_StructSolver;
			hypre_solver_->HA = scinew HYPRE_StructMatrix;
			hypre_solver_->HX = scinew HYPRE_StructVector;
			hypre_solver_->HB = scinew HYPRE_StructVector;

			hypre_solverP_.setData(hypre_solver_);
			new_dw->put(hypre_solverP_,hypre_solver_label[i]);
		}

  }

  //---------------------------------------------------------------------------------------------

  void
  HypreSolver2::initialize( const ProcessorGroup *,
                            const PatchSubset    * patches,
                            const MaterialSubset * matls,
                                  DataWarehouse  *,
                                  DataWarehouse  * new_dw )
  {
    allocateHypreMatrices( new_dw );
  } 

  //---------------------------------------------------------------------------------------------

  void
  HypreSolver2::scheduleSolve( const LevelP           & level,
                                     SchedulerP       & sched,
                               const MaterialSet      * matls,
                               const VarLabel         * A,    
                                     Task::WhichDW      which_A_dw,  
                               const VarLabel         * x,
                                     bool               modifies_x,
                               const VarLabel         * b,    
                                     Task::WhichDW      which_b_dw,  
                               const VarLabel         * guess,
                                     Task::WhichDW      which_guess_dw,
                               const SolverParameters * params,
                                     bool               modifies_hypre /* = false */ )
  {
    printSchedule(level, cout_doing, "HypreSolver:scheduleSolve");
    
    Task* task;
    // The extra handle arg ensures that the stencil7 object will get freed
    // when the task gets freed.  The downside is that the refcount gets
    // tweaked everytime solve is called.

    TypeDescription::Type domtype = A->typeDescription()->getType();
    ASSERTEQ(domtype, x->typeDescription()->getType());
    ASSERTEQ(domtype, b->typeDescription()->getType());

    const HypreSolver2Params* dparams = dynamic_cast<const HypreSolver2Params*>(params);
    if(!dparams){
      throw InternalError("Wrong type of params passed to hypre solver!", __FILE__, __LINE__);
    }

    //__________________________________
    // bulletproofing
    IntVector periodic = level->getPeriodicBoundaries();
    if(periodic != IntVector(0,0,0)){
      IntVector l,h;
      level->findCellIndexRange( l, h );
      IntVector range = (h - l ) * periodic;
      if( fmodf(range.x(),2) != 0  || fmodf(range.y(),2) != 0 || fmodf(range.z(),2) != 0 ) {
        ostringstream warn;
        warn << "\nINPUT FILE WARNING: hypre solver: \n"
             << "With periodic boundary conditions the resolution of your grid "<<range<<", in each periodic direction, must be as close to a power of 2 as possible (i.e. M x 2^n).\n";
        if (dparams->solvertype == "SMG") {
          throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
        }
        else {
          proc0cout << warn.str();
        }
      }
    }
    
    switch(domtype){
    case TypeDescription::SFCXVariable:
      {
        HypreStencil7<SFCXTypes>* that = scinew HypreStencil7<SFCXTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, dparams,modifies_hypre);
        Handle<HypreStencil7<SFCXTypes> > handle = that;
        task = scinew Task("Hypre:Matrix solve (SFCX)", that, &HypreStencil7<SFCXTypes>::solve, handle);
      }
      break;
    case TypeDescription::SFCYVariable:
      {
        HypreStencil7<SFCYTypes>* that = scinew HypreStencil7<SFCYTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, dparams,modifies_hypre);
        Handle<HypreStencil7<SFCYTypes> > handle = that;
        task = scinew Task("Hypre:Matrix solve (SFCY)", that, &HypreStencil7<SFCYTypes>::solve, handle);
      }
      break;
    case TypeDescription::SFCZVariable:
      {
        HypreStencil7<SFCZTypes>* that = scinew HypreStencil7<SFCZTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, dparams,modifies_hypre);
        Handle<HypreStencil7<SFCZTypes> > handle = that;
        task = scinew Task("Hypre:Matrix solve (SFCZ)", that, &HypreStencil7<SFCZTypes>::solve, handle);
      }
      break;
    case TypeDescription::CCVariable:
      {
        HypreStencil7<CCTypes>* that = scinew HypreStencil7<CCTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, dparams,modifies_hypre);
        Handle<HypreStencil7<CCTypes> > handle = that;
        task = scinew Task("Hypre:Matrix solve (CC)", that, &HypreStencil7<CCTypes>::solve, handle);
      }
      break;
    case TypeDescription::NCVariable:
      {
        HypreStencil7<NCTypes>* that = scinew HypreStencil7<NCTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, dparams,modifies_hypre);
        Handle<HypreStencil7<NCTypes> > handle = that;
        task = scinew Task("Hypre:Matrix solve (NC)", that, &HypreStencil7<NCTypes>::solve, handle);
      }
      break;
    default:
      throw InternalError("Unknown variable type in scheduleSolve", __FILE__, __LINE__);
    }

    task->requires(which_A_dw, A, Ghost::None, 0);
    if(modifies_x)
      task->modifies(x);
    else
      task->computes(x);
    
    if(guess){
      task->requires(which_guess_dw, guess, Ghost::None, 0); 
    }

    task->requires(which_b_dw, b, Ghost::None, 0);
    LoadBalancerPort * lb = sched->getLoadBalancer();
  	if (modifies_hypre) {
    	for(int i=0; i<m_hypre_num_of_threads ; i++)
    		task->requires(Task::NewDW,hypre_solver_label[i]);
    }  else {
    	for(int i=0; i<m_hypre_num_of_threads ; i++)
    	{
    		task->requires(Task::OldDW,hypre_solver_label[i]);
    		task->computes(hypre_solver_label[i]);
    	}
    }
    for(int i=0; i<m_hypre_num_of_threads ; i++)
    	sched->overrideVariableBehavior(hypre_solver_label[i]->getName(),false,false,
                                    	false,true,true);

    task->setType(Task::OncePerProc);
    sched->addTask(task, lb->getPerProcessorPatchSet(level), matls);
  }

  //---------------------------------------------------------------------------------------------
  
  string HypreSolver2::getName(){
    return "hypre";
  }
  
  //---------------------------------------------------------------------------------------------
} // end namespace Uintah
