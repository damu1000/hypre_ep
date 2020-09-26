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
#include <sys/time.h>


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
      	   //	for(int i=0; i<m_hypre_num_of_threads ; i++)
    	   // 		VarLabel::destroy(hypre_solver_label[i]);
    }

    //---------------------------------------------------------------------------------------------

    void solve(const ProcessorGroup* pg, 
               const PatchSubset* ps,
               const MaterialSubset* matls,
               DataWarehouse* old_dw, 
               DataWarehouse* new_dw,
               Handle<HypreStencil7<Types> > stencil)
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

      hypre_set_num_threads(m_hypre_num_of_threads, get_custom_team_id);

      double hypre_avg_comm_time=0.0;
      std::atomic<int> hypre_comm_threads_added{0};

#ifdef FUNNELED_COMM

	custom_partition_master(m_hypre_num_of_threads+1, m_partition_size, [&](int t1)
#else
	//for(int t = 0; t < hypre_num_of_threads; t++)
	custom_partition_master(m_hypre_num_of_threads,  m_partition_size, [&](int t1)
#endif
    	{
		//printf("calling hypre_init_thread\n");
    	int thread_id = hypre_init_thread();
    	//printf("hypre_init_thread id:%d\n",thread_id);

  		if(t1>=0)	//-1th thread manages comm, others execute hypre code.
  		{
  			//printf("if not flag\n");
  			////omp_set_num_threads(m_partition_size);

    		//int thread_id = omp_get_thread_num();
    		int part_id = thread_id;
    		hypre_comm_time = 0;	//defined in mpistubs.c. Used in strct_communication.c, mpistubs.c and printed in HypreSolve.cc (Uintah) for every timestep




    		//
    		//if(omp_get_thread_num()==0)
    		//	printf("number of threads: %d, available threads: %d\n", omp_get_num_threads(), omp_get_max_threads());
    		const PatchSubset * patches = dynamic_cast<const PatchSubset*>(new_patches[part_id]);
	

		  //________________________________________________________
		  // Matrix setup frequency - this will destroy and recreate a new Hypre matrix at the specified setupFrequency
		  int suFreq = params->getSetupFrequency();
		  bool mod_setup = true;
		  if (suFreq != 0)
			mod_setup = (timestep % suFreq);
		  bool do_setup = ((timestep == 1) || ! mod_setup);

		  // always setup on first pass through
		  if( firstPassThrough_){
			do_setup = true;
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
			if (timestep == 1 || do_setup || restart) {

				//std::cout << "damu 1\n";

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
				//std::cout << "damu 2\n";
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
			  //std::cout << "damu 3\n";
			  HYPRE_StructGridSetPeriodic(grid, periodic);
			  //std::cout << "damu 4\n";
			  // Assemble the grid
			  HYPRE_StructGridAssemble(grid);
			}

			//__________________________________
			// Create the stencil
			HYPRE_StructStencil stencil;
			if ( timestep == 1 || do_setup || restart) {
			  if(params->getSymmetric()){
				  //std::cout << "damu 5\n";
				HYPRE_StructStencilCreate(3, 4, &stencil);
				int offsets[4][3] = {{0,0,0},
				  {-1,0,0},
				  {0,-1,0},
				  {0,0,-1}};
				for(int i=0;i<4;i++) {
					//std::cout << "damu 6\n";
				  HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
				}

			  } else {
				  //std::cout << "damu 7\n";
				HYPRE_StructStencilCreate(3, 7, &stencil);
				int offsets[7][3] = {{0,0,0},
				  {1,0,0}, {-1,0,0},
				  {0,1,0}, {0,-1,0},
				  {0,0,1}, {0,0,-1}};

				for(int i=0;i<7;i++){
					//std::cout << "damu 8\n";
				  HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
				}
			  }
			}

			//__________________________________
			// Create the matrix
			HYPRE_StructMatrix* HA = hypre_solver_s->HA;

			if (timestep == 1 || restart) {
				//std::cout << "damu 9\n";
			  HYPRE_StructMatrixCreate(pg->getComm(), grid, stencil, HA);
			  //std::cout << "damu 10\n";
			  HYPRE_StructMatrixSetSymmetric(*HA, params->getSymmetric());
			  int ghost[] = {1,1,1,1,1,1};
			  //std::cout << "damu 11\n";
			  HYPRE_StructMatrixSetNumGhost(*HA, ghost);
			  //std::cout << "damu 12\n";
			  HYPRE_StructMatrixInitialize(*HA);
			} else if (do_setup) {
				//std::cout << "damu 13\n";
			  HYPRE_StructMatrixDestroy(*HA);
			  //std::cout << "damu 14\n";
			  HYPRE_StructMatrixCreate(pg->getComm(), grid, stencil, HA);
			  //std::cout << "damu 15\n";
			  HYPRE_StructMatrixSetSymmetric(*HA, params->getSymmetric());
			  int ghost[] = {1,1,1,1,1,1};
			  //std::cout << "damu 16\n";
			  HYPRE_StructMatrixSetNumGhost(*HA, ghost);
			  //std::cout << "damu 17\n";
			  HYPRE_StructMatrixInitialize(*HA);
			}

			// setup the coefficient matrix ONLY on the first timestep, if we are doing a restart, or if we set setupFrequency != 0, or if UpdateCoefFrequency != 0
			if (timestep == 1 || restart || do_setup || updateCoefs) {
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

				//__________________________________
				// Feed it to Hypre
				if(params->getSymmetric()){

				  double* values = scinew double[(h.x()-l.x())*4*m_partition_size];
				  int stencil_indices[] = {0,1,2,3};


				  // use stencil4 as coefficient matrix. NOTE: This should be templated
				  // on the stencil type. This workaround is to get things moving
				  // until we convince component developers to move to stencil4. You must
				  // set params->setUseStencil4(true) when you setup your linear solver
				  // if you want to use stencil4. You must also provide a matrix of type
				  // stencil4 otherwise this will crash.
				  if (params->getUseStencil4()) {

					//for(int z=l.z();z<h.z();z++){
					 custom_parallel_for(l.z(), h.z(), [&](int z){
					  for(int y=l.y();y<h.y();y++){

						const Stencil4* AA = &AStencil4[IntVector(l.x(), y, z)];
						double* p = &values[(h.x()-l.x())*4*cust_g_thread_id];

						for(int x=l.x();x<h.x();x++){
						  *p++ = AA->p;
						  *p++ = AA->w;
						  *p++ = AA->s;
						  *p++ = AA->b;
						  AA++;
						}
						IntVector ll(l.x(), y, z);
						IntVector hh(h.x()-1, y, z);
						//std::cout << "damu 18\n";
						HYPRE_StructMatrixSetBoxValues(*HA,
													   ll.get_pointer(), hh.get_pointer(),
													   4, stencil_indices, &values[(h.x()-l.x())*4*cust_g_thread_id]);

					  } // y loop
					});  // z loop

				  } else { // use stencil7

					//for(int z=l.z();z<h.z();z++){
					  custom_parallel_for(l.z(), h.z(), [&](int z){
					  for(int y=l.y();y<h.y();y++){

						const Stencil7* AA = &A[IntVector(l.x(), y, z)];
						double* p = &values[(h.x()-l.x())*4*cust_g_thread_id];

						for(int x=l.x();x<h.x();x++){
						  *p++ = AA->p;
						  *p++ = AA->w;
						  *p++ = AA->s;
						  *p++ = AA->b;
						  AA++;
						}
						IntVector ll(l.x(), y, z);
						IntVector hh(h.x()-1, y, z);
						//std::cout << "damu 19\n";
						HYPRE_StructMatrixSetBoxValues(*HA,
													   ll.get_pointer(), hh.get_pointer(),
													   4, stencil_indices, &values[(h.x()-l.x())*4*cust_g_thread_id]);

					  } // y loop
					});  // z loop
				  }
				  delete[] values;
				} else {
				  double* values = scinew double[(h.x()-l.x())*7*m_partition_size];
				  int stencil_indices[] = {0,1,2,3,4,5,6};

				  //for(int z=l.z();z<h.z();z++){
				  custom_parallel_for(l.z(), h.z(), [&](int z){
					for(int y=l.y();y<h.y();y++){

					  const Stencil7* AA = &A[IntVector(l.x(), y, z)];
					  double* p = &values[(h.x()-l.x())*7*cust_g_thread_id];

					  for(int x=l.x();x<h.x();x++){
						*p++ = AA->p;
						*p++ = AA->e;
						*p++ = AA->w;
						*p++ = AA->n;
						*p++ = AA->s;
						*p++ = AA->t;
						*p++ = AA->b;
						AA++;
					  }

					  IntVector ll(l.x(), y, z);
					  IntVector hh(h.x()-1, y, z);
					  //std::cout << "damu 20\n";
					  HYPRE_StructMatrixSetBoxValues(*HA,
													 ll.get_pointer(), hh.get_pointer(),
													 7, stencil_indices,
													 &values[(h.x()-l.x())*7*cust_g_thread_id]);
					}  // y loop
				  }); // z loop
				  delete[] values;
				}
			  }
			  if (timestep == 1 || restart || do_setup) {
				  //std::cout << "damu 21\n";
				  HYPRE_StructMatrixAssemble(*HA);
			  }
			}

			//__________________________________
			// Create the RHS
			HYPRE_StructVector* HB = hypre_solver_s->HB;

			if (timestep == 1 || restart) {
				//std::cout << "damu 22\n";
			  HYPRE_StructVectorCreate(pg->getComm(), grid, HB);
			  //std::cout << "damu \n";
			  HYPRE_StructVectorInitialize(*HB);
			} else if (do_setup) {
				//std::cout << "damu 23\n";
			  HYPRE_StructVectorDestroy(*HB);
			  //std::cout << "damu 24\n";
			  HYPRE_StructVectorCreate(pg->getComm(), grid, HB);
			  //std::cout << "damu 25\n";
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
			  //for(int z=l.z();z<h.z();z++){
			  custom_parallel_for(l.z(), h.z(), [&](int z){
				for(int y=l.y();y<h.y();y++){
				  const double* values = &B[IntVector(l.x(), y, z)];
				  IntVector ll(l.x(), y, z);
				  IntVector hh(h.x()-1, y, z);
				  //std::cout << "damu 26\n";
				  HYPRE_StructVectorSetBoxValues(*HB,
												 ll.get_pointer(), hh.get_pointer(),
												 const_cast<double*>(values));
				}
			  });
			}
			if (timestep == 1 || restart || do_setup) {
				//std::cout << "damu 27\n";
				HYPRE_StructVectorAssemble(*HB);
			}

			//__________________________________
			// Create the solution vector
			HYPRE_StructVector* HX = hypre_solver_s->HX;

			if (timestep == 1 || restart) {
				//std::cout << "damu 28\n";
			  HYPRE_StructVectorCreate(pg->getComm(), grid, HX);
			  //std::cout << "damu 29\n";
			  HYPRE_StructVectorInitialize(*HX);
			} else if (do_setup) {
				//std::cout << "damu 30\n";
			  HYPRE_StructVectorDestroy(*HX);
			  //std::cout << "damu 31\n";
			  HYPRE_StructVectorCreate(pg->getComm(), grid, HX);
			  //std::cout << "damu 32\n";
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
				//for(int z=l.z();z<h.z();z++){
				custom_parallel_for(l.z(), h.z(), [&](int z){
				  for(int y=l.y();y<h.y();y++){
					const double* values = &X[IntVector(l.x(), y, z)];
					IntVector ll(l.x(), y, z);
					IntVector hh(h.x()-1, y, z);
					//std::cout << "damu 33\n";
					HYPRE_StructVectorSetBoxValues(*HX,
												   ll.get_pointer(), hh.get_pointer(),
												   const_cast<double*>(values));
				  }
				});
			  }  // initialGuess
			} // patch loop
			if (timestep == 1 || restart || do_setup) {
				//std::cout << "damu 34\n";
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
				  //std::cout << "damu \n";
				HYPRE_StructSMGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = smg;
				hypre_solver_s->created_solver=true;
			  } else if (do_setup) {
				  //std::cout << "damu 35\n";
				HYPRE_StructSMGDestroy(*solver);
				//std::cout << "damu \n";
				HYPRE_StructSMGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = smg;
				hypre_solver_s->created_solver=true;
			  }
			  //std::cout << "damu 36\n";
			  HYPRE_StructSMGSetMemoryUse   (*solver,  0);
			  //std::cout << "damu 37\n";
			  HYPRE_StructSMGSetMaxIter     (*solver,  params->maxiterations);
			  //std::cout << "damu 38\n";
			  HYPRE_StructSMGSetTol         (*solver,  params->tolerance);
			  //std::cout << "damu 39\n";
			  HYPRE_StructSMGSetRelChange   (*solver,  0);
			  //std::cout << "damu 40\n";
			  HYPRE_StructSMGSetNumPreRelax (*solver,  params->npre);
			  //std::cout << "damu 41\n";
			  HYPRE_StructSMGSetNumPostRelax(*solver,  params->npost);
			  //std::cout << "damu 42\n";
			  HYPRE_StructSMGSetLogging     (*solver,  params->logging);

			  if (do_setup){
				  //std::cout << "damu \n";
				  HYPRE_StructSMGSetup          (*solver,  *HA, *HB, *HX);
			  }
			  //std::cout << "damu 43\n";
			  HYPRE_StructSMGSolve(*solver, *HA, *HB, *HX);
			  //std::cout << "damu 44\n";
			  HYPRE_StructSMGGetNumIterations(*solver, &num_iterations);
			  //std::cout << "damu 45\n";
			  HYPRE_StructSMGGetFinalRelativeResidualNorm(*solver, &final_res_norm);

			} else if(params->solvertype == "PFMG" || params->solvertype == "pfmg"){

			  HYPRE_StructSolver* solver =  hypre_solver_s->solver;

			  if (timestep == 1 || restart) {
				  //std::cout << "damu 46\n";
				HYPRE_StructPFMGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = pfmg;
				hypre_solver_s->created_solver=true;
			  } else if (do_setup) {
				  //std::cout << "damu 47\n";
				HYPRE_StructPFMGDestroy(*solver);
				//std::cout << "damu 47.5\n";
				HYPRE_StructPFMGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = pfmg;
				hypre_solver_s->created_solver=true;
			  }

			  //std::cout << "damu 48\n";
			  HYPRE_StructPFMGSetMaxIter    (*solver,      params->maxiterations);
			  //std::cout << "damu 49\n";
			  HYPRE_StructPFMGSetTol        (*solver,      params->tolerance);
			  //std::cout << "damu 50\n";
			  HYPRE_StructPFMGSetRelChange  (*solver,      0);

			  /* weighted Jacobi = 1; red-black GS = 2 */
			  //std::cout << "damu 51\n";
			  HYPRE_StructPFMGSetRelaxType   (*solver,  params->relax_type);
			  //std::cout << "damu 52\n";
			  HYPRE_StructPFMGSetNumPreRelax (*solver,  params->npre);
			  //std::cout << "damu 53\n";
			  HYPRE_StructPFMGSetNumPostRelax(*solver,  params->npost);
			  //std::cout << "damu 54\n";
			  HYPRE_StructPFMGSetSkipRelax   (*solver,  params->skip);
			  //std::cout << "damu 55\n";
			  HYPRE_StructPFMGSetLogging     (*solver,  params->logging);

			  if (do_setup){
				  //std::cout << "damu 56\n";
				HYPRE_StructPFMGSetup          (*solver,  *HA, *HB,  *HX);
			  }

			  //std::cout << "damu 57\n";
			  HYPRE_StructPFMGSolve(*solver, *HA, *HB, *HX);
			  //std::cout << "damu 58\n";
			  HYPRE_StructPFMGGetNumIterations(*solver, &num_iterations);
			  //std::cout << "damu 59\n";
			  HYPRE_StructPFMGGetFinalRelativeResidualNorm(*solver,
														   &final_res_norm);

			} else if(params->solvertype == "SparseMSG" || params->solvertype == "sparsemsg"){

			  HYPRE_StructSolver* solver = hypre_solver_s->solver;
			  if (timestep == 1 || restart) {
				  //std::cout << "damu 60\n";
				HYPRE_StructSparseMSGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = sparsemsg;
				hypre_solver_s->created_solver=true;
			  } else if (do_setup) {
				  //std::cout << "damu 61\n";
				HYPRE_StructSparseMSGDestroy(*solver);
				//std::cout << "damu 62\n";
				HYPRE_StructSparseMSGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = sparsemsg;
				hypre_solver_s->created_solver=true;
			  }

			  //std::cout << "damu 63\n";
			  HYPRE_StructSparseMSGSetMaxIter  (*solver, params->maxiterations);
			  //std::cout << "damu 64\n";
			  HYPRE_StructSparseMSGSetJump     (*solver, params->jump);
			  //std::cout << "damu \n";
			  HYPRE_StructSparseMSGSetTol      (*solver, params->tolerance);
			  //std::cout << "damu 65\n";
			  HYPRE_StructSparseMSGSetRelChange(*solver, 0);

			  /* weighted Jacobi = 1; red-black GS = 2 */
			  //std::cout << "damu 66\n";
			  HYPRE_StructSparseMSGSetRelaxType(*solver,  params->relax_type);
			  //std::cout << "damu 67\n";
			  HYPRE_StructSparseMSGSetNumPreRelax(*solver,  params->npre);
			  //std::cout << "damu 68\n";
			  HYPRE_StructSparseMSGSetNumPostRelax(*solver,  params->npost);
			  //std::cout << "damu 69\n";
			  HYPRE_StructSparseMSGSetLogging(*solver,  params->logging);
			  if (do_setup){
				  //std::cout << "damu 70\n";
				HYPRE_StructSparseMSGSetup(*solver, *HA, *HB,  *HX);
			  }

			  //std::cout << "damu 70\n";
			  HYPRE_StructSparseMSGSolve(*solver, *HA, *HB, *HX);
			  //std::cout << "damu 71\n";
			  HYPRE_StructSparseMSGGetNumIterations(*solver, &num_iterations);
			  //std::cout << "damu 72\n";
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
				  //std::cout << "damu 73\n";
				HYPRE_StructPCGCreate(pg->getComm(),solver);
				hypre_solver_s->solver_type = pcg;
				hypre_solver_s->created_solver=true;
			  } else if (do_setup) {
				  //std::cout << "damu 74\n";
				HYPRE_StructPCGDestroy(*solver);
				//std::cout << "damu 75\n";
				HYPRE_StructPCGCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = pcg;
				hypre_solver_s->created_solver=true;
			  }

			  //std::cout << "damu 76\n";
			  HYPRE_StructPCGSetMaxIter(*solver, params->maxiterations);
			  //std::cout << "damu 77\n";
			  HYPRE_StructPCGSetTol(*solver, params->tolerance);
			  //std::cout << "damu 78\n";
			  HYPRE_StructPCGSetTwoNorm(*solver,  1);
			  //std::cout << "damu 79\n";
			  HYPRE_StructPCGSetRelChange(*solver,  0);
			  //std::cout << "damu 80\n";
			  HYPRE_StructPCGSetLogging(*solver,  params->logging);


			  HYPRE_PtrToStructSolverFcn precond;
			  HYPRE_PtrToStructSolverFcn precond_setup;
			  HYPRE_StructSolver* precond_solver = hypre_solver_s->precond_solver;
			  SolverType precond_solver_type;

			  if (timestep == 1 || restart) {
				  //std::cout << "damu 81\n";
				setupPrecond(pg, precond, precond_setup, *precond_solver,
							 precond_tolerance,precond_solver_type);
				hypre_solver_s->precond_solver_type = precond_solver_type;
				hypre_solver_s->created_precond_solver=true;
				//std::cout << "damu 82\n";
				HYPRE_StructPCGSetPrecond(*solver, precond,precond_setup,
										  *precond_solver);

			  } else if (do_setup) {

				  //std::cout << "damu 83\n";
				destroyPrecond(*precond_solver);
				//std::cout << "damu 84\n";
				setupPrecond(pg, precond, precond_setup, *precond_solver,
							 precond_tolerance,precond_solver_type);
				hypre_solver_s->precond_solver_type = precond_solver_type;
				hypre_solver_s->created_precond_solver=true;

				//std::cout << "damu 85\n";
				HYPRE_StructPCGSetPrecond(*solver, precond,precond_setup,
										  *precond_solver);
			  }


			  if (do_setup) {
				  //std::cout << "damu 86\n";
			//	  if(omp_get_thread_num()==0) printf("setting up hypre\n");
				HYPRE_StructPCGSetup(*solver, *HA,*HB, *HX);
			//	if(omp_get_thread_num()==0) printf("setting up hypre complete\n");
				//std::cout << "damu 86 complete\n";
			  }

			  //std::cout << "damu 87\n";
			 // if(omp_get_thread_num()==0) printf("calling hypre solve\n");

			  HYPRE_StructPCGSolve(*solver, *HA, *HB, *HX);


			//  if(omp_get_thread_num()==0) printf("calling hypre solve complete\n");
			  //std::cout << "damu 88\n";
			  HYPRE_StructPCGGetNumIterations(*solver, &num_iterations);
			  //std::cout << "damu 89\n";
			  HYPRE_StructPCGGetFinalRelativeResidualNorm(*solver,&final_res_norm);

			} else if(params->solvertype == "Hybrid"
					  || params->solvertype == "hybrid"){
			  /*-----------------------------------------------------------
			   * Solve the system using Hybrid
			   *-----------------------------------------------------------*/
			  HYPRE_StructSolver* solver =  hypre_solver_s->solver;

			  if (timestep == 1 || restart) {
				  //std::cout << "damu 90\n";
				HYPRE_StructHybridCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = hybrid;
				hypre_solver_s->created_solver=true;
			  } else if (do_setup) {
				  //std::cout << "damu 91\n";
				HYPRE_StructHybridDestroy(*solver);
				//std::cout << "damu 92\n";
				HYPRE_StructHybridCreate(pg->getComm(), solver);
				hypre_solver_s->solver_type = hybrid;
				hypre_solver_s->created_solver=true;
			  }

			  //std::cout << "damu 93\n";
			  HYPRE_StructHybridSetDSCGMaxIter(*solver, 100);
			  //std::cout << "damu 94\n";
			  HYPRE_StructHybridSetPCGMaxIter(*solver, params->maxiterations);
			  //std::cout << "damu 95\n";
			  HYPRE_StructHybridSetTol(*solver, params->tolerance);
			  //std::cout << "damu 96\n";
			  HYPRE_StructHybridSetConvergenceTol(*solver, 0.90);
			  //std::cout << "damu 97\n";
			  HYPRE_StructHybridSetTwoNorm(*solver, 1);
			  //std::cout << "damu 98\n";
			  HYPRE_StructHybridSetRelChange(*solver, 0);
			  //std::cout << "damu 99\n";
			  HYPRE_StructHybridSetLogging(*solver, params->logging);


			  HYPRE_PtrToStructSolverFcn precond;
			  HYPRE_PtrToStructSolverFcn precond_setup;
			  HYPRE_StructSolver*  precond_solver = hypre_solver_s->precond_solver;
			  SolverType precond_solver_type;

			  if (timestep == 1 || restart) {
				  //std::cout << "damu 100\n";
				setupPrecond(pg, precond, precond_setup, *precond_solver,
							 precond_tolerance,precond_solver_type);
				hypre_solver_s->precond_solver_type = precond_solver_type;
				hypre_solver_s->created_precond_solver=true;
				//std::cout << "damu 101\n";
				HYPRE_StructHybridSetPrecond(*solver,
										   (HYPRE_PtrToStructSolverFcn)precond,
										   (HYPRE_PtrToStructSolverFcn)precond_setup,
										   (HYPRE_StructSolver)precond_solver);

			  } else if (do_setup) {
				  //std::cout << "damu \n";
				destroyPrecond(*precond_solver);
				//std::cout << "damu 102\n";
				setupPrecond(pg, precond, precond_setup, *precond_solver,
							 precond_tolerance,precond_solver_type);
				hypre_solver_s->precond_solver_type = precond_solver_type;
				hypre_solver_s->created_precond_solver=true;
				//std::cout << "damu 103\n";
				HYPRE_StructHybridSetPrecond(*solver,
										   (HYPRE_PtrToStructSolverFcn)precond,
										   (HYPRE_PtrToStructSolverFcn)precond_setup,
										   (HYPRE_StructSolver)precond_solver);
			  }

			  if (do_setup) {
				  //std::cout << "damu 104\n";
				HYPRE_StructHybridSetup(*solver, *HA, *HB, *HX);
			  }

			  //std::cout << "damu 105\n";
			  HYPRE_StructHybridSolve(*solver, *HA, *HB, *HX);

			  //std::cout << "damu 106\n";
			  HYPRE_StructHybridGetNumIterations(*solver,&num_iterations);
			  //std::cout << "damu 107\n";
			  HYPRE_StructHybridGetFinalRelativeResidualNorm(*solver,
															 &final_res_norm);
			  //__________________________________
			  //
			} else if(params->solvertype == "GMRES"
					  || params->solvertype == "gmres"){

			  HYPRE_StructSolver* solver =  hypre_solver_s->solver;

			  if (timestep == 1 || restart) {
				  //std::cout << "damu 108\n";
				HYPRE_StructGMRESCreate(pg->getComm(),solver);
				hypre_solver_s->solver_type = gmres;
				hypre_solver_s->created_solver=true;
			  } else if (do_setup) {
				  //std::cout << "damu 109\n";
				HYPRE_StructGMRESDestroy(*solver);
				//std::cout << "damu 110\n";
				HYPRE_StructGMRESCreate(pg->getComm(),solver);
				hypre_solver_s->solver_type = gmres;
				hypre_solver_s->created_solver=true;
			  }

			  //std::cout << "damu 111\n";
			  HYPRE_StructGMRESSetMaxIter(*solver,params->maxiterations);
			  //std::cout << "damu 112\n";
			  HYPRE_StructGMRESSetTol(*solver, params->tolerance);
			  //std::cout << "damu 113\n";
			  HYPRE_GMRESSetRelChange((HYPRE_Solver)solver,  0);
			  //std::cout << "damu 114\n";
			  HYPRE_StructGMRESSetLogging  (*solver, params->logging);

			  HYPRE_PtrToStructSolverFcn precond;
			  HYPRE_PtrToStructSolverFcn precond_setup;
			  HYPRE_StructSolver*   precond_solver = hypre_solver_s->precond_solver;
			  SolverType precond_solver_type;

			  if (timestep == 1 || restart) {
				  //std::cout << "damu 115\n";
				setupPrecond(pg, precond, precond_setup, *precond_solver,
							 precond_tolerance,precond_solver_type);
				hypre_solver_s->precond_solver_type = precond_solver_type;
				hypre_solver_s->created_precond_solver=true;
				//std::cout << "damu 116\n";
				HYPRE_StructGMRESSetPrecond(*solver, precond, precond_setup,
											*precond_solver);
			  }  else if (do_setup) {
				  //std::cout << "damu 117\n";
				destroyPrecond(*precond_solver);
				//std::cout << "damu 118\n";
				setupPrecond(pg, precond, precond_setup, *precond_solver,
							 precond_tolerance,precond_solver_type);
				hypre_solver_s->precond_solver_type = precond_solver_type;
				hypre_solver_s->created_precond_solver=true;
				//std::cout << "damu 119\n";
				HYPRE_StructGMRESSetPrecond(*solver,precond,precond_setup,
											*precond_solver);
			  }

			  if (do_setup) {
				  //std::cout << "damu 120\n";
				HYPRE_StructGMRESSetup(*solver,*HA,*HB,*HX);
			  }

			  //std::cout << "damu 121\n";
			  HYPRE_StructGMRESSolve(*solver,*HA,*HB,*HX);

			  //std::cout << "damu 122\n";
			  HYPRE_StructGMRESGetNumIterations(*solver, &num_iterations);
			  //std::cout << "damu 123\n";
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
			if(final_res_norm > params->tolerance || std::isfinite(final_res_norm) == 0){
			  if( params->getRestartTimestepOnFailure() ){
				if(pg->myrank() == 0)
				  cout << "HypreSolver not converged in " << num_iterations
					   << "iterations, final residual= " << final_res_norm
					   << ", requesting smaller timestep\n";
				//new_dw->abortTimestep();
				//new_dw->restartTimestep();
			  } else {
				throw ConvergenceFailure("HypreSolver variable: "+X_label->getName()+", solver: "+params->solvertype+", preconditioner: "+params->precondtype,
										 num_iterations, final_res_norm,
										 params->tolerance,__FILE__,__LINE__);
			  }
			}

			solve_timer.stop();

			//std::cout << "damu 124\n";
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



			 // __asm__ __volatile__ ("" ::: "memory");	//observed occasional crashes in parallel_for here because of invalid value for - h and function iteself.. may be fence will help??

			  //printf("start %d - %d: %d\n", pg->myrank(), cust_g_team_id, patch->getID() );
			  //g_rank_temp = pg->myrank();
			  // Get the solution back from hypre
			  //for(int z=l.z();z<h.z();z++){
			  custom_parallel_for(l.z(), h.z(), [&](int z){
				  //printf("in custom_parallel_for %d - %d %d\n", pg->myrank(), cust_g_team_id, cust_g_thread_id );
				for(int y=l.y();y<h.y();y++){
					double* values;
					//printf(" %d -> %d - %d - Patch : %d: %d %d %d to  %d %d %d\n", pg->myrank(), cust_g_team_id,cust_g_thread_id, patch->getID(), l.x(), l.y(), l.z(), h.x(), h.y(), h.z());
				  values = &Xnew[IntVector(l.x(), y, z)];

				  IntVector ll(l.x(), y, z);
				  IntVector hh(h.x()-1, y, z);

				  //std::cout << "damu 125\n";

				  HYPRE_StructVectorGetBoxValues(*HX,
					  ll.get_pointer(), hh.get_pointer(),
					  values);

				}

			  });


			 // printf("end %d - %d: %d\n", pg->myrank(), cust_g_team_id, patch->getID() );
			}//);
			//__________________________________
			// clean up
			if ( timestep == 1 || do_setup || restart) {
				//std::cout << "damu 126\n";
			  HYPRE_StructStencilDestroy(stencil);
			  //std::cout << "damu 127\n";
			  HYPRE_StructGridDestroy(grid);
			}
			//std::cout << "damu 128\n";
			hypre_EndTiming (tHypreAll_);
			//std::cout << "damu 129\n";
			hypre_PrintTiming   ("Hypre Timings:", pg->getComm());
			//std::cout << "damu 130\n";
			hypre_FinalizeTiming(tMatVecSetup_);
			//std::cout << "damu 131\n";
			hypre_FinalizeTiming(tSolveOnly_);
			//std::cout << "damu 132\n";
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
				hypre_avg_comm_time += hypre_comm_time;
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
			 // printf("deallocating funneled comm\n");
		  	deallocate_funneled_comm();
		  	//printf("deallocating funneled comm - done\n");
		  }
#endif
    }//if(!flag)

    });
	//MPI_Barrier(MPI_COMM_WORLD);
	//printf("hypre_destroy_thread\n");
  hypre_destroy_thread();
  //printf("hypre_destroy_thread - done\n");
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
    		//task->requires(Task::OldDW,hypre_solver_label[i]);
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
