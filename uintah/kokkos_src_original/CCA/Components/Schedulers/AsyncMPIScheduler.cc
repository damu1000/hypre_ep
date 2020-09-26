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

#include <CCA/Components/Schedulers/AsyncMPIScheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/TaskGraph.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DOUT.hpp>
#include <mpi.h>
#include <sci_defs/visit_defs.h>
#include <mutex>
#include <stdio.h>

using namespace Uintah;


namespace {

Dout g_dbg(          "DynamicMPI_DBG",         false);
Dout g_queue_length( "DynamicMPI_QueueLength", false);

}


//damodar: new variables used for task queuing and bookkeeping.
namespace{
#ifdef thread_join
std::vector<std::thread> AsyncThreads;
#endif


volatile int g_execution_completed=0; //master thread to set it to 1 in destructor. workers keep on poling g_execution_euque for tasks until g_execution_completed==1

//-------next 3 are used for offloading to slave cores. Monitored by head thread (or main thread) of every thread team
std::vector<DetailedTask *> g_execution_euque;	//task queue: master thread to add

int g_execution_euque_length=0; //editable for master. read only for workers;

std::atomic<int> g_curr_task{0}; //head of the queue. read only for masters. editable for workers.

//-------next 1 used as intimation of completion from slave to master
int* g_completed_task;	//execute initializes to 0. slave set to 1 after completing task. master (in execute) changes to -1 after mpi send


//------- following are used for parallel_for to distribute loop iterations among threads. Allows task and data parallelism both.
//volatile int* g_begin;	//beginning of loop.
std::atomic<int>* g_begin;	//beginning of loop.
volatile int* g_end;	//end of loop
volatile int* g_team_tasks;	//keep count of tasks completed by team. used to avoid repetition of same task. Head thread increments. workers keep track
std::function<void(int)>* g_fun; //function pointer
//volatile int* g_run_loop; //indicator by head node in set parallel_for. Worker threads start execution when set
//std::atomic<int> * g_run_loop; //indicator by head node in set parallel_for. Worker threads start execution when set
//std::mutex *g_mutex;
std::atomic<int> *g_threads_completed;	//synchronization. threads spin unless all in team complete job.
int g_team_size=0;	//threads per team
thread_local int tl_team_id;	//this is thread local team_id. needed in parallel
std::vector<int> g_affinity; //parse THREAD_PINNING environment variable and store affinity. index of g_affinity is thread id. value is core id pass in environment variable
int g_chunk=1;	//dynamic chunk
}



//______________________________________________________________________
//
AsyncMPIScheduler::AsyncMPIScheduler( const ProcessorGroup*      myworld,
                                          const Output*              oport,
                                                AsyncMPIScheduler* parentScheduler )
  : MPIScheduler( myworld, oport, parentScheduler )
{
  m_task_queue_alg =  MostMessages;

}

//______________________________________________________________________
//
AsyncMPIScheduler::~AsyncMPIScheduler()
{
	g_execution_completed=1;
#ifdef thread_join
	for(int i = 0; i < AsyncThreads.size(); i++)
		AsyncThreads[i].join();
#endif
	g_execution_euque.clear();
	//g_phaseTasksDone.clear();
	//delete []g_run_loop;
	delete []g_threads_completed;
	delete []g_begin;
	delete []g_end;
	delete []g_team_tasks;
	delete []g_fun;

}

//______________________________________________________________________
//
void
AsyncMPIScheduler::problemSetup( const ProblemSpecP&     prob_spec,
                                         SimulationStateP& state )
{
  std::string taskQueueAlg = "";

  ProblemSpecP params = prob_spec->findBlock("Scheduler");
  if (params) {
    params->get("taskReadyQueueAlg", taskQueueAlg);
  }
  if (taskQueueAlg == "") {
    taskQueueAlg = "MostMessages";  //default taskReadyQueueAlg
  }

  if (taskQueueAlg == "FCFS") {
    m_task_queue_alg = FCFS;
  }
  else if (taskQueueAlg == "Random") {
    m_task_queue_alg = Random;
  }
  else if (taskQueueAlg == "Stack") {
    m_task_queue_alg = Stack;
  }
  else if (taskQueueAlg == "MostMessages") {
    m_task_queue_alg = MostMessages;
  }
  else if (taskQueueAlg == "LeastMessages") {
    m_task_queue_alg = LeastMessages;
  }
  else if (taskQueueAlg == "PatchOrder") {
    m_task_queue_alg = PatchOrder;
  }
  else if (taskQueueAlg == "PatchOrderRandom") {
    m_task_queue_alg = PatchOrderRandom;
  }
  else {
    throw ProblemSetupException("Unknown task ready queue algorithm", __FILE__, __LINE__);
  }

  SchedulerCommon::problemSetup(prob_spec, state);
  createWorkerThreads();
#ifdef HAVE_VISIT
  static bool initialized = false;

  // Running with VisIt so add in the variables that the user can
  // modify.
  if( m_shared_state->getVisIt() && !initialized ) {
    m_shared_state->d_douts.push_back( &g_dbg );
    m_shared_state->d_douts.push_back( &g_queue_length );

    initialized = true;
  }
#endif
}

//______________________________________________________________________
//
SchedulerP
AsyncMPIScheduler::createSubScheduler()
{
  UintahParallelPort  * lbp      = getPort("load balancer");
  AsyncMPIScheduler * newsched = scinew AsyncMPIScheduler( d_myworld, m_out_port, this );
  newsched->m_shared_state = m_shared_state;
  newsched->attachPort( "load balancer", lbp );
  newsched->m_shared_state = m_shared_state;
  return newsched;
}



void set_affinity(int core)
{
#ifndef __APPLE__
	//disable affinity on OSX since sched_setaffinity() is not available in OSX API
	cpu_set_t mask;
	unsigned int len = sizeof(mask);
	CPU_ZERO(&mask);
	CPU_SET(core, &mask);
	sched_setaffinity(0, len, &mask);
#endif
}



void cpp_parallel_for(int begin, int end, std::function<void(int)> f)
{

	//std::cout << tl_team_id << "\n";
	int team_id = tl_team_id;
	g_fun[team_id] = f;
	g_end[team_id] = end;
	g_begin[team_id] = begin;
	g_threads_completed[team_id] = 0;
	g_team_tasks[team_id]++;
	//g_run_loop[team_id] = 1; //all parameters set. indicate workers to read and run.

	//g_mutex[team_id].unlock();

	//std::cout <<thread_id<< ": " <<  curr_task << " / " << g_execution_euque_length << "\n";
	g_threads_completed[team_id]--;
	while(g_begin[team_id] < g_end[team_id])
	{
		int b = g_begin[team_id];
		int b_next = b + g_chunk;
		if(g_begin[team_id].compare_exchange_strong(b, b_next))
		{
			int e = (b + g_chunk < end) ? b + g_chunk : end;
			//std::string a = std::to_string(team_id) + " - " + std::to_string(0) + " chunk " + std::to_string(b) + " to " + std::to_string(e) + "\n";
			//std::cout << a;
			for(int i=b; i<e; i++)
				g_fun[team_id](i);
		}
	}
	//while(g_mutex[team_id].try_lock());
	//g_run_loop[team_id] = 0;
	//g_begin[team_id] = 1000000;
	//g_end[team_id] = -0xefffffff;

	g_threads_completed[team_id]++;

	while(g_threads_completed[team_id] < 0);
	g_fun[team_id] = NULL;

}


void head_thread(AsyncMPIScheduler * sched, int team_id, int thread_id, int local_rank)
{
	tl_team_id = team_id;
	//while(g_mutex[team_id].try_lock());
	while(g_execution_completed==0)
	{
			int curr_task = g_curr_task ;
			//std::cout <<thread_id<< ": " <<  curr_task << " / " << g_execution_euque_length << "\n";
			while(curr_task < g_execution_euque_length)
			{
				int next_task = curr_task + 1;

				if(g_curr_task.compare_exchange_strong(curr_task, next_task)) //task found and thread successfully acquired task curr_task. g_curr_task incremented to next position
				{
					DetailedTask* task = nullptr;
					while(task == nullptr)	//wait till main thread places task at this location. bulletproofing against rearrangement of instructions by compiler / processor
					{
						task = g_execution_euque[curr_task];
						//std::cout << "------------------------error: invalid task picked from queue : " << curr_task << "/" << g_execution_euque_length << " ------------------------\n";
					}


					if(task)
						sched->runTask(task, 0, 0); // run current task. using dummy value for iteration parameter. As it used by HandleSend.


					g_completed_task[curr_task] = 1;
				}//if(g_curr_task.compare_exchange_strong(curr_task, next_task))

				curr_task = g_curr_task;
			}//while(g_curr_task < g_execution_euque_length)

	}//while(execution_euque_completed==0)
	//g_mutex[team_id].unlock();
}


void worker_thread(int team_id, int thread_id) //thread_id runs from 0 to helper_threads-1. helper_threads th is for main thread
{
	int team_tasks=0, neighbor_team_tasks=0;
	tl_team_id = team_id;
	int neighbor_team = team_id % 2 == 0 ? team_id + 1 : team_id - 1;
	while(g_execution_completed==0)
	{

		//if(/*g_run_loop[team_id]==1 &&*/ g_team_tasks[team_id] > team_tasks) //g_run_loop is signal for workers to start execution. g_team_tasks ensures same task is not repeated
		//if(g_mutex[team_id].try_lock())
		//if(g_threads_completed[team_id]<0)
		//{
		while(!g_fun[team_id]);

			g_threads_completed[team_id]--;
		//	g_mutex[team_id].unlock();
			while(g_begin[team_id] < g_end[team_id])
			{
				int end = g_end[team_id];
				int b = g_begin[team_id];
				int b_next = b + g_chunk;
				if(g_begin[team_id].compare_exchange_strong(b, b_next))
				{
					if(b < end)
					{
						int e = (b + g_chunk < end) ? b + g_chunk : end;
						//std::string a = std::to_string(team_id) + " - " + std::to_string(thread_id) + " chunk " + std::to_string(b) + " to " + std::to_string(e) + "\n";
						//std::cout << a;
						for(int i=b; i<e; i++)
							g_fun[team_id](i);
					}
				}
			}
			//g_run_loop[team_id] = 0;
			//g_begin[team_id] = 1000000;
			g_threads_completed[team_id]++;	//increment atomic variable to indicate main thread about completion. Main thread keeps on waiting till all threads complete. Synchronization point
			team_tasks++;
		//}
		//else if(g_team_tasks[neighbor_team] > neighbor_team_tasks)//i dont have iterations, execute 1 chunk of neighbor
		/*else if(g_threads_completed[neighbor_team] < 0)
		{
			g_threads_completed[neighbor_team]--;
			if(g_begin[neighbor_team] < g_end[neighbor_team])
			{
				int end = g_end[neighbor_team];
				int b = g_begin[neighbor_team];
				int b_next = b + g_chunk;

				if(g_begin[neighbor_team].compare_exchange_strong(b, b_next))
				{
					if(b < end)
					{
						int e = (b + g_chunk < end) ? b + g_chunk : end;
						std::string a = std::to_string(team_id) + " - " + std::to_string(thread_id) + " chunk " + std::to_string(b) + " to " + std::to_string(e) + "\n";
						std::cout << a;
						for(int i=b; i<e; i++)
							g_fun[neighbor_team](i);
					}
				}
			}
			g_threads_completed[neighbor_team]++;
			neighbor_team_tasks++;
		}*/

	}//while(m_keep_running==1)
}

void workerTasks(AsyncMPIScheduler * sched, int team_id, int thread_id, int local_rank)
{
	int core = g_team_size * team_id + thread_id;
	if(g_affinity.size()>0)
	{
		set_affinity(g_affinity[core]);
	}
	else
		set_affinity(core);

	//std::cout << team_id << "-"<< thread_id << "\n";
	if(thread_id==0) //head (or main) thread of each team monitors queue for tasks.
		head_thread(sched, team_id, thread_id, local_rank);
	else	//other threads in team monitor m_run_loop[team_id] for any parallel_for set
		worker_thread( team_id, thread_id);


}


void AsyncMPIScheduler::createWorkerThreads()
{

	int num_teams=0;

	num_teams = Uintah::Parallel::getNumThreads(); //nthreads pass number of teams
	const char* OMP_NUM_THREADS_str = std::getenv("TEAM_SIZE"); //use diff env variable if it conflicts with OMP. but using same will be consistent.
	if(OMP_NUM_THREADS_str)
	{
		g_team_size = atoi(OMP_NUM_THREADS_str);
		std::cout << "Creating " << num_teams << " teams of " << g_team_size << " threads per team.\n";
	}
	else
	{
		std::cout << "error: set TEAM_SIZE\n";
		exit(1);
	}

	char* THREAD_PINNING_str = std::getenv("THREAD_PINNING");
	if(THREAD_PINNING_str)
	{
		char * pch;
		pch = strtok (THREAD_PINNING_str,",");
		while (pch != NULL)
		{
			g_affinity.push_back(atoi(pch));
			pch = strtok (NULL, ",");
		}
	}

	//init team variables
	//g_begin = new  volatile int[num_teams] ;
	g_begin = new std::atomic<int>[num_teams];
	g_end = new  volatile int[num_teams] ;
	g_team_tasks = new  volatile int[num_teams] ;
	g_fun = new std::function<void(int)>[num_teams] ;
	g_threads_completed = new std::atomic<int>[num_teams];
	//g_run_loop = new volatile int[num_teams];
	//g_run_loop = new std::atomic<int>[num_teams];
	//g_mutex = new std::mutex[num_teams];

	for(int i = 0; i<num_teams; i++)
	{
		g_begin[i]=0;
		g_end[i]=0;
		g_team_tasks[i]=0;
		g_threads_completed[i]=0;
		g_fun[i] = NULL;
		//g_run_loop[i]=0;
	}



#ifdef thread_join
	AsyncThreads.resize(num_teams*g_team_size);
#endif


	int local_rank = 0;
#ifdef mpi_aware_thread_pinning //pin master thread to 0th core.
	local_rank = getLocalRank();
	int core = num_threads * local_rank;
	set_affinity(core);
#endif

	int k=0;
	for(int i=0; i<num_teams; i++)
	{
		for(int j=0; j<g_team_size; j++)
		{
#ifdef thread_join
			AsyncThreads[k++] = std::thread(workerTasks, this, i, j, local_rank);
#else
			std::thread t(workerTasks, this, i, j, local_rank);
			t.detach();
#endif
		}
	}//for(int i=0; i<num_threads; i++)
}



//______________________________________________________________________
//
void
AsyncMPIScheduler::execute( int tgnum     /*=0*/,
                              int iteration /*=0*/ )
{
  if (m_shared_state->isCopyDataTimestep()) {
    MPIScheduler::execute(tgnum, iteration);
    return;
  }

  // track total scheduler execution time across timesteps
  m_exec_timer.reset(true);

//damodar: reset all queue related variables.
  g_execution_euque.clear();
  g_execution_completed=0;
  g_execution_euque_length=0;
  g_curr_task=0;


  RuntimeStats::initialize_timestep(m_task_graphs);

  ASSERTRANGE(tgnum, 0, static_cast<int>(m_task_graphs.size()));
  TaskGraph* tg = m_task_graphs[tgnum];
  tg->setIteration(iteration);
  m_current_task_graph = tgnum;

  // multi TG model - each graph needs have its dwmap reset here (even with the same tgnum)
  if (static_cast<int>(m_task_graphs.size()) > 1) {
    tg->remapTaskDWs(m_dwmap);
  }

  DetailedTasks* dts = tg->getDetailedTasks();

  if(!dts) {
    if (d_myworld->myrank() == 0) {
      DOUT(true, "AsyncMPIScheduler skipping execute, no tasks");
    }
    return;
  }
  
  int ntasks = dts->numLocalTasks();
  g_execution_euque.resize(ntasks);
  dts->initializeScrubs(m_dws, m_dwmap);
  dts->initTimestep();

  for (int i = 0; i < ntasks; i++) {
    dts->localTask(i)->resetDependencyCounts();
  }

  int me = d_myworld->myrank();

  // This only happens if "-emit_taskgraphs" is passed to sus
  makeTaskGraphDoc(dts, me);

  mpi_info_.reset( 0 );

  if( m_reloc_new_pos_label && m_dws[m_dwmap[Task::OldDW]] != nullptr ) {
    m_dws[m_dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, getLoadBalancer(), m_reloc_new_pos_label, iteration);
  }


  int currphase = 0;
  std::map<int, int> phaseTasks;
  std::map<int, int>phaseTasksDone;
  std::map<int,  DetailedTask *> phaseSyncTask;
  int numTasksDone = 0;

  dts->setTaskPriorityAlg(m_task_queue_alg);

  for (int i = 0; i < ntasks; i++) {
    phaseTasks[dts->localTask(i)->getTask()->m_phase]++;
  }
  

  static int totaltasks;
  //std::set<DetailedTask*> pending_tasks;	//damodar: commented. not used anywhere

  //int g_numTasksDone = 0;
  bool abort       = false;
  int  abort_point = 987654;
  int i            = 0;

  g_completed_task = new int[ntasks];
  for (int i = 0; i < ntasks; i++) {
	  g_completed_task[i] = 0;		//mark all tasks not completed.
  }
  int completed_task_start=0;

#if 0
// hook to post all the messages up front
	if (!m_shared_state->isCopyDataTimestep()) {
	  // post the receives in advance
	  for (int i = 0; i < ntasks; i++) {
		initiateTask( dts->localTask(i), abort, abort_point, iteration );
	  }
	}
#endif


  while( numTasksDone < ntasks ) {

    i++;

    DetailedTask * task = nullptr;

    // if we have an internally-ready task, initiate its recvs
    while(dts->numInternalReadyTasks() > 0) { 
      DetailedTask * task = dts->getNextInternalReadyTask();

      if ((task->getTask()->getType() == Task::Reduction) || (task->getTask()->usesMPI())) {  //save the reduction task for later
        phaseSyncTask[task->getTask()->m_phase] = task;
      } else {
        initiateTask(task, abort, abort_point, iteration); //damodar: using Alan's hack. initiate all tasks upfront
        task->markInitiated();
        task->checkExternalDepCount();
        // if MPI has completed, it will run on the next iteration
        //pending_tasks.insert(task);		//damodar: commented. not used anywhere
      }
    }

    //if (dts->numExternalReadyTasks() > 0) {
    while (dts->numExternalReadyTasks() > 0) {
      // run a task that has its communication complete
      // tasks get in this queue automatically when their receive count hits 0
      //   in DependencyBatch::received, which is called when a message is delivered.
     
      DetailedTask * task = dts->getNextExternalReadyTask();

      //pending_tasks.erase(pending_tasks.find(task));		//damodar: commented. not used anywhere
      ASSERTEQ(task->getExternalDepCount(), 0);

      //damodar: add task to queue for worker threads instead of executing. probably can be moved out of loop. just 1 copy of m_dws should serve.
      //runTask(task, iteration);

      //--------------------------------------------equivalent of run() / offload-------------------------------------------------------
      //g_execution_euque.push_back(task); //add task in queue here.
      g_execution_euque[g_execution_euque_length] = task;
      g_execution_euque_length++;
      //--------------------------------------------equivalent of finish() / offload-------------------------------------------------------

      //numTasksDone++;		//damodar. moved to for loop checking completion.
      //phaseTasksDone[task->getTask()->m_phase]++;
    }


    //damodar: check for any completed tasks and post MPI sends and mark task as done.
    //--------------------------------------------equivalent of test()-------------------------------------------------------
    int advance_start=1;	//using combination of advance_start and completed_task_start to minimize rechecking of completed tasks.
    for(int i = completed_task_start; i<g_execution_euque_length; i++)
    {
    	int completed = g_completed_task[i];
    	if(completed==1)
    	{
    		DetailedTask* dts = g_execution_euque[i];
    		if(dts)
    		{
				HandleSend(dts, iteration);
				numTasksDone++;
				phaseTasksDone[dts->getTask()->m_phase]++;
				g_completed_task[i]=-1;
				//if(advance_start)
				//     completed_task_start = i+1;
				completed_task_start = advance_start*(i+1); //update completed_task_start if ALL tasks b4 current tasks are 1 or -1.
    		}
    		else
    			std::cout << "-----------------------------Canceling handleSend. Invalid task-----------------------------\n";
    	}
    	//if(g_completed_task[i]==0)
    	//	advance_start=false;
    	advance_start = advance_start*completed*completed;
    }
    //--------------------------------------------finish equivalent of test()-------------------------------------------------------


    if ((phaseSyncTask.find(currphase) != phaseSyncTask.end()) && (phaseTasksDone[currphase] == phaseTasks[currphase] - 1)) {  //if it is time to run the reduction task

      DetailedTask *reducetask = phaseSyncTask[currphase];
      if (reducetask->getTask()->getType() == Task::Reduction) {
        initiateReduction(reducetask);
      }
      else {  // Task::OncePerProc task
        ASSERT(reducetask->getTask()->usesMPI());
        initiateTask(reducetask, abort, abort_point, iteration);
        reducetask->markInitiated();
        ASSERT(reducetask->getExternalDepCount() == 0);
        runTask(reducetask, iteration);
      }
      ASSERT(reducetask->getTask()->m_phase == currphase);

      numTasksDone++;
      phaseTasksDone[reducetask->getTask()->m_phase]++;
    }

    if (numTasksDone < ntasks) {
      if (phaseTasks[currphase] == phaseTasksDone[currphase]) {
        currphase++;
      }
      else if (dts->numExternalReadyTasks() > 0 || dts->numInternalReadyTasks() > 0
               || (phaseSyncTask.find(currphase) != phaseSyncTask.end() && phaseTasksDone[currphase] == phaseTasks[currphase] - 1))  // if there is work to do
          {
        processMPIRecvs(TEST);  // receive what is ready and do not block
      }
      else {
    	processMPIRecvs(WAIT_ONCE);

      }
    }

    if (!abort && m_dws[m_dws.size() - 1] && m_dws[m_dws.size() - 1]->timestepAborted()) {
      // TODO - abort might not work with external queue...
      abort = true;
      abort_point = task->getTask()->getSortedOrder();
    }
  } // end while( g_numTasksDone < ntasks )


  if (g_queue_length) {
    float lengthsum = 0;
    totaltasks += ntasks;
    float queuelength = lengthsum / totaltasks;
    float allqueuelength = 0;
    Uintah::MPI::Reduce(&queuelength, &allqueuelength, 1, MPI_FLOAT, MPI_SUM, 0, d_myworld->getComm());
  }

  //---------------------------------------------------------------------------
  // New way of managing single MPI requests - avoids MPI_Waitsome & MPI_Donesome - APH 07/20/16
  // ---------------------------------------------------------------------------
  // wait on all pending requests
  auto ready_request = [](CommRequest const& r)->bool { return r.wait(); };
  while ( m_sends.size() != 0u ) {
    CommRequestPool::iterator comm_sends_iter;
    if ( (comm_sends_iter = m_sends.find_any(ready_request)) ) {
      m_sends.erase(comm_sends_iter);
    } else {
      // TODO - make this a sleep? APH 07/20/16
    }
  }
  //---------------------------------------------------------------------------

  ASSERT(m_sends.size() == 0u);
  ASSERT(m_recvs.size() == 0u);


  // Copy the restart flag to all processors
  if (m_restartable && tgnum == static_cast<int>(m_task_graphs.size()) - 1) {
    int myrestart = m_dws[m_dws.size() - 1]->timestepRestarted();
    int netrestart;

    Uintah::MPI::Allreduce(&myrestart, &netrestart, 1, MPI_INT, MPI_LOR, d_myworld->getComm());

    if (netrestart) {
      m_dws[m_dws.size() - 1]->restartTimestep();
      if (m_dws[0]) {
        m_dws[0]->setRestarted();
      }
    }
  }

  //damodar: wait till all tasks are over. Then finalize timestep.
  while(g_curr_task < g_execution_euque_length)
  {
	  //std::cout << "master waiting: " << g_curr_task << " / " << g_execution_euque_length << "\n";
  }



  finalizeTimestep();
  
  m_exec_timer.stop();

  // compute the net timings
  if (m_shared_state != nullptr) {
    computeNetRunTimeStats(m_shared_state->d_runTimeStats);
  }

  // only do on top-level scheduler
  if ( m_parent_scheduler == nullptr ) {
    outputTimingStats( "AsyncMPIScheduler" );
  }

  RuntimeStats::report(d_myworld->getComm());

  delete []g_completed_task;
} // end execute()
