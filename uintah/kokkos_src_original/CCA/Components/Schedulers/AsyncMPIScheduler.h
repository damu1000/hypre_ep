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

#ifndef CCA_COMPONENTS_SCHEDULERS_ASYNCMPISCHEDULER_H
#define CCA_COMPONENTS_SCHEDULERS_ASYNCMPISCHEDULER_H

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include<cstdlib>

#define thread_join 1	//to be used on MIRA as detach does not work. Use join instead.

namespace Uintah {

/**************************************

CLASS
   AsyncMPIScheduler
   

GENERAL INFORMATION
   AsyncMPIScheduler.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Dynamic MPI Scheduler

DESCRIPTION
   Dynamic scheduling with non-deterministic, out-of-order execution of
   tasks at runtime. One MPI rank per CPU core.

  
****************************************/

class AsyncMPIScheduler : public MPIScheduler {

  public:

    AsyncMPIScheduler( const ProcessorGroup* myworld, const Output* oport, AsyncMPIScheduler* parentScheduler = 0 );

    virtual ~AsyncMPIScheduler();

    virtual void problemSetup( const ProblemSpecP& prob_spec, SimulationStateP& state );

    virtual SchedulerP createSubScheduler();

    virtual void execute( int tgnum = 0, int iteration = 0 );
    
    virtual bool useInternalDeps() { return !m_shared_state->isCopyDataTimestep(); }
    
    //damodar
    void createWorkerThreads();

    //int ExecuteOneTask();
  private:

    // eliminate copy, assignment and move
    AsyncMPIScheduler( const AsyncMPIScheduler & )            = delete;
    AsyncMPIScheduler& operator=( const AsyncMPIScheduler & ) = delete;
    AsyncMPIScheduler( AsyncMPIScheduler && )                 = delete;
    AsyncMPIScheduler& operator=( AsyncMPIScheduler && )      = delete;

    QueueAlg m_task_queue_alg { MostMessages };




};

} // End namespace Uintah
   


#endif // End CCA_COMPONENTS_SCHEDULERS_AsyncMPIScheduler_H