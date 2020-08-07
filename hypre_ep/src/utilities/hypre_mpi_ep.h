#ifndef HYPRE_MPI_EP_H
#define HYPRE_MPI_EP_H

//damodar: added file to contain all MPI Endpoints related code

#ifdef HYPRE_USING_MPI_EP
//#define HYPRE_THREAD_LOCAL_EP __thread
#define HYPRE_THREAD_LOCAL_EP thread_local

#endif

#endif
