/*BHEADER**********************************************************************
* Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
* Produced at the Lawrence Livermore National Laboratory.
* This file is part of HYPRE.  See file COPYRIGHT for details.
*
* HYPRE is free software; you can redistribute it and/or modify it under the
* terms of the GNU Lesser General Public License (as published by the Free
* Software Foundation) version 2.1 dated February 1999.
*
* $Revision$
***********************************************************************EHEADER*/

#include "_hypre_utilities.h"
#include <sys/time.h>

#include<math.h>
//#include <threads.h>
#include "hypre_mpi_ep_mt_q.h"

 extern struct MPIR_Request * recvq_unexpected_head ;
/******************************************************************************
 * This routine is the same in both the sequential and normal cases
 *
 * The 'comm' argument for MPI_Comm_f2c is MPI_Fint, which is always the size of
 * a Fortran integer and hence usually the size of hypre_int.
 *****************************************************************************/
//#define debug_mpi_calls 1
 __thread MPI_Comm comm = MPI_COMM_WORLD;


#ifdef HYPRE_USING_MPI_EP

#include <vector>
#include <unordered_map>
#include <functional>
#include <iostream>

__thread int g_rank=0, g_real_rank=0;
__thread int g_thread_id=0;

//I know its a bad place. change later
#if defined(HYPRE_USING_CUDA) 
cudaStream_t *g_streams;
#endif 


const char* getOpName(MPI_Op op)
{
	/*switch(op)
	{*/
	if(op == 	 MPI_MAX	 ) return(" MPI_MAX");
	if(op == 	 MPI_MIN	 ) return(" MPI_MIN");
	if(op == 	 MPI_SUM	 ) return(" MPI_SUM");
	if(op == 	 MPI_PROD	 ) return(" MPI_PROD");
	if(op == 	 MPI_LAND	 ) return(" MPI_LAND");
	if(op == 	 MPI_BAND	 ) return(" MPI_BAND");
	if(op == 	 MPI_LOR	 ) return(" MPI_LOR");
	if(op == 	 MPI_BOR	 ) return(" MPI_BOR");
	if(op == 	 MPI_LXOR	 ) return(" MPI_LXOR");
	if(op == 	 MPI_BXOR	 ) return(" MPI_BXOR");
	if(op == 	 MPI_MINLOC	 ) return(" MPI_MINLOC");
	if(op == 	 MPI_MAXLOC	 ) return(" MPI_MAXLOC");
	if(op == 	 MPI_REPLACE ) return(" MPI_REPLACE");
	//case 	 MPI_NO_OP	 : return(" MPI_NO_OP");
	/*case 	 MPI_CHAR: return(" MPI_CHAR");
	case MPI_WCHAR: return(" MPI_WCHAR");
	case MPI_SHORT: return(" MPI_SHORT");
	case MPI_INT: return(" MPI_INT");
	case MPI_LONG: return(" MPI_LONG");
	case MPI_LONG_LONG	: return(" MPI_LONG_LONG	");
	case MPI_SIGNED_CHAR: return(" MPI_SIGNED_CHAR");
	case MPI_UNSIGNED_CHAR: return(" MPI_UNSIGNED_CHAR");
	case MPI_UNSIGNED_SHORT: return(" MPI_UNSIGNED_SHORT");
	case MPI_UNSIGNED_LONG: return(" MPI_UNSIGNED_LONG");
	case MPI_UNSIGNED: return(" MPI_UNSIGNED");
	case MPI_FLOAT: return(" MPI_FLOAT");
	case MPI_DOUBLE: return(" MPI_DOUBLE");
	case MPI_LONG_DOUBLE: return(" MPI_LONG_DOUBLE");
	case MPI_C_FLOAT_COMPLEX: return(" MPI_C_FLOAT_COMPLEX");
	case MPI_C_DOUBLE_COMPLEX: return(" MPI_C_DOUBLE_COMPLEX");
	case MPI_C_BOOL: return(" MPI_C_BOOL");
	case MPI_LOGICAL: return(" MPI_LOGICAL");
	case MPI_C_LONG_DOUBLE_COMPLEX: return(" MPI_C_LONG_DOUBLE_COMPLEX");
	case MPI_INT8_T: return(" MPI_INT8_T");
	case MPI_INT16_T: return(" MPI_INT16_T");
	case MPI_INT32_T: return(" MPI_INT32_T");
	case MPI_INT64_T	: return(" MPI_INT64_T	");
	case MPI_UINT8_T: return(" MPI_UINT8_T");
	case MPI_UINT16_T: return(" MPI_UINT16_T");
	case MPI_UINT32_T: return(" MPI_UINT32_T");
	case MPI_UINT64_T: return(" MPI_UINT64_T");
	case MPI_BYTE: return(" MPI_BYTE");
	case MPI_PACKED: return(" MPI_PACKED");
	


	default				 : return(" wrong op");
	}
	*/
}


int (*hypre_get_thread_id)();		//function pointer to store function which will return thread id. C does not allow thread local storage. Hence pass a function
							//to return thread id - could be custom written function or could be library call such as omp_get_thread_num
							//WHOLE IMPLEMENTATION WILL FAIL IF THIS FUNCTION FAILS


volatile void * volatile g_collective_sync_buff=NULL;
std::atomic<int> g_collective_sync_count;
std::atomic<int> g_collective_sync_copy_count;
std::atomic<int> wait_for_all_threads_to_exit;
//std::atomic<int> g_barrier1;
//std::atomic<int> g_barrier;

//prefix ir refers intra-rank. Implementing following buffers to exchange intra rank data. Lockless.
struct irMessage{
	int size; //size in bytes
	volatile int consumed{0};	//sender sets to 0. receiver sets to 1. sender checks the flag in test/wait for it to be 1. receiver waits if the flag is not 1.
	void * buf;

	irMessage(){}
	irMessage(int s, void * b):size(s), buf(b), consumed(0){}
};

struct irKey{
	int tid; //src/dest thread id within rank
  int tag;

	irKey(int id, int t):tid(id), tag(t){}
  bool operator==(const irKey &other) const
  { return (tid == other.tid &&
            tag == other.tag);
  }
};

namespace std {	//needed for hash of irKey used by unordered_map
  template <>
  struct hash<irKey>
  {
    std::size_t operator()(const irKey& k) const
    {
      using std::size_t;
      using std::hash;

      return ((hash<int>()(k.tid)     ) ^
              (hash<int>()(k.tag) << 1) );
/*
		  size_t val = k.tid*1000000;
			val = val + k.tag;
			return val;
*/
    }
  };
}

//create 2 arrays of unordered_maps
std::vector<std::unordered_map<irKey, irMessage>> g_ir_send, g_ir_recv;


void irSend(void               *buf,
		HYPRE_Int           count,
		hypre_MPI_Datatype  datatype,
		HYPRE_Int           dest,	//this is dest thread id
		HYPRE_Int           tag)
{
	int size;
	MPI_Type_size(datatype, &size);	
	size = size*count;	//size in bytes;
	irKey k(dest, tag);
	irMessage m(size, buf);
	g_ir_send[g_thread_id][k] = m;

		/*
	if(g_ir_send[g_thread_id].find(k)==g_ir_send[g_thread_id].end()){
		irMessage m(size, buf);
		printf("%d send: from %d to %d tag %d size %d\n",g_thread_id, g_thread_id, dest,  tag, size);  fflush(stdout);
		g_ir_send[g_thread_id][k] = m;
	}
	else{
		printf("Error: rank %d: element with dest thread %d tag %d already present. Error at %s %d\n", g_rank, dest, tag, __FILE__, __LINE__);
		exit(1);
	}*/
	
}

void irRecv(void               *buf,
		HYPRE_Int           count,
		hypre_MPI_Datatype  datatype,
		HYPRE_Int           src,	//this is src thread id
		HYPRE_Int           tag)
{
	int size;
	MPI_Type_size(datatype, &size);	
	size = size*count;	//size in bytes;
	irKey k(src, tag);
	
	if(g_ir_recv[g_thread_id].find(k)==g_ir_recv[g_thread_id].end()){
		irMessage m(size, buf);
		//printf("%d recv: from %d to %d tag %d size %d\n",g_thread_id, src, g_thread_id,  tag, size);  fflush(stdout);
		g_ir_recv[g_thread_id][k] = m;
	}
	else{
		printf("Error: rank %d: element with src thread %d tag %d already present. Error at %s %d\n", g_rank, src, tag, __FILE__, __LINE__);
		exit(1);
	}	
}

int irTestAll()
{
  int doner=0;
	for(auto it = g_ir_recv[g_thread_id].begin(); it != g_ir_recv[g_thread_id].end(); it++){
		irKey k(g_thread_id, it->first.tag); //use self thread_id as recv thread id to search into sender's buffer
		irMessage m = it->second;
		auto its = g_ir_send[it->first.tid].find(k);
		if(its != g_ir_send[it->first.tid].end()){//element found
			if(its->second.consumed==0 && it->second.consumed==0){//element is not consumed.
				if(its->second.size != m.size){
					printf("error: send size is different than recv size %s %d\n", __FILE__, __LINE__); fflush(stdout);
					exit(1);
				}//if(its->second.size > m.size){
				//printf("%d recved: from %d to %d tag %d size %d\n",g_thread_id, it->first.tid, g_thread_id, it->first.tag, its->second.size); fflush(stdout);
				memcpy(m.buf, its->second.buf, its->second.size);
				its->second.consumed = 1;
				it->second.consumed = 1;
			}//element is not consumed			
		}//element found
			if(it->second.consumed==1)
				doner++;
	}//for(auto it = g_ir_recv[g_thread_id].begin(); it != g_ir_recv[g_thread_id].end(); it++){

	//wait for sends
	int dones=0;
	for(auto it = g_ir_send[g_thread_id].begin(); it != g_ir_send[g_thread_id].end(); it++)
		if(it->second.consumed == 1)	dones++;	

	int done = (doner == g_ir_recv[g_thread_id].size() && dones == g_ir_send[g_thread_id].size()) ? 1 : 0;
	//std::cout << g_rank << "\t" << doner << "\t" << g_ir_recv[g_thread_id].size() << "\t" << dones << "\t" << g_ir_send[g_thread_id].size() << "\n";
/*
	for(auto it = g_ir_send[g_thread_id].begin(); it != g_ir_send[g_thread_id].end(); it++)
		printf("%d pending sends: dest %d tag %d size %d consumed %d\n",g_thread_id, it->first.tid, it->first.tag, it->second.size, it->second.consumed);

	for(auto it = g_ir_recv[g_thread_id].begin(); it != g_ir_recv[g_thread_id].end(); it++)
		printf("%d pending recvs: src %d tag %d size %d consumed %d\n",g_thread_id, it->first.tid, it->first.tag, it->second.size, it->second.consumed);

  fflush(stdout);
*/
	return done;
}

void irWaitAll()
{
	while(irTestAll()==0);
//	#pragma omp barrier
	g_ir_recv[g_thread_id].clear();	
	//g_ir_send[g_thread_id].clear();
//	#pragma omp barrier
	//printf("%d send recv cleared\n", g_thread_id);	
}

void hypre_set_num_threads(int n, int (*f)())	//call from master thread BEFORE workers are created. not thread safe
{
	g_num_of_threads = n;
	hypre_get_thread_id=f;
	std::atomic_store_explicit (&g_collective_sync_count, 0, std::memory_order_seq_cst);
	std::atomic_store_explicit (&g_collective_sync_copy_count, 0, std::memory_order_seq_cst);
	std::atomic_store_explicit (&wait_for_all_threads_to_exit, 0, std::memory_order_seq_cst);
	//std::atomic_store_explicit (&g_barrier1, 0, std::memory_order_seq_cst);
	//std::atomic_store_explicit (&g_barrier, 0, std::memory_order_seq_cst);

	//HYPRE_TAG format: <bits for threads>,<bits for tag>. same number of bits will be used for src and dest thread ids.
	//e.g. to get same as default values, export HYPRE_TAG=8,16. This will assign left most 8 bits for src, 8 bits for dest and last 16 bits for tag
	const char* hypre_tag_str = getenv("HYPRE_TAG"); //use diff env variable if it conflicts with OMP. but using same will be consistent.
	char tag_str[16];

	if(hypre_tag_str)
	{
			strcpy(tag_str, hypre_tag_str);
			const char s[2] = ",";
			char *token;
			token = strtok(tag_str, s);	/* get the first token */
			int dest_bits = atoi(token);
			token = strtok(NULL, s);
			int tag_bits =  atoi(token);

			dest_multiplier = pow(2, tag_bits);
			src_multiplier = pow(2, (tag_bits + dest_bits));
			src_eliminate_and = src_multiplier - 1;

			//printf("dest_multiplier: %08x, src_multiplier: %08x, src_eliminate_and: %08x\n", dest_multiplier, src_multiplier, src_eliminate_and);
	}

	if(!g_MessageHead)	//allocate only already allocated.
	{
		g_MessageHead = (volatile MessageNode**) malloc(sizeof(MessageNode*) * g_num_of_threads);	//init data structs needed for multi threaded Improbe
		g_MessageTail = (volatile MessageNode**) malloc(sizeof(MessageNode*) * g_num_of_threads);
		g_MessageCount = (std::atomic<int> *) malloc(sizeof(std::atomic<int>) * g_num_of_threads);
		gLock = (volatile std::atomic_flag *) malloc(sizeof(std::atomic_flag) * g_num_of_threads);
		int i=0;
		for(i=0; i<g_num_of_threads; i++ )
		{
			g_MessageHead[i] = NULL;
			g_MessageTail[i] = NULL;
			std::atomic_store_explicit (&g_MessageCount[i], 0, std::memory_order_seq_cst);
			//std::atomic_store_explicit (&gLock[i], 0, std::memory_order_seq_cst);
			std::atomic_flag_clear(&gLock[i]);
		}
	}
	std::atomic_flag_clear(&gProbeLock);

	g_ir_send.resize(g_num_of_threads);
	g_ir_recv.resize(g_num_of_threads);

#if defined(HYPRE_USING_CUDA) 
	g_streams = (cudaStream_t*) malloc(g_num_of_threads * sizeof(cudaStream_t));
	for (int i = 0; i < g_num_of_threads; i++) 
	        cudaStreamCreate(&g_streams[i]);
#endif
}


void hypre_init_thread()	//to be called by every thread
{
	MPI_Comm_rank(comm, &g_rank);
	g_real_rank = g_rank;
	g_thread_id = hypre_get_thread_id();
	g_rank = g_rank * g_num_of_threads + g_thread_id;
	//printf("rank: %d, thread: %d\n", g_rank, g_thread_id);
	tl_prevMessageCount = 0;
	tl_prevtag = -1;
}

void hypre_destroy_thread()	//ideally should be called after every call to multithreaded Hypre solve to avoid any memory leaks. //call from master thread AFTER workers are joined
{
	int i;
	if(g_MessageHead)	//allocate only already allocated.
	{
		for(i=0; i<g_num_of_threads; i++ )
		{
			if(g_MessageHead[i])
			{
				MessageNode * start = (MessageNode *)g_MessageHead[i], *prev;
				while(start!=NULL)
				{
					prev = start;
					start = (MessageNode *)start->m_next;
					free(prev);
				}
				free(prev);	//last node
				g_MessageHead[i] = NULL;
			}
		}
		g_MessageHead = NULL;
	}
	if(g_MessageTail)	//allocate only already allocated.
	{
		for(i=0; i<g_num_of_threads; i++ )
		{
			if(g_MessageTail[i])
			{
				MessageNode * start = (MessageNode *)g_MessageTail[i], *prev;
				while(start!=NULL)
				{
					prev = start;
					start = (MessageNode *)start->m_next;
					free(prev);
				}
				free(prev);	//last node
				g_MessageTail[i] = NULL;
			}
		}
		g_MessageTail = NULL;
	}
	if(gLock) 			free((std::atomic_flag*)gLock);
	if(g_MessageCount) 	free(g_MessageCount);
	g_MessageHead = NULL, g_MessageTail = NULL, gLock = NULL, g_MessageCount = NULL;
	
#if defined(HYPRE_USING_CUDA)
	if(g_streams)
		free(g_streams);
	g_streams = NULL;
#endif

}


void hypre_copy_data_for_sync_collectives(void *sendbuf, void *recvbuf, HYPRE_Int count, hypre_MPI_Datatype datatype)
{
	int thread_id = g_thread_id;
	int i,j;
	//unsigned char *crecvbuf = (unsigned char *)recvbuf;	//cast into bytes
	unsigned char *csendbuf = (unsigned char *)sendbuf;
	int mpi_datatype_size;
	MPI_Type_size(datatype, &mpi_datatype_size);

/*	std::atomic_fetch_add_explicit(&g_barrier, 1, std::memory_order_seq_cst);
	while(std::atomic_load_explicit(&g_barrier, std::memory_order_seq_cst)%g_num_of_threads != 0);	//wait till all threads reach here.

	if(thread_id==0)	//this approach never frees  memory chunk. so there will be leak. but seems faster than
	{
		if(g_collective_sync_buff)
			free(g_collective_sync_buff);
		g_collective_sync_buff = NULL;
		std::atomic_store_explicit (&g_collective_sync_count, 0, std::memory_order_seq_cst);
		std::atomic_store_explicit (&g_collective_sync_copy_count, 0, std::memory_order_seq_cst);
		//std::atomic_store_explicit (&g_barrier, 0, std::memory_order_seq_cst);
	}

	std::atomic_fetch_add_explicit(&g_barrier1, 1, std::memory_order_seq_cst);
	while(std::atomic_load_explicit(&g_barrier1, std::memory_order_seq_cst)%g_num_of_threads != 0);	//wait till all threads reach here.
*/
	if(thread_id==0)	//if thread id is 0, allocate buffer. other wait till buffer is allocated.
	{
		//g_collective_sync_buff = (volatile void volatile*) ((volatile unsigned char volatile* )malloc(g_num_of_threads*mpi_datatype_size*count));
		g_collective_sync_buff = (volatile void *) ((volatile unsigned char * )malloc(g_num_of_threads*mpi_datatype_size*count));
	}
	else
	{
		//printf("%d - %d waiting for thread 0 to allocate\n", me, thread_id);
		while(!g_collective_sync_buff);//{printf("%d wait %s %d\n",thread_id,  __FILE__, __LINE__);}	//other threads wait till buffer is allocated
		//printf("%d - %d waiting for thread 0 to allocate finished\n", me, thread_id);
	}

	for(i=0; i < count; i++)
		for(j = 0; j<mpi_datatype_size; j++)
		{
			//((volatile  unsigned char volatile*)g_collective_sync_buff)[mpi_datatype_size*(i*g_num_of_threads + thread_id) + j] = csendbuf[i*mpi_datatype_size+j];	//each thread copies data at UNIQUE location. no data races. Group together numbers corresponding to 1 reduction operation
			((volatile  unsigned char*)g_collective_sync_buff)[mpi_datatype_size*(i*g_num_of_threads + thread_id) + j] = csendbuf[i*mpi_datatype_size+j];	//each thread copies data at UNIQUE location. no data races. Group together numbers corresponding to 1 reduction operation
			//printf("location : %d %d %d\n", thread_id, mpi_datatype_size*(i*g_num_of_threads + thread_id) + j, i*mpi_datatype_size+j);
		}
	std::atomic_fetch_add_explicit(&g_collective_sync_count, 1, std::memory_order_seq_cst);	//increment thread counter.

	//wait till all threads get here
	//printf("%d - %d waiting for other threads in hypre_copy_data_for_sync_collectives\n", me, thread_id);
	while(std::atomic_load_explicit(&g_collective_sync_count, std::memory_order_seq_cst) < g_num_of_threads);//{printf("%d wait %s %d\n",thread_id, __FILE__, __LINE__);}
	//printf("%d - %d waiting for other threads in hypre_copy_data_for_sync_collectives finished\n", me, thread_id);

}



# define loop_over_reductions(count, op,recv, recv_init)	\
	int i,j;							\
	for(i=0; i < count ;i++)			\
	{									\
		recv[i] = recv_init;			\
		int s = i*g_num_of_threads;		\
		int e = i*g_num_of_threads+g_num_of_threads;	\
		for(j = s; j<e; j++)			\
			op;							\
	}

//macro builds expressions for reductions and calls loop_over_reductions to reduce values in local buffers.
#define local_threaded_reduce(send, recv,  count, op, r)	\
{									\
	if(op==MPI_SUM)		{loop_over_reductions(count, recv[i] += send[j], recv, 0);	}	\
	else if(op==MPI_MAX)		{loop_over_reductions(count, recv[i] = (recv[i] > send[j] ? recv[i] : send[j]), recv, 0);}		\
	else if(op==MPI_MIN)		{loop_over_reductions(count, recv[i] = (recv[i] < send[j] ? recv[i] : send[j]), recv, 0);}		\
	else if(op==MPI_PROD)		{loop_over_reductions(count, recv[i] *= send[j], recv, 1);}		\
	else printf("error. Operation %s is not added in hypre mpistubs.c\n", getOpName(op));	\
}

#define local_threaded_reduce_for_INT(send, recv,  count, op, r)	\
{									\
	if(op==MPI_SUM)		{loop_over_reductions(count, recv[i] += send[j], recv, 0);	}	\
	else if(op==MPI_MAX)		{loop_over_reductions(count, recv[i] = (recv[i] > send[j] ? recv[i] : send[j]), recv, -9999999);}		\
	else if(op==MPI_MIN)		{loop_over_reductions(count, recv[i] = (recv[i] < send[j] ? recv[i] : send[j]), recv, 9999999);}		\
	else if(op==MPI_PROD)		{loop_over_reductions(count, recv[i] *= send[j], recv, 1);}		\
  	else if(op==MPI_LAND)		{loop_over_reductions(count, recv[i] = recv[i] && send[j], recv, 1);}		\
	else if(op==MPI_BAND)		{loop_over_reductions(count, recv[i] = recv[i] &  send[j], recv, 0xffffffff);}		\
	else if(op==MPI_LOR)		{loop_over_reductions(count, recv[i] = recv[i] || send[j], recv, 0);}		\
	else if(op==MPI_BOR)		{loop_over_reductions(count, recv[i] = recv[i] |  send[j], recv, 0);}		\
	else printf("error. Operation %s is not added in hypre mpistubs.c\n", getOpName(op));	\
}


/*traverse linked list and check if node is present*/
#define search_message_list(cond)	\
		tl_prevMessageCount = 0;	\
		while(start != NULL)		\
		{	\
			if(cond)		\
			{	\
				found = start;		\
				while(std::atomic_flag_test_and_set(&gLock[thread_id]));		\
				if(start == g_MessageHead[thread_id])		\
					g_MessageHead[thread_id] = found->m_next;	\
				if(start == g_MessageTail[thread_id])	\
					g_MessageTail[thread_id] = prev;	\
				if(prev)		\
					prev->m_next = start->m_next;		\
				std::atomic_flag_clear(&gLock[thread_id]);	\
				break;	\
			}	\
			prev = start;	\
			start = (MessageNode *)start->m_next;	\
			tl_prevMessageCount++;	\
		}

#endif


/******************************************************************************
* This routine is the same in both the sequential and normal cases
*
* The 'comm' argument for MPI_Comm_f2c is MPI_Fint, which is always the size of
* a Fortran integer and hence usually the size of hypre_int.
****************************************************************************/

hypre_MPI_Comm
hypre_MPI_Comm_f2c( hypre_int comm )
{
#ifdef HYPRE_HAVE_MPI_COMM_F2C
return (hypre_MPI_Comm) MPI_Comm_f2c(comm);
#else
return (hypre_MPI_Comm) (size_t)comm;
#endif
}

/******************************************************************************
* MPI stubs to generate serial codes without mpi
*****************************************************************************/

#ifdef HYPRE_SEQUENTIAL

HYPRE_Int
hypre_MPI_Init( hypre_int   *argc,
char      ***argv )
{
return(0);
}

HYPRE_Int
hypre_MPI_Finalize( )
{
return(0);
}

HYPRE_Int
hypre_MPI_Abort( hypre_MPI_Comm comm,
HYPRE_Int      errorcode )
{
return(0);
}

HYPRE_Real
hypre_MPI_Wtime( )
{
return(0.0);
}

HYPRE_Real
hypre_MPI_Wtick( )
{
return(0.0);
}

HYPRE_Int
hypre_MPI_Barrier( hypre_MPI_Comm comm )
{
return(0);
}

HYPRE_Int
hypre_MPI_Comm_create( hypre_MPI_Comm   comm,
hypre_MPI_Group  group,
hypre_MPI_Comm  *newcomm )
{
return(0);
}

HYPRE_Int
hypre_MPI_Comm_dup( hypre_MPI_Comm  comm,
hypre_MPI_Comm *newcomm )
{
return(0);
}

HYPRE_Int
hypre_MPI_Comm_size( hypre_MPI_Comm  comm,
HYPRE_Int      *size )
{ 
*size = 1;
return(0);
}

HYPRE_Int
hypre_MPI_Comm_rank( hypre_MPI_Comm  comm,
HYPRE_Int      *rank )
{ 
*rank = 0;
return(0);
}

HYPRE_Int
hypre_MPI_Comm_free( hypre_MPI_Comm *comm )
{
return 0;
}

HYPRE_Int
hypre_MPI_Comm_group( hypre_MPI_Comm   comm,
hypre_MPI_Group *group )
{
return(0);
}

HYPRE_Int
hypre_MPI_Comm_split( hypre_MPI_Comm  comm,
HYPRE_Int       n,
HYPRE_Int       m,
hypre_MPI_Comm *comms )
{
return(0);
}

HYPRE_Int
hypre_MPI_Group_incl( hypre_MPI_Group  group,
HYPRE_Int        n,
HYPRE_Int       *ranks,
hypre_MPI_Group *newgroup )
{
return(0);
}

HYPRE_Int
hypre_MPI_Group_free( hypre_MPI_Group *group )
{
return 0;
}

HYPRE_Int
hypre_MPI_Address( void           *location,
hypre_MPI_Aint *address )
{
return(0);
}

HYPRE_Int
hypre_MPI_Get_count( hypre_MPI_Status   *status,
hypre_MPI_Datatype  datatype,
HYPRE_Int          *count )
{
return(0);
}

HYPRE_Int
hypre_MPI_Alltoall( void               *sendbuf,
HYPRE_Int           sendcount,
hypre_MPI_Datatype  sendtype,
void               *recvbuf,
HYPRE_Int           recvcount,
hypre_MPI_Datatype  recvtype,
hypre_MPI_Comm      comm )
{
return(0);
}

HYPRE_Int
hypre_MPI_Allgather( void               *sendbuf,
HYPRE_Int           sendcount,
hypre_MPI_Datatype  sendtype,
void               *recvbuf,
HYPRE_Int           recvcount,
hypre_MPI_Datatype  recvtype,
hypre_MPI_Comm      comm ) 
{
HYPRE_Int i;

switch (sendtype)
{
case hypre_MPI_INT:
{
HYPRE_Int *crecvbuf = (HYPRE_Int *)recvbuf;
HYPRE_Int *csendbuf = (HYPRE_Int *)sendbuf;
for (i = 0; i < sendcount; i++)
{
	crecvbuf[i] = csendbuf[i];
}
} 
break;

case hypre_MPI_DOUBLE:
{
double *crecvbuf = (double *)recvbuf;
double *csendbuf = (double *)sendbuf;
for (i = 0; i < sendcount; i++)
{
	crecvbuf[i] = csendbuf[i];
}
} 
break;

case hypre_MPI_CHAR:
{
char *crecvbuf = (char *)recvbuf;
char *csendbuf = (char *)sendbuf;
for (i = 0; i < sendcount; i++)
{
	crecvbuf[i] = csendbuf[i];
}
} 
break;

case hypre_MPI_BYTE:
{
hypre_Memcpy(recvbuf,  sendbuf,  sendcount, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
} 
break;

case hypre_MPI_REAL:
{
HYPRE_Real *crecvbuf = (HYPRE_Real *)recvbuf;
HYPRE_Real *csendbuf = (HYPRE_Real *)sendbuf;
for (i = 0; i < sendcount; i++)
{
	crecvbuf[i] = csendbuf[i];
}
} 
break;

case hypre_MPI_COMPLEX:
{
HYPRE_Complex *crecvbuf = (HYPRE_Complex *)recvbuf;
HYPRE_Complex *csendbuf = (HYPRE_Complex *)sendbuf;
for (i = 0; i < sendcount; i++)
{
	crecvbuf[i] = csendbuf[i];
}
} 
break;
}

return(0);
}

HYPRE_Int
hypre_MPI_Allgatherv( void               *sendbuf,
HYPRE_Int           sendcount,
hypre_MPI_Datatype  sendtype,
void               *recvbuf,
HYPRE_Int          *recvcounts,
HYPRE_Int          *displs, 
hypre_MPI_Datatype  recvtype,
hypre_MPI_Comm      comm ) 
{ 
return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
	recvbuf, *recvcounts, recvtype, comm) );
}

HYPRE_Int
hypre_MPI_Gather( void               *sendbuf,
HYPRE_Int           sendcount,
hypre_MPI_Datatype  sendtype,
void               *recvbuf,
HYPRE_Int           recvcount,
hypre_MPI_Datatype  recvtype,
HYPRE_Int           root,
hypre_MPI_Comm      comm )
{
return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
	recvbuf, recvcount, recvtype, comm) );
}

HYPRE_Int
hypre_MPI_Gatherv( void              *sendbuf,
HYPRE_Int           sendcount,
hypre_MPI_Datatype  sendtype,
void               *recvbuf,
HYPRE_Int          *recvcounts,
HYPRE_Int          *displs,
hypre_MPI_Datatype  recvtype,
HYPRE_Int           root,
hypre_MPI_Comm      comm )
{
return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
	recvbuf, *recvcounts, recvtype, comm) );
}

HYPRE_Int
hypre_MPI_Scatter( void               *sendbuf,
HYPRE_Int           sendcount,
hypre_MPI_Datatype  sendtype,
void               *recvbuf,
HYPRE_Int           recvcount,
hypre_MPI_Datatype  recvtype,
HYPRE_Int           root,
hypre_MPI_Comm      comm )
{
return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
	recvbuf, recvcount, recvtype, comm) );
}

HYPRE_Int
hypre_MPI_Scatterv( void               *sendbuf,
HYPRE_Int           *sendcounts,
HYPRE_Int           *displs,
hypre_MPI_Datatype   sendtype,
void                *recvbuf,
HYPRE_Int            recvcount,
hypre_MPI_Datatype   recvtype,
HYPRE_Int            root,
hypre_MPI_Comm       comm )
{
return ( hypre_MPI_Allgather(sendbuf, *sendcounts, sendtype,
	recvbuf, recvcount, recvtype, comm) );
}

HYPRE_Int
hypre_MPI_Bcast( void               *buffer,
HYPRE_Int           count,
hypre_MPI_Datatype  datatype,
HYPRE_Int           root,
hypre_MPI_Comm      comm ) 
{ 
return(0);
}

HYPRE_Int
hypre_MPI_Send( void               *buf,
HYPRE_Int           count,
hypre_MPI_Datatype  datatype,
HYPRE_Int           dest,
HYPRE_Int           tag,
hypre_MPI_Comm      comm ) 
{ 
return(0);
}

HYPRE_Int
hypre_MPI_Recv( void               *buf,
HYPRE_Int           count,
hypre_MPI_Datatype  datatype,
HYPRE_Int           source,
HYPRE_Int           tag,
hypre_MPI_Comm      comm,
hypre_MPI_Status   *status )
{ 
return(0);
}

HYPRE_Int
hypre_MPI_Isend( void               *buf,
HYPRE_Int           count,
hypre_MPI_Datatype  datatype,
HYPRE_Int           dest,
HYPRE_Int           tag,
hypre_MPI_Comm      comm,
hypre_MPI_Request  *request )
{ 
return(0);
}

HYPRE_Int
hypre_MPI_Irecv( void               *buf,
HYPRE_Int           count,
hypre_MPI_Datatype  datatype,
HYPRE_Int           source,
HYPRE_Int           tag,
hypre_MPI_Comm      comm,
hypre_MPI_Request  *request )
{ 
return(0);
}

HYPRE_Int
hypre_MPI_Send_init( void               *buf,
HYPRE_Int           count,
hypre_MPI_Datatype  datatype,
HYPRE_Int           dest,
HYPRE_Int           tag, 
hypre_MPI_Comm      comm,
hypre_MPI_Request  *request )
{
return 0;
}

HYPRE_Int
hypre_MPI_Recv_init( void               *buf,
HYPRE_Int           count,
hypre_MPI_Datatype  datatype,
HYPRE_Int           dest,
HYPRE_Int           tag, 
hypre_MPI_Comm      comm,
hypre_MPI_Request  *request )
{
return 0;
}

HYPRE_Int
hypre_MPI_Irsend( void               *buf,
HYPRE_Int           count,
hypre_MPI_Datatype  datatype,
HYPRE_Int           dest,
HYPRE_Int           tag, 
hypre_MPI_Comm      comm,
hypre_MPI_Request  *request )
{
return 0;
}

HYPRE_Int
hypre_MPI_Startall( HYPRE_Int          count,
hypre_MPI_Request *array_of_requests )
{
return 0;
}

HYPRE_Int
hypre_MPI_Probe( HYPRE_Int         source,
HYPRE_Int         tag,
hypre_MPI_Comm    comm,
hypre_MPI_Status *status )
{
return 0;
}

HYPRE_Int
hypre_MPI_Iprobe( HYPRE_Int         source,
HYPRE_Int         tag,
hypre_MPI_Comm    comm,
HYPRE_Int        *flag,
hypre_MPI_Status *status )
{
return 0;
}

HYPRE_Int
hypre_MPI_Test( hypre_MPI_Request *request,
HYPRE_Int         *flag,
hypre_MPI_Status  *status )
{
*flag = 1;
return(0);
}

HYPRE_Int
hypre_MPI_Testall( HYPRE_Int          count,
hypre_MPI_Request *array_of_requests,
HYPRE_Int         *flag,
hypre_MPI_Status  *array_of_statuses )
{
*flag = 1;
return(0);
}

HYPRE_Int
hypre_MPI_Wait( hypre_MPI_Request *request,
hypre_MPI_Status  *status )
{
return(0);
}

HYPRE_Int
hypre_MPI_Waitall( HYPRE_Int          count,
hypre_MPI_Request *array_of_requests,
hypre_MPI_Status  *array_of_statuses )
{
return(0);
}

HYPRE_Int
hypre_MPI_Waitany( HYPRE_Int          count,
hypre_MPI_Request *array_of_requests,
HYPRE_Int         *index,
hypre_MPI_Status  *status )
{
return(0);
}

HYPRE_Int
hypre_MPI_Allreduce( void              *sendbuf,
void              *recvbuf,
HYPRE_Int          count,
hypre_MPI_Datatype datatype,
hypre_MPI_Op       op,
hypre_MPI_Comm     comm )
{ 
HYPRE_Int i;

switch (datatype)
{
case hypre_MPI_INT:
{
HYPRE_Int *crecvbuf = (HYPRE_Int *)recvbuf;
HYPRE_Int *csendbuf = (HYPRE_Int *)sendbuf;
for (i = 0; i < count; i++)
{
	crecvbuf[i] = csendbuf[i];
}

} 
break;

case hypre_MPI_DOUBLE:
{
double *crecvbuf = (double *)recvbuf;
double *csendbuf = (double *)sendbuf;
for (i = 0; i < count; i++)
{
	crecvbuf[i] = csendbuf[i];
}
} 
break;

case hypre_MPI_CHAR:
{
char *crecvbuf = (char *)recvbuf;
char *csendbuf = (char *)sendbuf;
for (i = 0; i < count; i++)
{
	crecvbuf[i] = csendbuf[i];
}
} 
break;

case hypre_MPI_BYTE:
{
hypre_Memcpy(recvbuf,  sendbuf,  count, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
} 
break;

case hypre_MPI_REAL:
{
HYPRE_Real *crecvbuf = (HYPRE_Real *)recvbuf;
HYPRE_Real *csendbuf = (HYPRE_Real *)sendbuf;
for (i = 0; i < count; i++)
{
	crecvbuf[i] = csendbuf[i];
}
} 
break;

case hypre_MPI_COMPLEX:
{
HYPRE_Complex *crecvbuf = (HYPRE_Complex *)recvbuf;
HYPRE_Complex *csendbuf = (HYPRE_Complex *)sendbuf;
for (i = 0; i < count; i++)
{
	crecvbuf[i] = csendbuf[i];
}
} 
break;
}

return 0;
}

HYPRE_Int
hypre_MPI_Reduce( void               *sendbuf,
void               *recvbuf,
HYPRE_Int           count,
hypre_MPI_Datatype  datatype,
hypre_MPI_Op        op,
HYPRE_Int           root,
hypre_MPI_Comm      comm )
{ 
hypre_MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
return 0;
}

HYPRE_Int
hypre_MPI_Scan( void               *sendbuf,
void               *recvbuf,
HYPRE_Int           count,
hypre_MPI_Datatype  datatype,
hypre_MPI_Op        op,
hypre_MPI_Comm      comm )
{ 
hypre_MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
return 0;
}

HYPRE_Int
hypre_MPI_Request_free( hypre_MPI_Request *request )
{
return 0;
}

HYPRE_Int
hypre_MPI_Type_contiguous( HYPRE_Int           count,
hypre_MPI_Datatype  oldtype,
hypre_MPI_Datatype *newtype )
{
return(0);
}

HYPRE_Int
hypre_MPI_Type_vector( HYPRE_Int           count,
HYPRE_Int           blocklength,
HYPRE_Int           stride,
hypre_MPI_Datatype  oldtype,
hypre_MPI_Datatype *newtype )
{
return(0);
}

HYPRE_Int
hypre_MPI_Type_hvector( HYPRE_Int           count,
HYPRE_Int           blocklength,
hypre_MPI_Aint      stride,
hypre_MPI_Datatype  oldtype,
hypre_MPI_Datatype *newtype )
{
return(0);
}

HYPRE_Int
hypre_MPI_Type_struct( HYPRE_Int           count,
HYPRE_Int          *array_of_blocklengths,
hypre_MPI_Aint     *array_of_displacements,
hypre_MPI_Datatype *array_of_types,
hypre_MPI_Datatype *newtype )
{
return(0);
}

HYPRE_Int
hypre_MPI_Type_commit( hypre_MPI_Datatype *datatype )
{
return(0);
}

HYPRE_Int
hypre_MPI_Type_free( hypre_MPI_Datatype *datatype )
{
return(0);
}

HYPRE_Int
hypre_MPI_Op_create( hypre_MPI_User_function *function, hypre_int commute, hypre_MPI_Op *op )
{
return(0);
}

HYPRE_Int
hypre_MPI_Op_free( hypre_MPI_Op *op )
{
return(0);
}

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
HYPRE_Int hypre_MPI_Comm_split_type( hypre_MPI_Comm comm, HYPRE_Int split_type, HYPRE_Int key, hypre_MPI_Info info, hypre_MPI_Comm *newcomm )
{
return (0);
}

HYPRE_Int hypre_MPI_Info_create( hypre_MPI_Info *info )
{
return (0);
}

HYPRE_Int hypre_MPI_Info_free( hypre_MPI_Info *info )
{
return (0);
}
#endif

/******************************************************************************
* MPI stubs to do casting of HYPRE_Int and hypre_int correctly
*****************************************************************************/

#else

HYPRE_THREAD_LOCAL_EP double _hypre_comm_time=0.0;

#define start_time()	\
struct timeval  tv1, tv2;	\
gettimeofday(&tv1, NULL);	

#define end_time()	\
gettimeofday(&tv2, NULL);	\
_hypre_comm_time += (double) (tv2.tv_usec - tv1.tv_usec) / 1000000.0 + (double) (tv2.tv_sec - tv1.tv_sec);

HYPRE_Int
hypre_MPI_Init( hypre_int   *argc,
		char      ***argv )
{
	return (HYPRE_Int) MPI_Init(argc, argv);
}

HYPRE_Int
hypre_MPI_Finalize( )
{
	return (HYPRE_Int) MPI_Finalize();
}

HYPRE_Int
hypre_MPI_Abort( hypre_MPI_Comm comm,
		HYPRE_Int      errorcode )
{
	return (HYPRE_Int) MPI_Abort(comm, (hypre_int)errorcode);
}

HYPRE_Real
hypre_MPI_Wtime( )
{
	return MPI_Wtime();
}

HYPRE_Real
hypre_MPI_Wtick( )
{
	return MPI_Wtick();
}

HYPRE_Int
hypre_MPI_Barrier( hypre_MPI_Comm comm )
{
	start_time();
	HYPRE_Int ret = (HYPRE_Int) MPI_Barrier(comm);
	end_time();
	return ret;
}

HYPRE_Int
hypre_MPI_Comm_create( hypre_MPI_Comm   comm,
		hypre_MPI_Group  group,
		hypre_MPI_Comm  *newcomm )
{
	start_time();
	HYPRE_Int ret = (HYPRE_Int) MPI_Comm_create(comm, group, newcomm);
	end_time();
	return ret;
}

HYPRE_Int
hypre_MPI_Comm_dup( hypre_MPI_Comm  comm,
		hypre_MPI_Comm *newcomm )
{
	start_time();
	HYPRE_Int ret =  (HYPRE_Int) MPI_Comm_dup(comm, newcomm);
	end_time();
	return ret;
}

HYPRE_Int
hypre_MPI_Comm_size( hypre_MPI_Comm  comm,
		HYPRE_Int      *size )
{
	start_time();
	hypre_int mpi_size;
	HYPRE_Int ierr;
	ierr = (HYPRE_Int) MPI_Comm_size(comm, &mpi_size);
#ifdef HYPRE_USING_MPI_EP
   *size = (HYPRE_Int) (mpi_size * g_num_of_threads);
#else
   *size = (HYPRE_Int) mpi_size;
#endif

   end_time();
   return ierr;
}

HYPRE_Int
hypre_MPI_Comm_rank( hypre_MPI_Comm  comm,
		HYPRE_Int      *rank )
{ 
	start_time();
#ifdef HYPRE_USING_MPI_EP
	HYPRE_Int ierr=MPI_SUCCESS;
	*rank = (HYPRE_Int) (g_rank);
#else
	hypre_int mpi_rank;
	HYPRE_Int ierr;
	ierr = (HYPRE_Int) MPI_Comm_rank(comm, &mpi_rank);
	*rank = (HYPRE_Int) mpi_rank;
#endif

	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Comm_free( hypre_MPI_Comm *comm )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Comm_free(comm);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Comm_group( hypre_MPI_Comm   comm,
		hypre_MPI_Group *group )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Comm_group(comm, group);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Comm_split( hypre_MPI_Comm  comm,
		HYPRE_Int       n,
		HYPRE_Int       m,
		hypre_MPI_Comm *comms )
{
	start_time();
	HYPRE_Int ierr =  (HYPRE_Int) MPI_Comm_split(comm, (hypre_int)n, (hypre_int)m, comms);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Group_incl( hypre_MPI_Group  group,
		HYPRE_Int        n,
		HYPRE_Int       *ranks,
		hypre_MPI_Group *newgroup )
{
	start_time();
	hypre_int *mpi_ranks;
	HYPRE_Int  i;
	HYPRE_Int  ierr;

	mpi_ranks = hypre_TAlloc(hypre_int,  n, HYPRE_MEMORY_HOST);
	for (i = 0; i < n; i++)
	{
		mpi_ranks[i] = (hypre_int) ranks[i];
	}
	ierr = (HYPRE_Int) MPI_Group_incl(group, (hypre_int)n, mpi_ranks, newgroup);
	hypre_TFree(mpi_ranks, HYPRE_MEMORY_HOST);

	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Group_free( hypre_MPI_Group *group )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Group_free(group);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Address( void           *location,
		hypre_MPI_Aint *address )
{
	start_time();
#if MPI_VERSION > 1
	HYPRE_Int ierr = (HYPRE_Int) MPI_Get_address(location, address);
#else
	HYPRE_Int ierr = (HYPRE_Int) MPI_Address(location, address);
#endif
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Get_count( hypre_MPI_Status   *status,
		hypre_MPI_Datatype  datatype,
		HYPRE_Int          *count )
{
	start_time();
	hypre_int mpi_count;
	HYPRE_Int ierr;

#ifdef HYPRE_USING_MPI_EP
	//damodar
	int temp_tag = status->MPI_TAG;
	int temp_src = status->MPI_SOURCE;
	//status->MPI_TAG = temp_tag == MPI_ANY_TAG ? MPI_ANY_TAG : (int)0x0000ffff * temp_src + temp_tag % (int)0x0000ffff;
	status->MPI_TAG = temp_tag == MPI_ANY_TAG ? MPI_ANY_TAG : (temp_src * src_multiplier) + (g_rank * dest_multiplier) + temp_tag % dest_multiplier;
	status->MPI_SOURCE = temp_src /g_num_of_threads;
	ierr = (HYPRE_Int) MPI_Get_count(status, datatype, &mpi_count);
	*count = (HYPRE_Int) mpi_count;
	status->MPI_TAG = temp_tag;
	status->MPI_SOURCE = temp_src;
#else
	ierr = (HYPRE_Int) MPI_Get_count(status, datatype, &mpi_count);
	*count = (HYPRE_Int) mpi_count;
#endif

	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Alltoall( void               *sendbuf,
		HYPRE_Int           sendcount,
		hypre_MPI_Datatype  sendtype,
		void               *recvbuf,
		HYPRE_Int           recvcount,
		hypre_MPI_Datatype  recvtype,
		hypre_MPI_Comm      comm )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Alltoall(sendbuf, (hypre_int)sendcount, sendtype,
			recvbuf, (hypre_int)recvcount, recvtype, comm);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Allgather( void               *sendbuf,
		HYPRE_Int           sendcount,
		hypre_MPI_Datatype  sendtype,
		void               *recvbuf,
		HYPRE_Int           recvcount,
		hypre_MPI_Datatype  recvtype,
		hypre_MPI_Comm      comm ) 
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Allgather(sendbuf, (hypre_int)sendcount, sendtype,
			recvbuf, (hypre_int)recvcount, recvtype, comm);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Allgatherv( void               *sendbuf,
		HYPRE_Int           sendcount,
		hypre_MPI_Datatype  sendtype,
		void               *recvbuf,
		HYPRE_Int          *recvcounts,
		HYPRE_Int          *displs, 
		hypre_MPI_Datatype  recvtype,
		hypre_MPI_Comm      comm ) 
{
	start_time();
	hypre_int *mpi_recvcounts, *mpi_displs, csize;
	HYPRE_Int  i;
	HYPRE_Int  ierr;

	MPI_Comm_size(comm, &csize);
	mpi_recvcounts = hypre_TAlloc(hypre_int,  csize, HYPRE_MEMORY_HOST);
	mpi_displs = hypre_TAlloc(hypre_int,  csize, HYPRE_MEMORY_HOST);
	for (i = 0; i < csize; i++)
	{
		mpi_recvcounts[i] = (hypre_int) recvcounts[i];
		mpi_displs[i] = (hypre_int) displs[i];
	}
	ierr = (HYPRE_Int) MPI_Allgatherv(sendbuf, (hypre_int)sendcount, sendtype,
			recvbuf, mpi_recvcounts, mpi_displs, 
			recvtype, comm);
	hypre_TFree(mpi_recvcounts, HYPRE_MEMORY_HOST);
	hypre_TFree(mpi_displs, HYPRE_MEMORY_HOST);

	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Gather( void               *sendbuf,
		HYPRE_Int           sendcount,
		hypre_MPI_Datatype  sendtype,
		void               *recvbuf,
		HYPRE_Int           recvcount,
		hypre_MPI_Datatype  recvtype,
		HYPRE_Int           root,
		hypre_MPI_Comm      comm )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Gather(sendbuf, (hypre_int) sendcount, sendtype,
			recvbuf, (hypre_int) recvcount, recvtype,
			(hypre_int)root, comm);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Gatherv(void               *sendbuf,
		HYPRE_Int           sendcount,
		hypre_MPI_Datatype  sendtype,
		void               *recvbuf,
		HYPRE_Int          *recvcounts,
		HYPRE_Int          *displs,
		hypre_MPI_Datatype  recvtype,
		HYPRE_Int           root,
		hypre_MPI_Comm      comm )
{
	start_time();
	hypre_int *mpi_recvcounts = NULL;
	hypre_int *mpi_displs = NULL;
	hypre_int csize, croot;
	HYPRE_Int  i;
	HYPRE_Int  ierr;

	MPI_Comm_size(comm, &csize);
	MPI_Comm_rank(comm, &croot);
	if (croot == (hypre_int) root)
	{
		mpi_recvcounts = hypre_TAlloc(hypre_int,  csize, HYPRE_MEMORY_HOST);
		mpi_displs = hypre_TAlloc(hypre_int,  csize, HYPRE_MEMORY_HOST);
		for (i = 0; i < csize; i++)
		{
			mpi_recvcounts[i] = (hypre_int) recvcounts[i];
			mpi_displs[i] = (hypre_int) displs[i];
		}
	}
	ierr = (HYPRE_Int) MPI_Gatherv(sendbuf, (hypre_int)sendcount, sendtype,
			recvbuf, mpi_recvcounts, mpi_displs, 
			recvtype, (hypre_int) root, comm);
	hypre_TFree(mpi_recvcounts, HYPRE_MEMORY_HOST);
	hypre_TFree(mpi_displs, HYPRE_MEMORY_HOST);

	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Scatter( void               *sendbuf,
		HYPRE_Int           sendcount,
		hypre_MPI_Datatype  sendtype,
		void               *recvbuf,
		HYPRE_Int           recvcount,
		hypre_MPI_Datatype  recvtype,
		HYPRE_Int           root,
		hypre_MPI_Comm      comm )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Scatter(sendbuf, (hypre_int)sendcount, sendtype,
			recvbuf, (hypre_int)recvcount, recvtype,
			(hypre_int)root, comm);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Scatterv(void               *sendbuf,
		HYPRE_Int          *sendcounts,
		HYPRE_Int          *displs,
		hypre_MPI_Datatype  sendtype,
		void               *recvbuf,
		HYPRE_Int           recvcount,
		hypre_MPI_Datatype  recvtype,
		HYPRE_Int           root,
		hypre_MPI_Comm      comm )
{
	start_time();
	hypre_int *mpi_sendcounts = NULL;
	hypre_int *mpi_displs = NULL;
	hypre_int csize, croot;
	HYPRE_Int  i;
	HYPRE_Int  ierr;

	MPI_Comm_size(comm, &csize);
	MPI_Comm_rank(comm, &croot);
	if (croot == (hypre_int) root)
	{
		mpi_sendcounts = hypre_TAlloc(hypre_int,  csize, HYPRE_MEMORY_HOST);
		mpi_displs = hypre_TAlloc(hypre_int,  csize, HYPRE_MEMORY_HOST);
		for (i = 0; i < csize; i++)
		{
			mpi_sendcounts[i] = (hypre_int) sendcounts[i];
			mpi_displs[i] = (hypre_int) displs[i];
		}
	}
	ierr = (HYPRE_Int) MPI_Scatterv(sendbuf, mpi_sendcounts, mpi_displs, sendtype,
			recvbuf, (hypre_int) recvcount, 
			recvtype, (hypre_int) root, comm);
	hypre_TFree(mpi_sendcounts, HYPRE_MEMORY_HOST);
	hypre_TFree(mpi_displs, HYPRE_MEMORY_HOST);

	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Bcast( void               *buffer,
		HYPRE_Int           count,
		hypre_MPI_Datatype  datatype,
		HYPRE_Int           root,
		hypre_MPI_Comm      comm ) 
{ 
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Bcast(buffer, (hypre_int)count, datatype,
			(hypre_int)root, comm);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Send( void               *buf,
		HYPRE_Int           count,
		hypre_MPI_Datatype  datatype,
		HYPRE_Int           dest,
		HYPRE_Int           tag,
		hypre_MPI_Comm      comm ) 
{ 
	start_time();
	HYPRE_Int ierr;

#ifdef HYPRE_USING_MPI_EP
	int new_tag, real_dest = dest / g_num_of_threads;
	int src_thread = g_thread_id, dest_thread = dest % g_num_of_threads;
	new_tag = (src_thread * src_multiplier) + (dest_thread * dest_multiplier) + tag % dest_multiplier;

	if(tag > dest_multiplier)
		printf("###### increase bits for tag by environment variable HYPRE_TAG. %d send: dest %d, real dest %d, tag %d, new_tag %d ########\n", g_rank, dest, real_dest, tag, new_tag);

#ifdef debug_mpi_calls
	printf("%d send: dest %d, real dest %d, tag %d, new_tag %d\n", g_rank, dest, real_dest, tag, new_tag);
#endif


	ierr =  (HYPRE_Int) MPI_Send(buf, (hypre_int)count, datatype, (hypre_int)real_dest, (hypre_int)new_tag, comm);

#else
	int new_tag=tag, rank=g_rank, real_dest = dest;
	printf("%d send : dest %d, real dest %d, tag %d, new_tag %d\n", rank ,dest, real_dest, tag, new_tag);
	ierr = (HYPRE_Int) MPI_Send(buf, (hypre_int)count, datatype,
                               (hypre_int)dest, (hypre_int)tag, comm);
#endif	
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Recv( void               *buf,
		HYPRE_Int           count,
		hypre_MPI_Datatype  datatype,
		HYPRE_Int           source,
		HYPRE_Int           tag,
		hypre_MPI_Comm      comm,
		hypre_MPI_Status   *status )
{ 
	start_time();

	HYPRE_Int ierr ;
#ifdef HYPRE_USING_MPI_EP
	int real_source = source == MPI_ANY_SOURCE ? MPI_ANY_SOURCE : source / g_num_of_threads;
	int dest_thread = g_thread_id, src_thread = source % g_num_of_threads;
	int new_tag = tag == MPI_ANY_TAG ? MPI_ANY_TAG : (src_thread * src_multiplier) + (dest_thread * dest_multiplier) + tag % dest_multiplier;

	if(source == MPI_ANY_SOURCE && tag != MPI_ANY_TAG)	//not handled cause we need to know src_thread to calculate new_tag
		printf("condition of source == MPI_ANY_SOURCE && tag != MPI_ANY_TAG is NOT handled in hypre_MPI_Recv\n");

#ifdef debug_mpi_calls
	printf("%d recv: source %d, real source %d, tag %d, new_tag %d\n", g_rank ,source, real_source, tag, new_tag);
#endif

	ierr = MPI_Recv(buf, (hypre_int)count, datatype, (hypre_int)real_source, (hypre_int)new_tag, comm, status);

	if(status != MPI_STATUS_IGNORE)
	{
		   int incoming_tag = status->MPI_TAG;
		   int src_thread =  incoming_tag / src_multiplier;
		   incoming_tag = incoming_tag & src_eliminate_and;
		   int dest_thread = incoming_tag / dest_multiplier;
		   //incoming_tag = incoming_tag - src_thread * (int)dest_multiplier;
		   incoming_tag = incoming_tag - dest_thread * (int)dest_multiplier;
		   status->MPI_SOURCE = status->MPI_SOURCE * g_num_of_threads + src_thread;
		   status->MPI_TAG = incoming_tag;
	}

#else
	int new_tag=tag, rank=g_rank, real_source = source;
	printf("%d recv : source %d, real source %d, tag %d, new_tag %d\n", rank ,source, real_source, tag, new_tag);
	ierr = (HYPRE_Int) MPI_Recv(buf, (hypre_int)count, datatype,
                               (hypre_int)source, (hypre_int)tag, comm, status);
#endif

	end_time();
	return ierr;
}

thread_local bool do_setup=true;

HYPRE_Int
hypre_MPI_Isend( void               *buf,
		HYPRE_Int           count,
		hypre_MPI_Datatype  datatype,
		HYPRE_Int           dest,
		HYPRE_Int           tag,
		hypre_MPI_Comm      comm,
		hypre_MPI_Request  *request )
{ 
	start_time();
	HYPRE_Int ierr;

#ifdef HYPRE_USING_MPI_EP

	int new_tag, real_dest = dest / g_num_of_threads;
	int src_thread = g_thread_id , dest_thread = dest % g_num_of_threads;
	new_tag = (src_thread * src_multiplier) + (dest_thread * dest_multiplier) + tag % dest_multiplier;

	//printf("%d isend: dest %d tag %d size %d real_dest %d g_real_rank %d\n", g_thread_id, dest, tag, count, real_dest, g_real_rank);

	if(real_dest == g_real_rank && do_setup==false){
		irSend(buf, count, datatype, dest_thread, tag);
		*request = MPI_REQUEST_NULL;
		return MPI_SUCCESS;
	}

	if(tag > dest_multiplier)
		printf("###### increase bits for tag by environment variable HYPRE_TAG. %d send: dest %d, real dest %d, tag %d, new_tag %d ########\n", g_rank, dest, real_dest, tag, new_tag);

	ierr = (HYPRE_Int) MPI_Isend(buf, (hypre_int)count, datatype, (hypre_int)real_dest, (hypre_int)new_tag, comm, request);

#else
	int new_tag=tag, rank=g_rank, real_dest = dest;
	//printf("%d send i: dest %d, real dest %d, tag %d, new_tag %d\n", rank ,dest, real_dest, tag, new_tag);
	ierr = (HYPRE_Int) MPI_Isend(buf, (hypre_int)count, datatype,
			(hypre_int)dest, (hypre_int)tag, comm, request);
#endif

	end_time();
	return ierr;
}

HYPRE_Int hypre_MPI_Mrecv(void 				*buf,
					HYPRE_Int 			count,
					hypre_MPI_Datatype 	datatype,
					hypre_MPI_Message 		*message,
					hypre_MPI_Status 	*status)
{
#ifdef debug_mpi_calls
	printf("%d mrecv: message %d\n", g_rank ,*message);
#endif

	   int i = (HYPRE_Int) MPI_Mrecv(buf, (hypre_int)count, datatype, message, status);
		if(status != MPI_STATUS_IGNORE)
		{
			   int incoming_tag = status->MPI_TAG;
			   int src_thread =  incoming_tag / src_multiplier;
			   incoming_tag = incoming_tag & src_eliminate_and;
			   int dest_thread = incoming_tag / dest_multiplier;
			   //incoming_tag = incoming_tag - src_thread * (int)dest_multiplier;
			   incoming_tag = incoming_tag - dest_thread * (int)dest_multiplier;
			   status->MPI_SOURCE = status->MPI_SOURCE * g_num_of_threads + src_thread;
			   status->MPI_TAG = incoming_tag;
		}

		return i;
}

HYPRE_Int hypre_MPI_Imrecv(void 				*buf,
					HYPRE_Int 			count,
					hypre_MPI_Datatype 	datatype,
					hypre_MPI_Message 		*message,
					hypre_MPI_Request *request)
{
#ifdef debug_mpi_calls
	printf("%d imrecv: message %d\n", g_rank ,*message);
#endif

	   return ((HYPRE_Int) MPI_Imrecv(buf, (hypre_int)count, datatype, message, request));
}

HYPRE_Int
hypre_MPI_Irecv( void               *buf,
		HYPRE_Int           count,
		hypre_MPI_Datatype  datatype,
		HYPRE_Int           source,
		HYPRE_Int           tag,
		hypre_MPI_Comm      comm,
		hypre_MPI_Request  *request )
{ 
	start_time();
	HYPRE_Int ierr ;


#ifdef HYPRE_USING_MPI_EP
	int real_source = source == MPI_ANY_SOURCE ? MPI_ANY_SOURCE : source / g_num_of_threads;
	int dest_thread = g_thread_id, src_thread = source % g_num_of_threads;
	int new_tag = tag == MPI_ANY_TAG ? MPI_ANY_TAG : (src_thread * src_multiplier) + (dest_thread * dest_multiplier) + tag % dest_multiplier;
	
	//printf("%d irecv: source %d tag %d size %d real_source %d g_real_rank %d\n", g_thread_id, source, tag, count, real_source, g_real_rank);

	if(real_source == g_real_rank && do_setup==false){
		irRecv(buf, count, datatype, src_thread, tag);
		*request = MPI_REQUEST_NULL;
		return MPI_SUCCESS;
	}

	if(source == MPI_ANY_SOURCE && tag != MPI_ANY_TAG)	//not handled cause we need to know src_thread to calculate new_tag
			printf("condition of source == MPI_ANY_SOURCE && tag != MPI_ANY_TAG is NOT handled in hypre_MPI_Irecv\n");

	ierr = (HYPRE_Int) MPI_Irecv(buf, (hypre_int)count, datatype, (hypre_int)real_source, (hypre_int)new_tag, comm, request);

#else
	int new_tag=tag, rank=g_rank, real_source = source;
	//printf("%d recv i: source %d, real source %d, tag %d, new_tag %d\n", rank ,source, real_source, tag, new_tag);
	ierr = (HYPRE_Int) MPI_Irecv(buf, (hypre_int)count, datatype,
                                (hypre_int)source, (hypre_int)tag, comm, request);
#endif
	
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Send_init( void               *buf,
		HYPRE_Int           count,
		hypre_MPI_Datatype  datatype,
		HYPRE_Int           dest,
		HYPRE_Int           tag, 
		hypre_MPI_Comm      comm,
		hypre_MPI_Request  *request )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Send_init(buf, (hypre_int)count, datatype,
			(hypre_int)dest, (hypre_int)tag,
			comm, request);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Recv_init( void               *buf,
		HYPRE_Int           count,
		hypre_MPI_Datatype  datatype,
		HYPRE_Int           dest,
		HYPRE_Int           tag, 
		hypre_MPI_Comm      comm,
		hypre_MPI_Request  *request )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Recv_init(buf, (hypre_int)count, datatype,
			(hypre_int)dest, (hypre_int)tag,
			comm, request);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Irsend( void               *buf,
		HYPRE_Int           count,
		hypre_MPI_Datatype  datatype,
		HYPRE_Int           dest,
		HYPRE_Int           tag, 
		hypre_MPI_Comm      comm,
		hypre_MPI_Request  *request )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Irsend(buf, (hypre_int)count, datatype,
			(hypre_int)dest, (hypre_int)tag, comm, request);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Startall( HYPRE_Int          count,
		hypre_MPI_Request *array_of_requests )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Startall((hypre_int)count, array_of_requests);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Probe( HYPRE_Int         source,
		HYPRE_Int         tag,
		hypre_MPI_Comm    comm,
		hypre_MPI_Status *status )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Probe((hypre_int)source, (hypre_int)tag, comm, status);
	end_time();
	return ierr;
}


HYPRE_Int
hypre_MPI_Improbe( 	HYPRE_Int         source,
					HYPRE_Int         tag,
					hypre_MPI_Comm    comm,
					HYPRE_Int        *flag,
					hypre_MPI_Message 	 *mpi_message,
					hypre_MPI_Status *status )
{
#ifdef debug_mpi_calls
	printf("%d hypre_MPI_Improbe %d, %d\n",g_rank, source, tag);
#endif

	//return (HYPRE_Int) searchMessageQ(g_thread_id, source, tag, comm, flag, message, status);
	int thread_id = g_thread_id;

	*flag = 0;

	int ierr=MPI_SUCCESS, new_src=MPI_ANY_SOURCE, new_tag=MPI_ANY_TAG;
	MessageNode * start = (MessageNode *)g_MessageHead[thread_id], *found=NULL, *prev=NULL;

	//check link list only if tag of current call is different than prev call and if either there are new messages added in queue or queue was not checked till end on last search
	//tl_prevMessageCount is incremented for every node searched. If node is not found, tl_prevMessageCount will be equal to g_MessageCount[thread_id]
	// no need to traverse queue if g_MessageCount[thread_id] value is still same
	if(tag != tl_prevtag || tl_prevMessageCount < std::atomic_load_explicit(&g_MessageCount[thread_id], std::memory_order_seq_cst))
	{
		if(source == MPI_ANY_SOURCE && tag != MPI_ANY_TAG)	//as of now this is the only condition used.
		{
			search_message_list(tag == start->m_tag);
		}
		else if(source == MPI_ANY_SOURCE && tag == MPI_ANY_TAG)
		{
			search_message_list(1==1);	//any message will do, no need to check condition
			//traverse linked list only if new nodes are added since last checked.
		}
		else if (source != MPI_ANY_SOURCE && tag == MPI_ANY_TAG)
		{
			search_message_list(source == start->m_source);
			new_src = source / g_num_of_threads;
		}
		else //this is if(source != MPI_ANY_SOURCE && tag != MPI_ANY_TAG)
		{
			search_message_list(source == start->m_source && tag == start->m_tag)
			int real_src = source / g_num_of_threads;
			int dest_thread = thread_id , src_thread = source % g_num_of_threads;
			new_tag = (src_thread * src_multiplier) + (dest_thread * dest_multiplier) + tag % dest_multiplier;
			new_src = source / g_num_of_threads;
		}
	}
	tl_prevtag = tag;
	if(found)	//if node found copy return values and free node.
	{
		*flag=1;
		memcpy(mpi_message, &found->m_mpi_message, sizeof(MPI_Message));
		memcpy(status, &found->m_mpi_status, sizeof(MPI_Status));
		free(found);
		std::atomic_fetch_add_explicit(&g_MessageCount[thread_id], -1, std::memory_order_seq_cst);

#ifdef debug_mpi_calls
	printf("%d hypre_MPI_Improbe found in queue. message %d, source %d, tag %d\n",g_rank, *mpi_message, status->MPI_SOURCE, status->MPI_TAG);
#endif

	}
	else //if not found in existing list, issue Improbe
	{
		int mpi_flag=0, limit=0;
#ifdef debug_mpi_calls
	printf("%d not found, probing\n", g_rank);
#endif
		do
		{
			MPI_Status probe_status;
			MPI_Message probe_message;
			//if MPI_ANY_TAG is used when tag != MPI_ANY_TAG, Improbe reads and removes messages which are not intended for imrecv also. This causes wait / test for irecv to hang
			//to overcome it, first check head of queue with iprobe. If it fits tag passed on, read using improbe. else wait for test / wait to read head.
			//as exchange_data calls probe and test one after other, it should not take long time.
			//if(source == hypre_MPI_ANY_SOURCE && tag!=MPI_ANY_TAG)	//commenting for now as other conditions of iprobe not used
			//{
				//while(std::atomic_flag_test_and_set(&gProbeLock));	//should work without lock ideally. but not working
				mpi_flag=0;
				MPI_Iprobe(new_src, new_tag, comm, &mpi_flag, &probe_status);	//first probe and check if message is of desired tag, improbe only if tag is same.
				if(mpi_flag)
				{
					mpi_flag=0;
					int probe_tag = probe_status.MPI_TAG;
					int probe_src = probe_status.MPI_SOURCE;
					if((probe_tag & (dest_multiplier-1)) == tag) //check if message is of desired tag, improbe only if tag is same
						ierr = MPI_Improbe(probe_src, probe_tag, comm, &mpi_flag, &probe_message, &probe_status);
					else
					{
#ifdef debug_mpi_calls
						int incoming_tag = probe_status.MPI_TAG;
						int src =  incoming_tag / src_multiplier;
						incoming_tag = incoming_tag & src_eliminate_and;
						int dest = incoming_tag / dest_multiplier;
						incoming_tag = incoming_tag - dest * (int)dest_multiplier;
						int fake_src = probe_status.MPI_SOURCE * g_num_of_threads + src;
						printf("%d ignoring src %d, tag %d\n", g_rank, fake_src, incoming_tag);
#endif
						break;
					}

				}

				//std::atomic_flag_clear(&gProbeLock);
			//}
			/*else
			{
				ierr = MPI_Improbe(new_src, new_tag, comm, &mpi_flag, &probe_message, &probe_status);
			}*/
			if(mpi_flag)
			{
				int incoming_tag = probe_status.MPI_TAG;
				int src =  incoming_tag / src_multiplier;
				incoming_tag = incoming_tag & src_eliminate_and;
				int dest = incoming_tag / dest_multiplier;
				incoming_tag = incoming_tag - dest * (int)dest_multiplier;
				int fake_src = probe_status.MPI_SOURCE * g_num_of_threads + src;

				if(dest == thread_id && ( (source == MPI_ANY_SOURCE && tag == MPI_ANY_TAG ) ||
										(source != MPI_ANY_SOURCE && tag != MPI_ANY_TAG && tag == incoming_tag && source == fake_src) ||
										(source == hypre_MPI_ANY_SOURCE && tag!=MPI_ANY_TAG && tag == incoming_tag) ||
										(source != hypre_MPI_ANY_SOURCE && tag==MPI_ANY_TAG && source == fake_src)   ) )	//set values and return if message found
				{
					probe_status.MPI_SOURCE = probe_status.MPI_SOURCE * g_num_of_threads + src;
					probe_status.MPI_TAG = incoming_tag;
					*flag = 1;
					memcpy(mpi_message, &probe_message, sizeof(MPI_Message));
					memcpy(status, &probe_status, sizeof(MPI_Status));

#ifdef debug_mpi_calls
	printf("%d hypre_MPI_Improbe found by prob. message %d, source %d, tag %d\n",g_rank, probe_message, probe_status.MPI_SOURCE, probe_status.MPI_TAG);
#endif
					break;
				}
				else	//add into queue of dest thread
				{
					*flag = 0;
					volatile MessageNode * newMessage = (volatile MessageNode*)malloc(sizeof(MessageNode));
					memcpy((MPI_Message *) &newMessage->m_mpi_message, &probe_message, sizeof(MPI_Message));	//copy message and status and update status
					memcpy((MPI_Status *) &newMessage->m_mpi_status, &probe_status, sizeof(MPI_Status));
					newMessage->m_mpi_status.MPI_SOURCE = fake_src;
					newMessage->m_mpi_status.MPI_TAG = incoming_tag;
					newMessage->m_source = fake_src;
					newMessage->m_tag = incoming_tag;
					newMessage->m_next = NULL;
#ifdef debug_mpi_calls
	printf("%d hypre_MPI_Improbe adding to queue: dest: %d message %d, source %d, tag %d\n",g_rank, dest, probe_message, newMessage->m_mpi_status.MPI_SOURCE, newMessage->m_mpi_status.MPI_TAG);
#endif
					while(std::atomic_flag_test_and_set(&gLock[dest]));
					if(g_MessageTail[dest])	//if head and tail already exist, then use them else this is first node, assign new node to head and tail
					{
						g_MessageTail[dest]->m_next = newMessage;
						g_MessageTail[dest] = newMessage;
					}
					else	//this is first node, assign new node to head and tail
					{
						g_MessageHead[dest] = newMessage;
						g_MessageTail[dest] = newMessage;
					}
					std::atomic_flag_clear(&gLock[dest]);

					std::atomic_fetch_add_explicit(&g_MessageCount[dest], 1, std::memory_order_seq_cst);

				}
			}
			limit++;
		}while(mpi_flag==1 && limit < 10);	//check max 10 messages / call
	}

	return(ierr);
}

HYPRE_Int
hypre_MPI_Iprobe( HYPRE_Int         source,
		HYPRE_Int         tag,
		hypre_MPI_Comm    comm,
		HYPRE_Int        *flag,
		hypre_MPI_Status *status )
{
	start_time();
	hypre_int mpi_flag;
	HYPRE_Int ierr;
	

	//new_tag = (source * src_multiplier) + (rank * dest_multiplier) + tag % dest_multiplier;

#ifdef HYPRE_USING_MPI_EP

	int thread_id = g_thread_id;
	if(source == hypre_MPI_ANY_SOURCE && tag==MPI_ANY_TAG)
	{
		ierr = (HYPRE_Int) MPI_Iprobe((hypre_int)source, (hypre_int)tag, comm, &mpi_flag, status);
		//compare tag value with received tag and return accordingly
		*flag = (HYPRE_Int) mpi_flag;
		if(mpi_flag)
		{
			int incoming_tag = status->MPI_TAG;
			int src =  incoming_tag / src_multiplier;
			incoming_tag = incoming_tag & src_eliminate_and;
			int dest = incoming_tag / dest_multiplier;
			incoming_tag = incoming_tag - dest * (int)dest_multiplier;
			if(dest == thread_id)
			{
				status->MPI_SOURCE = status->MPI_SOURCE * g_num_of_threads + src;
				status->MPI_TAG = incoming_tag;	//big assumption: Original tag generated are ALWAYs < 0x0000ffff, else this will fail
			}

		}		
	}
	else if(source == hypre_MPI_ANY_SOURCE && tag!=MPI_ANY_TAG)	// can lead to deadlock. e.g. 0 sends 2 msgs with tag 100 and 200. At 1 200 arrives 1st and 1 probes for msg tag 100, MPI_IProbe will always return 200 and we never reach 100 for which 1 is probing
	{
		//using MPI_ANY_TAG is faster than generating tag for all sources. Use this if supported by MPI
		// have to probe with MPI_ANY_TAG, as without know source, tag can not be converted. This is bad. will lead to serialized probe and receive.
		*flag = 0;
		ierr = (HYPRE_Int) MPI_Iprobe((hypre_int)source, MPI_ANY_TAG, comm, &mpi_flag, status);
		if(mpi_flag)
		{
			int incoming_tag = status->MPI_TAG;
			int src =  incoming_tag / src_multiplier;
			incoming_tag = incoming_tag & src_eliminate_and;
			int dest = incoming_tag / dest_multiplier;
			incoming_tag = incoming_tag - dest * (int)dest_multiplier;

			if(dest == thread_id && tag == incoming_tag)
			{
				status->MPI_SOURCE = status->MPI_SOURCE * g_num_of_threads + src;
				status->MPI_TAG = incoming_tag;
				*flag = 1;
			}
		}

		/*
		   int i, dest_thread = g_thread_id, j;
		   for(i=0; i<g_num_of_threads; i++)
		   //for(i=dest_thread, j=0; j < g_num_of_threads; i=(i+1)%g_num_of_threads, j++)
		   {
				int  src_thread = i;
				int new_tag =  (src_thread * src_multiplier) + (dest_thread * dest_multiplier) + tag % dest_multiplier;
		 *flag = 0;
				ierr = (HYPRE_Int) MPI_Iprobe((hypre_int)source, new_tag, comm, &mpi_flag, status);
				if(mpi_flag)
				{
				   int incoming_tag = status->MPI_TAG;
				   int src =  incoming_tag / src_multiplier;
				   incoming_tag = incoming_tag & src_eliminate_and;
				   int dest = incoming_tag / dest_multiplier;
				   incoming_tag = incoming_tag - dest * (int)dest_multiplier;

				   if(dest == thread_id && tag == incoming_tag)
				   {
					   status->MPI_SOURCE = status->MPI_SOURCE * g_num_of_threads + src;
					   status->MPI_TAG = incoming_tag;
		 *flag = 1;
					   break;
				   }
				}
		   }
		 */

	}
	else if(source != hypre_MPI_ANY_SOURCE && tag==MPI_ANY_TAG)
	{
		*flag = 0;
		int real_src = source / g_num_of_threads;
		ierr = (HYPRE_Int) MPI_Iprobe((hypre_int)real_src, tag, comm, &mpi_flag, status);
		if(mpi_flag)
		{
			int incoming_tag = status->MPI_TAG;
			int src =  incoming_tag / src_multiplier;
			incoming_tag = incoming_tag & src_eliminate_and;
			int dest = incoming_tag / dest_multiplier;
			incoming_tag = incoming_tag - dest * (int)dest_multiplier;
			int fake_src = status->MPI_SOURCE * g_num_of_threads + src;
			if(dest == thread_id && source == fake_src)
			{
				status->MPI_SOURCE = fake_src;
				status->MPI_TAG = incoming_tag;
				*flag = 1;
			}
		}
	}
	else if(source != hypre_MPI_ANY_SOURCE && tag!=MPI_ANY_TAG)
	{
		*flag = 0;
		int real_src = source / g_num_of_threads;
		int dest_thread = g_thread_id , src_thread = source % g_num_of_threads;
		int new_tag = (src_thread * src_multiplier) + (dest_thread * dest_multiplier) + tag % dest_multiplier;
		ierr = (HYPRE_Int) MPI_Iprobe((hypre_int)real_src, new_tag, comm, &mpi_flag, status);
		if(mpi_flag)
		{
			int incoming_tag = status->MPI_TAG;
			int src =  incoming_tag / src_multiplier;
			incoming_tag = incoming_tag & src_eliminate_and;
			int dest = incoming_tag / dest_multiplier;
			incoming_tag = incoming_tag - dest * (int)dest_multiplier;
			int fake_src = status->MPI_SOURCE * g_num_of_threads + src;
			if(dest == thread_id && source == fake_src && tag == incoming_tag)
			{
				status->MPI_SOURCE = fake_src;
				status->MPI_TAG = incoming_tag;
				*flag = 1;
			}
		}
	}

#else
	ierr = (HYPRE_Int) MPI_Iprobe((hypre_int)source, (hypre_int)tag, comm, &mpi_flag, status);
	*flag = (HYPRE_Int) mpi_flag;	   
#endif	

	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Test( hypre_MPI_Request *request,
		HYPRE_Int         *flag,
		hypre_MPI_Status  *status )
{
	start_time();
	hypre_int mpi_flag;
	HYPRE_Int ierr;
	ierr = (HYPRE_Int) MPI_Test(request, &mpi_flag, status);
	*flag = (HYPRE_Int) mpi_flag;
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Testall( HYPRE_Int          count,
		hypre_MPI_Request *array_of_requests,
		HYPRE_Int         *flag,
		hypre_MPI_Status  *array_of_statuses )
{
	start_time();
	hypre_int mpi_flag;
	HYPRE_Int ierr;

	ierr = (HYPRE_Int) MPI_Testall((hypre_int)count, array_of_requests,
			&mpi_flag, array_of_statuses);
	*flag = (HYPRE_Int) mpi_flag;
/*
	int done = irTestAll();
	if(done==0){
		*flag=0;
		return ierr;
	}
*/
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Wait( hypre_MPI_Request *request,
		hypre_MPI_Status  *status )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Wait(request, status);
	end_time();
	return ierr;
}



HYPRE_Int
hypre_MPI_Waitall( HYPRE_Int          count,
		hypre_MPI_Request *array_of_requests,
		hypre_MPI_Status  *array_of_statuses )
{
	start_time();
	
	irWaitAll();

	HYPRE_Int ierr = (HYPRE_Int) MPI_Waitall((hypre_int)count,
			array_of_requests, array_of_statuses);


	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Waitany( HYPRE_Int          count,
		hypre_MPI_Request *array_of_requests,
		HYPRE_Int         *index,
		hypre_MPI_Status  *status )
{
	start_time();
	hypre_int mpi_index;
	HYPRE_Int ierr;
	ierr = (HYPRE_Int) MPI_Waitany((hypre_int)count, array_of_requests,
			&mpi_index, status);
	*index = (HYPRE_Int) mpi_index;
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Allreduce( void              *sendbuf,
		void              *recvbuf,
		HYPRE_Int          count,
		hypre_MPI_Datatype datatype,
		hypre_MPI_Op       op,
		hypre_MPI_Comm     comm )
{
	start_time();
	HYPRE_Int ierr = MPI_SUCCESS;

#ifdef HYPRE_USING_MPI_EP
	int i;//, j;
	int mpi_datatype_size;
	MPI_Type_size(datatype, &mpi_datatype_size);

	hypre_copy_data_for_sync_collectives(sendbuf, recvbuf, count, datatype);

	int thread_id = g_thread_id;

	if(thread_id==0)	//0 th thread reduces first across threads and then across ranks
	{
		std::atomic_store_explicit(&wait_for_all_threads_to_exit, g_num_of_threads, std::memory_order_seq_cst);

		//try macro somehow for operations and may be datatype as well???
		//int result;
		//print OPs. ints have some extra ops. that why failing????
		if(datatype==MPI_DOUBLE)
		{
			double * collective_sync_buff = (double *) g_collective_sync_buff;
			double * local_reduction = (double*) malloc(sizeof(double)*count);
			local_threaded_reduce(collective_sync_buff,local_reduction, count, op, result);
			MPI_Allreduce(local_reduction, recvbuf, (hypre_int)count, datatype, op, comm);
			free(local_reduction);
			collective_sync_buff = NULL;
		}
		else if(datatype==MPI_INT)
		{
			int * collective_sync_buff = (int *) g_collective_sync_buff;
			int * local_reduction = (int*) malloc(sizeof(int)*count);
			//int k=0;
			//for(k=0;k<count*g_num_of_threads; k++)
			//	printf("%d ",collective_sync_buff[k]);

			local_threaded_reduce_for_INT(collective_sync_buff,local_reduction, count, op, result);
			MPI_Allreduce(local_reduction, recvbuf, (hypre_int)count, datatype, op, comm);
			free(local_reduction);
			collective_sync_buff = NULL;
		}
		else if(datatype==MPI_FLOAT)
		{
			float * collective_sync_buff = (float *) g_collective_sync_buff;
			float * local_reduction = (float*) malloc(sizeof(float)*count);
			local_threaded_reduce(collective_sync_buff,local_reduction, count, op, result);
			MPI_Allreduce(local_reduction, recvbuf, (hypre_int)count, datatype, op, comm);
			free(local_reduction);
			collective_sync_buff = NULL;
		}
		else	printf("reduction for other datatypes not yet implemented in hypre mpistubs.c\n");

		for(i=0; i<count*mpi_datatype_size ; i++)
			((unsigned char *)g_collective_sync_buff)[i] = ((unsigned char *)recvbuf)[i];	//copy data

		//printf("%d reduced copying complete by thread 0\n", me);
		std::atomic_fetch_add_explicit(&g_collective_sync_copy_count, 1, std::memory_order_seq_cst);	//copying of data over
		//printf("%d - %d waiting for child threads to finish\n", g_rank, thread_id);
		while(std::atomic_load_explicit(&g_collective_sync_copy_count, std::memory_order_seq_cst) <  g_num_of_threads );//{printf("%d wait %s %d\n",thread_id, __FILE__, __LINE__);}	//thread 0 waits before freeing memory, till all other threads copying data

		free((void*) g_collective_sync_buff);
		g_collective_sync_buff = NULL;
		std::atomic_store_explicit(&g_collective_sync_copy_count, 0, std::memory_order_seq_cst);	//reset g_collective_sync_copy_count from prev reduce
		std::atomic_store_explicit(&g_collective_sync_count, 0, std::memory_order_seq_cst);	//signal other threads

		while(std::atomic_load_explicit(&wait_for_all_threads_to_exit, std::memory_order_seq_cst) > 1 );	//wait till all child threads exit.
	}
	else
	{
		while(std::atomic_load_explicit(&g_collective_sync_copy_count, std::memory_order_seq_cst) ==  0 );//{printf("%d wait %s %d\n",thread_id, __FILE__, __LINE__);}	//all other threads wait.
		//read values from global buffer
		for(i=0; i<count*mpi_datatype_size ; i++)
			((unsigned char *)recvbuf)[i] = ((unsigned char *)g_collective_sync_buff)[i];	//copy data

		std::atomic_fetch_add_explicit(&g_collective_sync_copy_count, 1, std::memory_order_seq_cst);	//copying of data over

		while(std::atomic_load_explicit(&g_collective_sync_count, std::memory_order_seq_cst) > 0 );	//wait till variables are reset or it conflicts with next reduce

		std::atomic_fetch_add_explicit(&wait_for_all_threads_to_exit, -1, std::memory_order_seq_cst);

	}

#else
   return (HYPRE_Int) MPI_Allreduce(sendbuf, recvbuf, (hypre_int)count,
                                    datatype, op, comm);
#endif	
	
	
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Reduce( void               *sendbuf,
		void               *recvbuf,
		HYPRE_Int           count,
		hypre_MPI_Datatype  datatype,
		hypre_MPI_Op        op,
		HYPRE_Int           root,
		hypre_MPI_Comm      comm )
{ 
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Reduce(sendbuf, recvbuf, (hypre_int)count,
			datatype, op, (hypre_int)root, comm);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Scan( void               *sendbuf,
		void               *recvbuf,
		HYPRE_Int           count,
		hypre_MPI_Datatype  datatype,
		hypre_MPI_Op        op,
		hypre_MPI_Comm      comm )
{ 
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Scan(sendbuf, recvbuf, (hypre_int)count,
			datatype, op, comm);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Request_free( hypre_MPI_Request *request )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Request_free(request);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Type_contiguous( HYPRE_Int           count,
		hypre_MPI_Datatype  oldtype,
		hypre_MPI_Datatype *newtype )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Type_contiguous((hypre_int)count, oldtype, newtype);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Type_vector( HYPRE_Int           count,
		HYPRE_Int           blocklength,
		HYPRE_Int           stride,
		hypre_MPI_Datatype  oldtype,
		hypre_MPI_Datatype *newtype )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Type_vector((hypre_int)count, (hypre_int)blocklength,
			(hypre_int)stride, oldtype, newtype);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Type_hvector( HYPRE_Int           count,
		HYPRE_Int           blocklength,
		hypre_MPI_Aint      stride,
		hypre_MPI_Datatype  oldtype,
		hypre_MPI_Datatype *newtype )
{
	start_time();
#if MPI_VERSION > 1
	HYPRE_Int ierr = (HYPRE_Int) MPI_Type_create_hvector((hypre_int)count, (hypre_int)blocklength,
			stride, oldtype, newtype);
#else
	HYPRE_Int ierr = (HYPRE_Int) MPI_Type_hvector((hypre_int)count, (hypre_int)blocklength,
	stride, oldtype, newtype);
#endif
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Type_struct( HYPRE_Int           count,
		HYPRE_Int          *array_of_blocklengths,
		hypre_MPI_Aint     *array_of_displacements,
		hypre_MPI_Datatype *array_of_types,
		hypre_MPI_Datatype *newtype )
{
	start_time();
	hypre_int *mpi_array_of_blocklengths;
	HYPRE_Int  i;
	HYPRE_Int  ierr;

	mpi_array_of_blocklengths = hypre_TAlloc(hypre_int,  count, HYPRE_MEMORY_HOST);
	for (i = 0; i < count; i++)
	{
		mpi_array_of_blocklengths[i] = (hypre_int) array_of_blocklengths[i];
	}

#if MPI_VERSION > 1
	ierr = (HYPRE_Int) MPI_Type_create_struct((hypre_int)count, mpi_array_of_blocklengths,
	array_of_displacements, array_of_types,
	newtype);
#else
ierr = (HYPRE_Int) MPI_Type_struct((hypre_int)count, mpi_array_of_blocklengths,
	array_of_displacements, array_of_types,
	newtype);
#endif

hypre_TFree(mpi_array_of_blocklengths, HYPRE_MEMORY_HOST);

	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Type_commit( hypre_MPI_Datatype *datatype )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Type_commit(datatype);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Type_free( hypre_MPI_Datatype *datatype )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Type_free(datatype);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Op_free( hypre_MPI_Op *op )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Op_free(op);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Op_create( hypre_MPI_User_function *function, hypre_int commute, hypre_MPI_Op *op )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Op_create(function, commute, op);
	end_time();
	return ierr;
}

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
HYPRE_Int
hypre_MPI_Comm_split_type( hypre_MPI_Comm comm, HYPRE_Int split_type, HYPRE_Int key, hypre_MPI_Info info, hypre_MPI_Comm *newcomm )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Comm_split_type(comm, split_type, key, info, newcomm );
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Info_create( hypre_MPI_Info *info )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Info_create(info);
	end_time();
	return ierr;
}

HYPRE_Int
hypre_MPI_Info_free( hypre_MPI_Info *info )
{
	start_time();
	HYPRE_Int ierr = (HYPRE_Int) MPI_Info_free(info);
	end_time();
	return ierr;
}
#endif

#endif
