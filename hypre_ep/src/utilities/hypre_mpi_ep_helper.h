#ifndef hypre_MPI_EP_HELPER_H
#define hypre_MPI_EP_HELPER_H

#include<stdlib.h>
#include<stdio.h>
#include <mpi.h>
#include <stddef.h>
#include <atomic>

#define USE_INTER_THREAD_COMM
#define USE_MULTIPLE_COMMS
#define USE_ODD_EVEN_COMMS

/************************************************************************************************************************************

Global declaration needed for EP

************************************************************************************************************************************/

volatile int g_num_of_threads = 0;	//total number of threads.
//tag multipliers. assuming default values of 8,8,16 -> from LSB to MSB, 16 bits for tag, 8 bits for dest thread id, 8 bits for src thread id
int dest_multiplier = 0x00010000, src_multiplier=0x01000000, src_eliminate_and=0x00ffffff;
//int new_tag = tag == MPI_ANY_TAG ? MPI_ANY_TAG : (src_thread * 0x01000000) + (dest_thread * 0x00010000) + tag % 0x00010000; converted to
//int new_tag = tag == MPI_ANY_TAG ? MPI_ANY_TAG : (src_thread * src_multiplier) + (dest_thread * dest_multiplier) + tag % dest_multiplier;



#ifdef HYPRE_USING_MPI_EP

__thread int g_rank=0, g_real_rank=0;
__thread int g_thread_id=0;

//I know its a bad place. change later
#if defined(HYPRE_USING_CUDA) 
cudaStream_t *g_streams;
#endif //#if defined(HYPRE_USING_CUDA)


/************************************************************************************************************************************

intra-rank send-recv without MPI

************************************************************************************************************************************/
//This is hypre specific case where data is packed. As a result, ALWAYs only ONE message is sent from one source to one destination
//during one exchange (i.e. one wait_all). So assuming max possible number of messages is #threads x #threads and allocating the buffer
//accordingly. Also not doing any validations such as tags, message size etc. because those happen in MPI. So if MPI Only works,
//this also should ideally work as long as threads wait for each other to send, receive and complete the oprations.

#include<thread>

thread_local bool do_setup=true;

void ** g_send_buff;
void ** g_recv_buff;
int * g_size_buff; //receiver size
thread_local int recv_count=0, recv_done=0;

void irInsertMessage(const char* type, //hardcode as "send" or "recv". Used for debugging
		void               *buf,
		HYPRE_Int           count,
		hypre_MPI_Datatype  datatype,
		HYPRE_Int           tid,	//this is dest/src thread id
		HYPRE_Int           tag)
{
	if(type[0]=='r'){ //receive
		void **rbuf = &g_recv_buff[g_thread_id * g_num_of_threads];
		int *size= &g_size_buff[g_thread_id * g_num_of_threads];

		//try to progress
		for(int i=0; i<g_num_of_threads; i++){
			if(rbuf[i] != NULL){//data is received
				if(g_send_buff[i * g_num_of_threads + g_thread_id] != NULL){
					memcpy(rbuf[i], g_send_buff[i * g_num_of_threads + g_thread_id], size[i]);
					size[i] = 0;
					rbuf[i] = NULL;
					g_send_buff[i * g_num_of_threads + g_thread_id] = NULL;
					recv_done++;
				}
			}
		}

		if(g_send_buff[tid * g_num_of_threads + g_thread_id] != NULL){//copy at once if sender has sent data
			memcpy(buf, g_send_buff[tid * g_num_of_threads + g_thread_id], count);
			g_send_buff[tid * g_num_of_threads + g_thread_id] = NULL;
			recv_count++;
			recv_done++;
		}
		else{
			recv_count++;
			size[tid] = count;
			rbuf[tid] = buf; //tid is src
		}
	}
	else //send
		g_send_buff[g_thread_id * g_num_of_threads + tid] = buf; //tid is dest
}


int irWaitAll()
{
	void **rbuf = &g_recv_buff[g_thread_id * g_num_of_threads];
	int *size= &g_size_buff[g_thread_id * g_num_of_threads];
	while(recv_done < recv_count){
		for(int i=0; i<g_num_of_threads; i++){
			if(rbuf[i] != NULL){//data is received
				if(g_send_buff[i * g_num_of_threads + g_thread_id] != NULL)
				{
					memcpy(rbuf[i], g_send_buff[i * g_num_of_threads + g_thread_id], size[i]);
					size[i] = 0;
					rbuf[i] = NULL;
					g_send_buff[i * g_num_of_threads + g_thread_id] = NULL;
					recv_done++;
				}
			}
		}
	}

	int ircount = recv_count;
	recv_count=0;
	recv_done =0;

	for(int i=0; i<g_num_of_threads; i++){
		while(g_send_buff[g_num_of_threads * g_thread_id + i] != NULL) std::this_thread::yield();
	}

	return ircount;
}


/************************************************************************************************************************************

Multiple comms for EP

************************************************************************************************************************************/


#ifdef USE_MULTIPLE_COMMS
#include <unordered_map>
#include <vector>
#include <functional>
#define COMMS_PER_THREAD 28

MPI_Comm * g_comm_pool;

struct SrcDestKey{
	int src{-1},dest{-1}; //rank (EP)

	SrcDestKey(int s, int d):src(s), dest(d){}

  inline bool operator==(const SrcDestKey &a) const  { return (src == a.src && dest == a.dest); }

};

namespace std {	//needed for hash of SrcDestKey used by unordered_map
  template <>
  struct hash<SrcDestKey>
  {
    std::size_t operator()(const SrcDestKey& k) const {
      return ((std::hash<int>()(k.src)      ) ^
              (std::hash<int>()(k.dest) << 1) );
    }
  };
}

std::vector<std::unordered_map<SrcDestKey, MPI_Comm>> g_comm_map;
int *g_ep_superpatch_map;
int g_x_superpatches, g_y_superpatches, g_z_superpatches;
int g_x_ranks, g_y_ranks, g_z_ranks; //sspatches -> super super patches: super super patch is huge patch of all the patches from all the threads of real dest rank

//convert super patch ids from 1d to 3d
#define OneDtoThreeD(t, p, x_patches, y_patches, z_patches)	\
	t[2] = p / (y_patches*x_patches);						\
	t[1] = (p % (y_patches*x_patches)) / x_patches;			\
	t[0] = p % x_patches;

#endif //#ifdef USE_MULTIPLE_COMMS


void createCommMap(int *ep_superpatch, int *super_dims, int *rank_dims, int xthreads, int ythreads, int zthreads)
{
#ifdef USE_MULTIPLE_COMMS
	g_ep_superpatch_map = ep_superpatch;
	g_x_superpatches = super_dims[0], g_y_superpatches = super_dims[1], g_z_superpatches = super_dims[2];
	g_x_ranks = rank_dims[0],  g_y_ranks = rank_dims[1],  g_z_ranks = rank_dims[2];
#endif
}


inline MPI_Comm getComm(int src, int dest)
{
	MPI_Comm comm = MPI_COMM_WORLD;

#ifdef USE_MULTIPLE_COMMS
	int dest_thread = dest % g_num_of_threads;
	SrcDestKey key(src, dest);
	auto it = g_comm_map[g_thread_id].find(key);
	if(it != g_comm_map[g_thread_id].end())
		comm = it->second;
	else{
		//find superpatches corresponding to src and dest
		int src_patch = g_ep_superpatch_map[src], dest_patch = g_ep_superpatch_map[dest];
		int src3d[3], dest3d[3], dir[3], real_rank[3];

		//convert super patch ids from 1d to 3d
		OneDtoThreeD(src3d,  src_patch,  g_x_superpatches, g_y_superpatches, g_z_superpatches);
		OneDtoThreeD(dest3d, dest_patch, g_x_superpatches, g_y_superpatches, g_z_superpatches);

		//find comm direction = dest - src and compute comm_id
		for(int i=0; i<3; i++){
			dir[i] = dest3d[i]-src3d[i];
			if(dir[i] != 0) dir[i] = dir[i] / abs(dir[i]); //divide by abs value to make dir value 1 or -1
		}

		int comm_id = abs(dir[0]) + abs(dir[1]) * 3 + abs(dir[2]) * 9;

		//figure out even / odd and subtract 13 to offset for odd destinations
#ifdef USE_ODD_EVEN_COMMS
		int dest_real_rank = dest / g_num_of_threads;
		OneDtoThreeD(real_rank, dest_real_rank, g_x_ranks, g_y_ranks, g_z_ranks);
		if(real_rank[0]%2 == 1 || real_rank[1]%2 == 1 || real_rank[2]%2 == 1)
			comm_id = comm_id + 14; //subtract 13 to offset for odd destinations
#endif

		//pick comm from dest thread's queue with offset given by comm_id
		int dest_thread = dest % g_num_of_threads;
		comm =  g_comm_pool[dest_thread*COMMS_PER_THREAD + comm_id];
//		printf("%d added comm for src %d dest %d comm_id %d\n", g_rank, src, dest, comm_id);
		g_comm_map[g_thread_id][key] = comm;
	}
#endif //#ifdef USE_MULTIPLE_COMMS

	return comm;
}

/************************************************************************************************************************************

reduction helpers

************************************************************************************************************************************/

typedef struct MessageNode {
	int m_source, m_tag;	//store fake src and old tag (store values passed by hypre wo modification)
	MPI_Message m_mpi_message;
	MPI_Status m_mpi_status;
	volatile struct MessageNode * m_next;
} MessageNode;


volatile MessageNode **g_MessageHead=NULL, **g_MessageTail=NULL;
std::atomic<int> * g_MessageCount;
__thread int tl_prevMessageCount, tl_prevtag;
volatile std::atomic_flag *gLock;
volatile std::atomic_flag gProbeLock;


volatile void * volatile g_collective_sync_buff=NULL;
std::atomic<int> g_collective_sync_count;
std::atomic<int> g_collective_sync_copy_count;
std::atomic<int> wait_for_all_threads_to_exit;
//std::atomic<int> g_barrier1;
//std::atomic<int> g_barrier;


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



/************************************************************************************************************************************

Setup part for hypre EP. Should be at the end to know all the data types

************************************************************************************************************************************/



int (*hypre_get_thread_id)();		//function pointer to store function which will return thread id. C does not allow thread local storage. Hence pass a function
							//to return thread id - could be custom written function or could be library call such as omp_get_thread_num
							//WHOLE IMPLEMENTATION WILL FAIL IF THIS FUNCTION FAILS


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

#if defined(HYPRE_USING_CUDA) 
	g_streams = (cudaStream_t*) malloc(g_num_of_threads * sizeof(cudaStream_t));
	for (int i = 0; i < g_num_of_threads; i++) 
	        cudaStreamCreate(&g_streams[i]);
#endif //#if defined(HYPRE_USING_CUDA)



#ifdef USE_MULTIPLE_COMMS	
	g_comm_pool = new MPI_Comm[g_num_of_threads * COMMS_PER_THREAD];	//allocate 9 comms per thread
	for(int i=0; i<g_num_of_threads * COMMS_PER_THREAD; i++)
		MPI_Comm_dup( MPI_COMM_WORLD, &g_comm_pool[i] );

	g_comm_map.resize(g_num_of_threads);
#endif //#ifdef USE_MULTIPLE_COMMS


	g_send_buff = new void* [g_num_of_threads*g_num_of_threads];
	g_recv_buff = new void* [g_num_of_threads*g_num_of_threads];
	g_size_buff = new int[g_num_of_threads*g_num_of_threads]; //receiver size

	for(int i=0;i<g_num_of_threads*g_num_of_threads; i++){
		g_send_buff[i] = NULL;
		g_recv_buff[i] = NULL;
		g_size_buff[i] = 0;
	}

}


void hypre_init_thread()	//to be called by every thread
{
	MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
	g_real_rank = g_rank;
	g_thread_id = hypre_get_thread_id();
	g_rank = g_rank * g_num_of_threads + g_thread_id;
  //printf("hypre_init_thread - rank: %d, thread: %d\n", g_rank, g_thread_id);
	tl_prevMessageCount = 0;
	tl_prevtag = -1;
}

void hypre_destroy_thread()	//ideally should be called after every call to multithreaded Hypre solve to avoid any memory leaks. //call from master thread AFTER workers are joined
{
	int i;
	if(g_thread_id==0){
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
	#endif //#if defined(HYPRE_USING_CUDA)


#ifdef USE_MULTIPLE_COMMS	
	for(int i=0; i<g_num_of_threads * COMMS_PER_THREAD; i++)
		MPI_Comm_free( &g_comm_pool[i] );
	delete []g_comm_pool;
#endif //#ifdef USE_MULTIPLE_COMMS
	delete []g_send_buff;
	delete []g_recv_buff;
	delete []g_size_buff;

	}

}



/************************************************************************************************************************************

End of EP helper. Do not add anything here. Add before setup.

************************************************************************************************************************************/



#endif //#ifdef HYPRE_USING_MPI_EP



#endif //hypre_MPI_EP_HELPER_H








