#ifndef custom_thread_h
#define custom_thread_h

//#define USE_FUNNELLED_COMM

//#define help_neighbor


#include<cmath>
#include<vector>
#include<omp.h>
#include<stdio.h>
#include<chrono>
#include <ctime>
#include<iostream>
#include<string.h>
#include<functional>
#include<atomic>
#include<mutex>
#include<condition_variable>
#include<sched.h>
#include<mpi.h>
#include<unistd.h>
#include<omp.h>

void custom_partition_master(int b, int e, std::function<void(int)> f);
void custom_parallel_for(int b, int e, std::function<void(int)> f);

#ifdef __cplusplus /* If this is a C++ compiler, use C linkage */
extern "C" {
#endif

extern thread_local int cust_g_team_id, cust_g_thread_id;
extern int g_rank_temp;

int get_custom_team_id();
int get_custom_thread_id();
int get_team_size();
//int l1, l2;
//int patch_dim, number_of_patches, N;//, max_vec_rank;

void cpartition_master(int b, int e, void(*f)(int));
void cparallel_for(int b, int e, void(*f)(int));
void thread_init(int num_partitions, int threads_per_partition, int *affinity, int g_nodal_rank);
void destroy();
void wait_for_init(int p, int t, int *affinity, int g_nodal_rank);

#ifdef __cplusplus /* If this is a C++ compiler, use C linkage */
}
#endif




double loop_time = 0.0;
int l1, l2;



volatile int spin=1;

std::atomic<int> g_threads_init;

#ifdef USE_FUNNELLED_COMM
thread_local int cust_g_team_id=-1, cust_g_thread_id=-1;
#else
thread_local int cust_g_team_id=0, cust_g_thread_id=0;
#endif

int get_custom_team_id()
{
	return cust_g_team_id;
}

int get_custom_thread_id()
{
	return cust_g_thread_id;
}

int get_team_size()
{
	return l2;
}

//across teams
std::function<void(int)> g_function;
std::atomic<int> g_begin;	//beginning of team loop.
volatile int g_end;	//beginning of team loop.
std::atomic<int> g_completed;
volatile int g_num_of_calls{0};


typedef struct thread_team
{
	std::function<void(int)> m_fun;	//size 32 bytes
	std::atomic<int> m_begin;	//beginning of team loop.	//size 4 bytes
	volatile int m_end;	//beginning of team loop.	//size 4 bytes
	std::atomic<int> m_completed{0};	//size 4 bytes
	volatile int m_num_of_calls{0};		//size 4 bytes
	char dummy[16];

} thread_team;


thread_team * g_team;

//#define __USE_GNU
void SetAffinity( int proc_unit)
{
//	printf("affinity rank, thread id, core: %d\t%d\t%d\n", rank, omp_id , proc_unit);

#ifndef __APPLE__
  //disable affinity on OSX since sched_setaffinity() is not available in OSX API
  cpu_set_t mask;
  unsigned int len = sizeof(mask);
  CPU_ZERO(&mask);
  CPU_SET(proc_unit, &mask);
  sched_setaffinity(0, len, &mask);
#endif
}


inline void master_work()
{
	while(g_begin < g_end)
	{
		int b = g_begin;
		int b_next = b + 1;
		if(g_begin.compare_exchange_strong(b, b_next))
		{
			if(b < g_end)
			{
				g_function(b);
			}
		}
	}
	g_completed++;
}
void team_master(int team_id, int thread_id)
{
	int num_of_calls=0;
	g_threads_init++;
	//printf("master team: %d, thread %d\n", team_id, thread_id);
	while(spin)
	{
		//printf("master waiting %d\n", team_id);

		while(g_function==NULL && spin==1){asm("pause");}

		//printf("master got fun %d\n", team_id);

		 if(num_of_calls < g_num_of_calls)
		 {
			// printf("master working\n");
			 master_work();
			 num_of_calls++;
		 }

	}
	//printf("team master exited %d\n", team_id);
}


void team_worker(int team_id, int thread_id)
{
	int num_of_calls=0;
	thread_team *my_team = &g_team[cust_g_team_id];
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	g_threads_init++;
	//printf("worker team: %d, thread %d\n", team_id, thread_id);
	while(spin)
	{
		//printf("worker waiting %d-%d\n", team_id, thread_id);

				while(!my_team->m_fun && spin==1){
					asm("pause");
				}
				if(num_of_calls < g_team[cust_g_team_id].m_num_of_calls)
				{
					num_of_calls++;
					//printf("%d worker %d-%d\n",rank, team_id, thread_id);
					int e = my_team->m_end;
					int chunk = e - my_team->m_begin;
					chunk = chunk / l2 / 2;
					chunk = std::max(chunk, 1);
					int b;
					while((b = my_team->m_begin.fetch_add(chunk, std::memory_order_seq_cst))< e)
					{
						int b_next = std::min(e, b+chunk);
						for(; b<b_next; b++)
						{
							my_team->m_fun(b);
						}
					}
					my_team->m_completed++;
				}


	#ifdef help_neighbor
				//help neighbor is not working with m_completed < l2 logic

				int t;
				for(t=(cust_g_team_id + 1)%l1; t != cust_g_team_id; t=(t+1)%l1)//iterate over all teams
				{

					if(g_team[t].m_begin < g_team[t].m_end &&
						  g_team[t].m_completed ==0 &&
						  g_team[t].m_fun!=NULL  ) //g_team[t].m_completed < 0 means work is in progress for the team. help!
					{
						//printf("%d-%d helping %d\n", team_id, thread_id, t);
						g_team[t].m_completed--;

						int e = g_team[t].m_end;
						int chunk = e - g_team[t].m_begin;
						chunk = chunk / l2 / 2;
						chunk = std::max(chunk, 1);
						int b;
						while((b = g_team[t].m_begin.fetch_add(chunk, std::memory_order_seq_cst))< e)
						{
							int b_next = std::min(e, b+chunk);
							for(; b<b_next; b++)
							{
								g_team[t].m_fun(b);
							}
						}
						g_team[t].m_completed++;

					}

					if(num_of_calls < g_team[cust_g_team_id].m_num_of_calls || spin==0)	//continue own work if available
						break;
				}
	#endif


	}
	//printf("team worker exited %d\n", team_id);
}


void thread_init(int num_partitions, int threads_per_partition, int *affinity, int g_nodal_rank)
{
	spin=1;
	g_num_of_calls=0;

	l1 = num_partitions;
	l2 = threads_per_partition;
	g_threads_init = 1;
	//printf("thread_init: %d\n", omp_get_thread_num());
	g_team = new thread_team[l1];
#ifdef USE_FUNNELLED_COMM
	int num_threads = l1*l2, first_thread = 0;
#else
	int num_threads = l1*l2-1, first_thread = 1;
#endif
	omp_set_num_threads(num_threads);
	#pragma omp parallel for num_threads(num_threads) schedule(static,1)
	for(int omp_thread_id=first_thread; omp_thread_id<l1*l2; omp_thread_id++)
	{

		if(affinity)
			SetAffinity(affinity[g_nodal_rank * l1*l2 + omp_thread_id]);
		int team_id = omp_thread_id / l2;
		int thread_id = omp_thread_id % l2;
		cust_g_team_id = team_id;
		cust_g_thread_id = thread_id;

		//printf("%d threads %d %d on core %d\n",g_nodal_rank, team_id, thread_id, affinity[g_nodal_rank * l1*l2 + omp_thread_id]);

		if(thread_id==0)
			team_master(team_id, thread_id);
		else
			team_worker(team_id, thread_id);

	}
	//printf("thread_init returning \n");
}

void wait_for_init(int p, int t, int *affinity, int g_nodal_rank)
{

	while(g_threads_init < p*t){asm("pause");}	//wait till threads are initialized

#ifdef USE_FUNNELLED_COMM

	if(affinity)
	{
		SetAffinity(affinity[g_nodal_rank]);
		//printf("%d threads %d %d on core %d\n",g_nodal_rank, -1, -1, affinity[g_nodal_rank]);
	}
	//printf("init completed\n");
#else
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(affinity)
		SetAffinity(affinity[g_nodal_rank * p*t + 0]);

#endif
}

void destroy()
{
	spin = 0;
	g_threads_init=0;
	delete []g_team;
}

int g_rank_temp;
void custom_parallel_for(int s, int e, std::function<void(int)> f)
{
	thread_team *my_team = &g_team[cust_g_team_id];

	my_team->m_fun = f;
//	std::atomic_thread_fence(std::memory_order_seq_cst);
	my_team->m_end = e;
	my_team->m_begin = s;
	my_team->m_completed = 0;
	my_team->m_num_of_calls++;


	int chunk = e - s;
	chunk = chunk / l2 / 2;
	chunk = std::max(chunk, 1);


	int b;
	while((b = my_team->m_begin.fetch_add(chunk, std::memory_order_seq_cst))< e)
	{
		int b_next = std::min(e, b+chunk);
		for(; b<b_next; b++)
		{
			my_team->m_fun(b);
		}
	}

	my_team->m_completed++;

	while(my_team->m_completed.load(std::memory_order_seq_cst)<l2)	//wait for execution to complete
	{
		asm("pause");
		//printf("waiting %d %d %d\n", g_completed.load(std::memory_order_seq_cst), g_begin.load(std::memory_order_seq_cst), g_end);
	}

	if(my_team->m_fun)
		my_team->m_fun=NULL;
	//printf("%d - %d %d: completed custom_parallel_for %d / %d\n",
	//		g_rank_temp, cust_g_team_id, cust_g_thread_id, my_team->m_completed.load(std::memory_order_seq_cst), l2);
}

double parallel_time = 0.0;

void cparallel_for(int b, int e, void(*f)(int))
{
	//auto start = std::chrono::system_clock::now();
	custom_parallel_for(b, e, f);
	/*auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	parallel_time += elapsed_seconds.count();*/
}

void custom_partition_master(int b, int e, std::function<void(int)> f)
{
	g_function = f;
//	std::atomic_thread_fence(std::memory_order_seq_cst);
	g_completed = 0;
	g_end = e;
	g_begin = b;
	g_num_of_calls++;

	//printf("thread 0: posted work, contributing now\n");

	//auto start = std::chrono::system_clock::now();

	master_work();	//this thread is team 0, thread 0. it will do its part

	//auto end = std::chrono::system_clock::now();
	//std::chrono::duration<double> elapsed_seconds = end-start;
	//printf("custom_partition_master time: %f, parallel_for_time: %f\n", elapsed_seconds.count(), parallel_time);

#ifdef USE_FUNNELLED_COMM
	while(g_completed<l1+1)	//wait for execution to complete. Funneled comm will have 1 extra thread for comm. wait for partition. otherwise it exits before all partitions finish and program crashes
#else
	while(g_completed<l1)	//wait for execution to complete
#endif
	{
		asm("pause");
		//printf("waiting %d %d %d\n", g_completed.load(std::memory_order_seq_cst), g_begin.load(std::memory_order_seq_cst), g_end);
	}
	g_function=NULL;
	//printf("completed custom_partition_master %d\n",g_num_of_calls.load(std::memory_order_seq_cst));
}

void ccustom_partition_master(int b, int e, void(*f)(int))
{
	custom_partition_master(b, e, f);
}



template<typename Driver, typename Arg1, typename Arg2 >
void custom_thread_driver(int m_hypre_num_of_threads, int m_partition_size, Driver driver, Arg1 arg1, Arg2 arg2)
{
	int g_nodal_rank=-1, g_nodal_size=-1, g_global_rank=-1;
	MPI_Comm shmcomm;
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
	MPI_Comm_rank(shmcomm, &g_nodal_rank);
	MPI_Comm_size(shmcomm, &g_nodal_size);
	MPI_Comm_free(&shmcomm);

	const char* hypre_binding_str = getenv("HYPRE_BINDING"); //use diff env variable if it conflicts with OMP. but using same will be consistent.
	char temp_str[1024];
	const char s[2] = ",";
	char *token;
	int affinity[m_hypre_num_of_threads*m_partition_size*g_nodal_size];
#ifdef USE_FUNNELLED_COMM
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

#ifdef USE_FUNNELLED_COMM
	//read HYPRE_BINDING_COMM
	const char* hypre_binding_comm_str = getenv("HYPRE_BINDING_COMM");
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

			if(flag){	// main thread should call hypre_solve and worker should init further worker threads. This main thread to act as master thread for funneled comm
				//printf("main thread waiting for init\n");
				wait_for_init(m_hypre_num_of_threads, m_partition_size, comm_affinity, g_nodal_rank);
				//printf("calling Hypre solve \n");
				driver(arg1, arg2);
				destroy();
			}
			else{
				//printf("worker thread init\n");
				thread_init(m_hypre_num_of_threads, m_partition_size, affinity, g_nodal_rank);
			}

		}

#pragma omp section
		{
			int flag;
			MPI_Is_thread_main( &flag );
			if(flag){	// main thread should call hypre_solve and worker should init further worker threads. This main thread to act as master thread for funneled comm
				wait_for_init(m_hypre_num_of_threads, m_partition_size, comm_affinity, g_nodal_rank);
				driver(arg1, arg2);
				destroy();
			}
			else{
				thread_init(m_hypre_num_of_threads, m_partition_size, affinity, g_nodal_rank);
			}
		}
	}
	//printf("omp section end\n");
	omp_set_num_threads(curr_threads);
}


#endif
