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
#include <CCA/Components/Schedulers/custom_thread.h>

double loop_time = 0.0;
int l1, l2;



volatile int spin=1;

std::atomic<int> g_threads_init;

#ifdef FUNNELED_COMM
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
	volatile int m_active_threads{0};		//size 4 bytes
	char dummy[12];

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
					if(thread_id < my_team->m_active_threads){
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
}

void custom_partition_master(int num_partitions, int threads_per_partition, int *affinity, int g_nodal_rank, std::function<void(int)> f)
{
#ifdef FUNNELED_COMM
#error "FUNNELED_COMM not supported"
#endif
	spin=1;
	l1 = num_partitions;
	l2 = threads_per_partition;
	g_team = new thread_team[l1];
	int num_threads = l1*l2;
#pragma omp parallel num_threads(num_threads)
	{
		int omp_thread_id = omp_get_thread_num();
		if(affinity)
			SetAffinity(affinity[g_nodal_rank * l1*l2 + omp_thread_id]);
		int team_id = omp_thread_id / l2;
		int thread_id = omp_thread_id % l2;
		cust_g_team_id = team_id;
		cust_g_thread_id = thread_id;
		if(thread_id==0)
			f(team_id);
		else
			team_worker(team_id, thread_id);

		if(team_id==0){
			spin = 0;
			g_threads_init=0;
			delete []g_team;
		}
	}

}

void custom_parallel_for(int s, int e, std::function<void(int)> f, int active_threads)
{
	thread_team *my_team = &g_team[cust_g_team_id];
	my_team->m_active_threads = active_threads;
	my_team->m_fun = f;
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
}

