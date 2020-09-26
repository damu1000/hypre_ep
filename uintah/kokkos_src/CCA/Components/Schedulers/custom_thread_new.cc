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
#include <Kokkos_Core.hpp>

int l1, l2;


thread_local int cust_g_team_id=-1, cust_g_thread_id=0;


int get_custom_team_id()
{
	return cust_g_team_id;
}

int get_custom_thread_id()
{
	return omp_get_thread_num();
}

int get_team_size()
{
	return l2;
}

int g_rank_temp;
void custom_parallel_for(int s, int e, std::function<void(int)> f)
{

#pragma omp parallel num_threads(l2)
	{
		cust_g_thread_id = omp_get_thread_num();

#pragma omp for
	for(int i=s; i<e; i++)
		f(i);

	}
}


void cparallel_for(int b, int e, void(*f)(int))
{
#pragma omp parallel num_threads(l2)
	{
		cust_g_thread_id = omp_get_thread_num();

#pragma omp for
		for(int i=b; i<e; i++)
			f(i);
	}
}

int affinity_set = 0;
void SetAffinity1( int proc_unit)
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

void custom_partition_master(int num_partitions, int partition_size, std::function<void(int)> f)
{
	l1 = num_partitions-1;
	l2 = partition_size;

	int total_threads = l1*l2;

	if(affinity_set==0){
		affinity_set=1;
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);


#pragma omp parallel num_threads(total_threads+1)
		{
			int thread_id = omp_get_thread_num();
			if(thread_id < total_threads)
				SetAffinity1(rank*total_threads + thread_id);
		}

	}

	omp_set_num_threads(total_threads+1);

#pragma omp parallel num_threads(num_partitions)
    {
    	int new_partition_size = partition_size;
    	cust_g_team_id = omp_get_thread_num();

#ifdef FUNNELED_COMM
  		//the last partition thread will be used for comm, so no need of worker threads
    	if(cust_g_team_id == num_partitions-1){
    		new_partition_size = 1;
    		cust_g_team_id = -1;
    	}

#endif
    	omp_set_num_threads(new_partition_size);

    	f( cust_g_team_id );
    }
}



















void ccustom_partition_master(int b, int e, void(*f)(int))
{
	printf("deprecated %s %d\n", __FILE__, __LINE__);
	exit(1);
	//custom_partition_master(b, e, f);
}


void SetAffinity( int proc_unit)
{
	printf("deprecated %s %d\n", __FILE__, __LINE__);
	exit(1);
}


inline void master_work()
{
	printf("deprecated %s %d\n", __FILE__, __LINE__);
	exit(1);
}
void team_master(int team_id, int thread_id)
{
	printf("deprecated %s %d\n", __FILE__, __LINE__);
	exit(1);
}


void team_worker(int team_id, int thread_id)
{
	printf("deprecated %s %d\n", __FILE__, __LINE__);
	exit(1);
}


void thread_init(int num_partitions, int threads_per_partition, int *affinity, int g_nodal_rank)
{
	printf("deprecated %s %d\n", __FILE__, __LINE__);
	exit(1);
}

void wait_for_init(int p, int t, int *affinity, int g_nodal_rank)
{

	printf("deprecated %s %d\n", __FILE__, __LINE__);
	exit(1);
}

void destroy()
{
	printf("deprecated %s %d\n", __FILE__, __LINE__);
	exit(1);
}
