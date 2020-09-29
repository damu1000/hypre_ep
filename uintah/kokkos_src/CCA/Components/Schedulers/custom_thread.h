#ifndef custom_thread_h
#define custom_thread_h

//#define FUNNELED_COMM


#ifdef __cplusplus /* If this is a C++ compiler, use C linkage */

#include<functional>

void custom_partition_master(int b, int e, std::function<void(int)> f);
void custom_parallel_for(int b, int e, std::function<void(int)> f, int active_threads);

extern "C++" {
#endif

extern thread_local int cust_g_team_id, cust_g_thread_id;
extern int g_rank_temp;

int get_custom_team_id();
int get_custom_thread_id();
int get_team_size();
//int l1, l2;
//int patch_dim, number_of_patches, N;//, max_vec_rank;

//void cpartition_master(int b, int e, void(*f)(int));
//void cparallel_for(int b, int e, void(*f)(int));
void thread_init(int num_partitions, int threads_per_partition, int *affinity, int g_nodal_rank);
void destroy();
void wait_for_init(int p, int t, int *affinity, int g_nodal_rank);

#ifdef __cplusplus /* If this is a C++ compiler, use C linkage */
}
#endif

#endif
