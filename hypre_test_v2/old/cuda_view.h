//---------------------------------------------CUDA-----------------------------------------------------------
#define CUDA_BLOCK_SIZE 256

#define CUDA_CALL( call )               \
{                                       \
	if ( cudaSuccess != call ){         \
		printf("CUDA Error at %s %d: %s %s\n", __FILE__, __LINE__,cudaGetErrorName( call ),  cudaGetErrorString( call ) );  \
		exit(1);						\
	}									\
}

template <typename T>
struct cudaView{
	T *val{nullptr};
	int N;

	cudaView(int n){
		N = n;
		CUDA_CALL(cudaMalloc(&val, N*sizeof(T)));
	}

	~cudaView(){
		//CUDA_CALL(cudaFree(val));
	}

	inline __host__ __device__ T& operator()(int i) const{return val[i];}

	inline T *data() {return val;}

	inline int getN() {return N;}
};

template<typename T>
struct HostMirror{
	T *val;
	int N;

	template<typename cudaV>
	HostMirror(cudaV A){
		val = new T[A.getN()];
		N = A.getN();
		CUDA_CALL(cudaMemcpy(val, A.data(), sizeof(T)*N, cudaMemcpyDeviceToHost));
	}
	~HostMirror(){
		//delete []val;
	}

	inline T& operator()(int i) {return val[i];}

	inline T *data() {return val;}

	inline int getN() {return N;}
};



template<typename functor>
__global__ void cuda_kernel(int N, functor f){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < N)
		f(tid);
}

template<typename functor>
void parallel_for(int N, functor f){
	if (N == 0) return;
	int blocks = (N + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
	cuda_kernel<<<blocks, CUDA_BLOCK_SIZE>>>(N, f);
}
//-----------------------------------------------CUDA---------------------------------------------------------
/*
//add flags: -finstrument-functions -Wno-builtin-declaration-mismatch -rdynamic

#include <sys/time.h>
#include <string.h>
#include <execinfo.h>
#include <unordered_map>
#include <stack>
#include <dlfcn.h>

struct function_details{
	function_details():exe_time(0), parent(NULL), num_calls(0), inst_overhead{0}{}
	double exe_time;
	void *parent;
	unsigned long num_calls;
	double inst_overhead;
};
thread_local std::unordered_map<void *, function_details> mymap;
thread_local std::stack<struct timeval> mytime;
thread_local int __inst_g_active=0;
//thread_local double __inst_g_overhead=0.0;
thread_local double __inst_g_time_overhead, __inst_g_push_overhead, __inst_g_func_call_overhead, __inst_g_calc_overhead, __inst_g_cond_overhead;

int inst_add(void *this_fun) __attribute__((no_instrument_function));
void __cyg_profile_func_enter(void *this_fun, void *parent) __attribute__((no_instrument_function));
void __cyg_profile_func_exit(void *this_fun, void *parent) __attribute__((no_instrument_function));

void __attribute__((noinline)) inst_overhead(){}

//__attribute__((optimize("O0")))
inline void  __attribute__((no_instrument_function)) inst_init() {
	double N=100000000;
	struct timeval  tv1, tv2;

	//gettimeofday overhead
	gettimeofday(&tv1, NULL);
	for(int i=0; i<N; i++)
		gettimeofday(&tv2, NULL);
	gettimeofday(&tv2, NULL);
	double time_ms = (double) (tv2.tv_usec - tv1.tv_usec) / 1000.0 + (double) (tv2.tv_sec - tv1.tv_sec)*1000.0;
	__inst_g_time_overhead = time_ms / N;

	//stack push overhead
	std::stack<struct timeval> time_test;
	gettimeofday(&tv1, NULL);
	for(int i=0; i<N; i++)
		time_test.push(tv1);
	gettimeofday(&tv2, NULL);
	time_ms = (double) (tv2.tv_usec - tv1.tv_usec) / 1000.0 + (double) (tv2.tv_sec - tv1.tv_sec)*1000.0;
	__inst_g_push_overhead = time_ms / N;

	//function call overhead
	gettimeofday(&tv1, NULL);
	for(int i=0; i<N; i++)
		inst_overhead();
	gettimeofday(&tv2, NULL);
	time_ms = (double) (tv2.tv_usec - tv1.tv_usec) / 1000.0 + (double) (tv2.tv_sec - tv1.tv_sec)*1000.0;
	__inst_g_func_call_overhead = time_ms / N;

	//calc overhead
	function_details *fd, *par;
	int fdsize=1;
	const char* fdsize_char = getenv("FDSIZE");
	if(fdsize_char)	 fdsize=atoi(fdsize_char);
	fd = new function_details[fdsize];
	par = new function_details[fdsize];
	gettimeofday(&tv1, NULL);
	for(int i=0; i<N; i++){
		fd->inst_overhead +=  __inst_g_time_overhead + __inst_g_push_overhead + 18*__inst_g_cond_overhead;
		par->inst_overhead += fd->inst_overhead + time_ms +  __inst_g_time_overhead + __inst_g_calc_overhead + 2*__inst_g_func_call_overhead;
		fd->exe_time -= fd->inst_overhead;
		fd->inst_overhead = 0;
		__inst_g_active = i;
	}
	gettimeofday(&tv2, NULL);
	time_ms = (double) (tv2.tv_usec - tv1.tv_usec) / 1000.0 + (double) (tv2.tv_sec - tv1.tv_sec)*1000.0;
	__inst_g_calc_overhead = time_ms / N;
	delete []fd;
	delete []par;
	__inst_g_active = 0;


	gettimeofday(&tv1, NULL);
	for(int i=0; i<N; i++){
		if(__inst_g_active)
			__inst_g_active = 0;
		else
			__inst_g_active = 1;
	}
	gettimeofday(&tv2, NULL);
	time_ms = (double) (tv2.tv_usec - tv1.tv_usec) / 1000.0 + (double) (tv2.tv_sec - tv1.tv_sec)*1000.0;
	__inst_g_cond_overhead = time_ms / N;

	printf(" ## overheads - gettimeofday: %f, stack push: %f, function call: %f, calc: %f, cond: %f ##\n", \
			__inst_g_time_overhead, __inst_g_push_overhead, __inst_g_func_call_overhead, __inst_g_calc_overhead, __inst_g_cond_overhead);

	FILE *fptr = fopen("/dev/null", "w");
	fprintf(fptr, "%f %f %f %d %lu", fd->inst_overhead, par->inst_overhead, fd->exe_time, __inst_g_active, time_test.size());
	fclose(fptr);
}

inline void __attribute__((no_instrument_function)) inst_start() {
	__inst_g_active=1;
}

inline void __attribute__((no_instrument_function)) inst_stop() {
	__inst_g_active=0;
}

inline void __attribute__((no_instrument_function)) inst_report(int rank) {

	std::string filename = "inst." + std::to_string(rank) + ".txt";
	FILE *fptr = fopen(filename.data(), "w");
	fprintf(fptr, "exe_time(ms) num_calls function address parent par_address\n");
//	printf("exe_time(ms) num_calls function address parent par_address\n");
	for(auto iter = mymap.begin(); iter != mymap.end(); ++iter){
		auto fun = iter->first;
		auto exe_time = iter->second.exe_time;
		auto parent = iter->second.parent;
		auto num_calls = iter->second.num_calls;
		auto inst_overhead = iter->second.inst_overhead;
		char ** fun_name =  backtrace_symbols(&fun, 1);
		char ** par_name =  backtrace_symbols(&parent, 1);
		fprintf(fptr, "%f %lu %s %s\n", exe_time, num_calls, fun_name[0], par_name[0]);
//		printf("%f %lu %s %s\n", exe_time, num_calls, fun_name[0], par_name[0]);
		free(fun_name);
		free(par_name);
	}

	fclose(fptr);
	mymap.clear();
}

//int cond_count = 0;
//int start=0;
void __cyg_profile_func_enter(void *this_fun, void *parent) {

	if(__inst_g_active){	//only main thread will find __inst_g_active set. Because its a thread_local variable.
		__inst_g_active = 0;	//avoid recursive calling
//		start = 1;
//		cond_count = 0;
		struct timeval  tv1, tv2;
		gettimeofday(&tv1, NULL);

		mytime.push(tv1);

		gettimeofday(&tv2, NULL);

		//double time_ms = (double) (tv2.tv_usec - tv1.tv_usec) / 1000.0 + (double) (tv2.tv_sec - tv1.tv_sec)*1000.0;
		//__inst_g_overhead += time_ms;
//		start=0;
		__inst_g_active = 1;	//avoid recursive calling
	}

//	if(start)
//		cond_count++;

}


void __cyg_profile_func_exit(void *this_fun, void *parent) {

	if(__inst_g_active){	//only main thread will find __inst_g_active set. Because its a thread_local variable.
		__inst_g_active = 0;	//avoid recursive calling
		//stop timer and update index values of exe_time
		struct timeval  tv1, tv2, tvstart;
		gettimeofday(&tv1, NULL);

		tvstart = mytime.top();
		mytime.pop();
		double time_ms = (double) (tv1.tv_usec - tvstart.tv_usec) / 1000.0 + (double) (tv1.tv_sec - tvstart.tv_sec)*1000.0;
		Dl_info info;
		dladdr(parent, &info);
		function_details *fd = &mymap[this_fun], *par= &mymap[info.dli_saddr];
		fd->exe_time += time_ms;
		fd->parent = parent;	//this will always show the last parent. Will overwrite earlier calls
		fd->num_calls++;

		//asm volatile("" ::: "memory");

		gettimeofday(&tv2, NULL);

		time_ms = (double) (tv2.tv_usec - tv1.tv_usec) / 1000.0 + (double) (tv2.tv_sec - tv1.tv_sec)*1000.0;

		//the overhead becomes significant if a small function is invoked for 100000s of times.
		//So compute overhead of the current call(time_ms), add it to total self overhead (fd->inst_overhead)
		//add self overhead to parent's overhead. This way each function will have sum of overheads of all of its children + self.
		//subtract own overhead from exe_time and reset own overhead.
		fd->inst_overhead +=  __inst_g_time_overhead + __inst_g_push_overhead + 18*__inst_g_cond_overhead;
		par->inst_overhead += fd->inst_overhead + time_ms +  __inst_g_time_overhead + __inst_g_calc_overhead + 2*__inst_g_func_call_overhead;
		fd->exe_time -= fd->inst_overhead;
		fd->inst_overhead = 0;
		__inst_g_active = 1;
//		printf("%d\n", cond_count);

	}
//	if(start)
//		cond_count++;

}

*/
