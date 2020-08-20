#ifndef IO_H
#define IO_H
//#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <cstring>
#include <omp.h>

#include <stdio.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <unistd.h>
#include <stdlib.h>

using namespace std;

static int verify=1;

template<class ViewDouble>
void inline write_to_file(ViewDouble A, size_t size, int patches_per_rank){
	//typename ViewDouble::HostMirror AM = Kokkos::create_mirror_view(A);
	//Kokkos::deep_copy (AM, A);
	ViewDouble AM = A;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int patch_start = rank * patches_per_rank;

	for(int i=0; i<patches_per_rank; i++){
		int patchid = patch_start + i;
		string filename = "output/output.txt." + to_string(patchid);

		ofstream out(filename, ios::out | ios::binary);
		if(!out) {
			cout << "Error: Cannot open file.\n";
			exit(1);
		}

		out.write((char *) (&AM[i*size]), sizeof(double) * size);

		out.close();
	}
}

template<class ViewDouble>
void inline verifyX(ViewDouble A, size_t size, std::vector<int>& my_patches){ //read per rank as pre ep in the context of this function
	if(verify){
		//typename ViewDouble::HostMirror AM = Kokkos::create_mirror_view(A);
		//Kokkos::deep_copy (AM, A);
		ViewDouble AM = A;

		double *correct = new double[size];

		for(int i=0; i<my_patches.size(); i++){
			int patch = my_patches[i];
			string filename = "output/output.txt." + to_string(patch);

			ifstream in(filename, ios::in | ios::binary);
			if(!in) {
				verify=0;
				//cout << "Cannot open output.bin. returning without verification.\n";
				return;
			}
			in.read((char *) correct, sizeof(double) * size);
			in.close();

			//compare

			for(int j=0; j<size; j++){
				if(fabs(AM[i*size + j] - correct[j]) > 1e-9 ){
					printf("Error at node %d -> correct: %f \t computed:%f \t diff: %f in file %s\n", j, correct[j], AM[i*size + j], fabs(AM[i*size + j] - correct[j]), filename.c_str());
					exit(1);
				}
			}
			printf("output matched!!\n");
		}

		delete[] correct;
	}
}


struct xmlInput{
	int patch_size{-1};	// patch size
	int xpatches{-1};	// number of patches in x dimension
	int ypatches{-1};	// number of patches in y dimension
	int zpatches{-1};	// number of patches in z dimension
//	std::string patch_assignment_scheme{"linear"}; 	//scheme to assign patches. Valid values are "linear" or "cubical". xthreads, ythreads, zthreads arguments are required for cubical scheme. Default value: linear
	int xthreads{1}; // number of threads in x dimension (threads will be arranged in 3d grid)
	int ythreads{1}; // number of threads in y dimension (threads will be arranged in 3d grid)
	int zthreads{1}; // number of threads in z dimension (threads will be arranged in 3d grid)
	int verify{0};
	int timesteps{10};
	int output_interval{10};
};


xmlInput parseInput(const char * file, int rank){

   xmlInput input;
   xmlDoc *doc = xmlReadFile(file, NULL, 0);

   if (doc == NULL) {
	   printf("error: could not parse input file %s\n", file);
	   exit(1);
   }

   xmlNode *root_element = xmlDocGetRootElement(doc);

   for (xmlNode *cur = root_element->children; cur; cur = cur->next) {
       if (cur->type == XML_ELEMENT_NODE) {
    	   std::string name = std::string(reinterpret_cast<const char*>( cur->name ));
    	   std::string val =  std::string(reinterpret_cast<const char*>( xmlNodeGetContent(cur) ));

    	   if(name == "patch_size"){
    		   input.patch_size = atoi(val.c_str());
    	   }else if(name == "xpatches"){
    		   input.xpatches = atoi(val.c_str());
    	   }else if(name == "ypatches"){
    		   input.ypatches = atoi(val.c_str());
    	   }else if(name == "zpatches"){
    		   input.zpatches = atoi(val.c_str());
    	   }/*else if(name == "patch_assignment_scheme"){
    		   input.patch_assignment_scheme = val;
    	   }*/else if(name == "xthreads"){
    		   input.xthreads = atoi(val.c_str());
    	   }else if(name == "ythreads"){
    		   input.ythreads = atoi(val.c_str());
    	   }else if(name == "zthreads"){
    		   input.zthreads = atoi(val.c_str());
    	   }else if(name == "verify"){
    		   input.verify = atoi(val.c_str());
    	   }else if(name == "timesteps"){
    		   input.timesteps = atoi(val.c_str());
    	   }else if(name == "output_interval"){
    		   input.output_interval = atoi(val.c_str());
    	   }
    	   else{
    		   printf("input element %s is not supported at %s:%d\n", cur->name, __FILE__, __LINE__);
    		   exit(1);
    	   }
       }
   }

   //bulletproofing
   if(input.patch_size==-1 || input.xpatches==-1 || input.ypatches==-1 || input.zpatches==-1){
	   printf("Please enter patch_size, xpatches, ypatches and zpatches in the input file at %s:%d\n", __FILE__, __LINE__); exit(1);
   }

   if(input.xpatches % input.xthreads != 0 || input.ypatches % input.ythreads != 0 || input.zpatches % input.zthreads != 0){
	   printf("The number of patches in a dimension should be divisible by the number of threads in that dimension at %s:%d\n", __FILE__, __LINE__); exit(1);
   }

//   if(input.patch_assignment_scheme=="linear"){
//	   input.xthreads = omp_get_max_threads();
//   }
//   else if(input.patch_assignment_scheme=="cubical"){
////	  if(input.xthreads==-1 || input.ythreads==-1 || input.zthreads==-1){
////	   printf("Please enter patch_size, xthreads, ythreads and zthreads in the input file for cubical patch_assignment_scheme at %s:%d\n", __FILE__, __LINE__); exit(1);
////	  }
//	  if(input.xpatches % input.xthreads != 0 || input.ypatches % input.ythreads != 0 || input.zpatches % input.zthreads != 0){
//		 printf("The number of patches in a dimension should be divisible by the number of threads in that dimension at %s:%d\n", __FILE__, __LINE__); exit(1);
//	  }
//   }

   if(rank == 0){
	   printf("Running with inputs:\n");
	   printf("patch size: %d\n", input.patch_size);
	   printf("number of patches : %d, %d, %d\n", input.xpatches, input.ypatches, input.zpatches);
	   //printf("patch assignment scheme: %s\n", input.patch_assignment_scheme.c_str());
   }

   xmlFreeDoc(doc);
   xmlCleanupParser();

   return input;
}


#endif	// EXPORT_H
