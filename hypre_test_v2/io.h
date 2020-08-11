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
void inline write_to_file(ViewDouble A, size_t size){
	//typename ViewDouble::HostMirror AM = Kokkos::create_mirror_view(A);
	//Kokkos::deep_copy (AM, A);
	ViewDouble AM = A;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  string filename = "output/output.txt." + to_string(rank);

	ofstream out(filename, ios::out | ios::binary);
	if(!out) {
		cout << "Error: Cannot open file.\n";
		exit(1);
	}

	out.write((char *) AM, sizeof(double) * size);

	out.close();
}

template<class ViewDouble>
void inline verifyX(ViewDouble A, size_t size){
	if(verify){
		//typename ViewDouble::HostMirror AM = Kokkos::create_mirror_view(A);
		//Kokkos::deep_copy (AM, A);
		ViewDouble AM = A;

		double *correct = new double[size];
  	int rank;
	  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		int newrank = rank * omp_get_num_threads() + omp_get_thread_num();
    string filename = "output/output.txt." + to_string(newrank);

		ifstream in(filename, ios::in | ios::binary);
		if(!in) {
			verify=0;
			//cout << "Cannot open output.bin. returning without verification.\n";
			return;
		}
		in.read((char *) correct, sizeof(double) * size);
		in.close();

		//compare
#pragma omp parallel for
		for(int i=0; i<size; i++){
			if(fabs(AM[i] - correct[i]) > 1e-9 ){
				printf("Error at node %d -> correct: %f \t computed:%f \t diff: %f in file %s\n", i, correct[i], AM[i], fabs(AM[i] - correct[i]), filename.c_str());
				exit(1);
			}
		}

		printf("output matched!!\n");

		delete[] correct;
	}
}


struct xmlInput{
	int patch_size{-1};	// patch size
	int xpatches{-1};	// number of patches in x dimension
	int ypatches{-1};	// number of patches in y dimension
	int zpatches{-1};	// number of patches in z dimension
	std::string patch_assignment_scheme{"linear"}; 	//scheme to assign patches. Valid values are "linear" or "cubical". xthreads, ythreads, zthreads arguments are required for cubical scheme. Default value: linear
	int xthreads{-1}; // number of threads in x dimension (threads will be arranged in 3d grid)
	int ythreads{-1}; // number of threads in y dimension (threads will be arranged in 3d grid)
	int zthreads{-1}; // number of threads in z dimension (threads will be arranged in 3d grid)

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
    	   }else if(name == "patch_assignment_scheme"){
    		   input.patch_assignment_scheme = val;
    	   }else if(name == "xthreads"){
    		   input.xthreads = atoi(val.c_str());
    	   }else if(name == "ythreads"){
    		   input.ythreads = atoi(val.c_str());
    	   }else if(name == "zthreads"){
    		   input.zthreads = atoi(val.c_str());
    	   }
    	   else{
    		   printf("input element %s is not supported\n", cur->name);
    		   exit(1);
    	   }
       }
   }

   //bulletproofing
   if(input.patch_size==-1 || input.xpatches==-1 || input.ypatches==-1 || input.zpatches==-1){
	   printf("Please enter patch_size, xpatches, ypatches and zpatches in the input file\n"); exit(1);
   }

   if(input.patch_assignment_scheme=="cubical"){
	  if(input.xthreads==-1 || input.ythreads==-1 || input.zthreads==-1){
	   printf("Please enter patch_size, xthreads, ythreads and zthreads in the input file for cubical patch_assignment_scheme\n"); exit(1);
	  }
	  if(input.xpatches % input.xthreads != 0 || input.ypatches % input.ythreads != 0 || input.zpatches % input.zthreads != 0){
		 printf("The number of patches in a dimension should be divisible by the number of threads in that dimension\n"); exit(1);
	  }
   }

   if(rank == 0){
	   printf("Running with inputs:\n");
	   printf("patch size: %d\n", input.patch_size);
	   printf("number of patches : %d, %d, %d\n", input.xpatches, input.ypatches, input.zpatches);
	   printf("patch assignment scheme: %s\n", input.patch_assignment_scheme.c_str());



   }

   xmlFreeDoc(doc);
   xmlCleanupParser();

   return input;
}


#endif	// EXPORT_H
