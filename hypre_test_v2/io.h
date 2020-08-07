#ifndef IO_H
#define IO_H
//#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <cstring>
#include <omp.h>

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
				printf("Error at node %d -> correct: %f \t computed:%f \t diff: %f\n", i, correct[i], AM[i], fabs(AM[i] - correct[i]));
				exit(1);
			}
		}

		printf("output matched!!\n");

		delete[] correct;
	}
}

#endif	// EXPORT_H
