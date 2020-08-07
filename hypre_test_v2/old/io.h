#ifndef IO_H
#define IO_H
//#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include "cuda_view.h"

using namespace std;

static int verify=1;


template<class ViewDouble>
void inline write_to_file(ViewDouble A, int rank){
	HostMirror<double> AM(A);

	ofstream out("output/output.bin"+ to_string(rank), ios::out | ios::binary);
	if(!out) {
		cout << "Error: Cannot open file.\n";
		exit(1);
	}

	out.write((char *) AM.data(), sizeof(double) * AM.getN());

	out.close();
}

template<class ViewDouble>
void inline verifyX(ViewDouble A, int rank){
	if(verify){
		HostMirror<double> AM(A);

		double *correct = new double[AM.getN()];

		ifstream in("output/output.bin" + to_string(rank), ios::in | ios::binary);
		if(!in) {
			verify=0;
			//cout << "Cannot open output.bin. returning without verification.\n";
			return;
		}
		in.read((char *) correct, sizeof(double) * AM.getN());
		in.close();

		//compare
//#pragma omp parallel for
		for(int i=0; i<AM.getN(); i++){
			if(fabs(AM(i) - correct[i]) > 1e-9 ){
				printf("########## Output does not match. Error in %s %d: at node %d -> correct: %f \t computed:%f \t diff: %f\n", __FILE__, __LINE__, i, correct[i], AM(i), fabs(AM(i) - correct[i]));
				exit(1);
			}
		}

		printf("output matched!!\n");

		delete[] correct;
	}
}

#endif	// EXPORT_H
