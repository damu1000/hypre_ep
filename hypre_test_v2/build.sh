#!/bin/bash

export LIB_XML_INCLUDE=/usr/include/libxml2
export HYPRE_CPU=/home/sci/damodars/hypre_ep_reproduce/hypre_cpu/src/build

mpicxx hypre_cpu.cc -std=c++11 -I$HYPRE_CPU/include/ -L$HYPRE_CPU/lib -lHYPRE -I$LIB_XML_INCLUDE -lxml2 -g -O2 -xMIC-AVX512 -o 1_cpu -fopenmp 

export HYPRE_EP=/home/sci/damodars/hypre_ep_reproduce/hypre_ep/src/build
export HYPRE_EP_SRC=/home/sci/damodars/hypre_ep_reproduce/hypre_ep/src

mpicxx hypre_cpu_custom_ep.cc -std=c++11 -I$HYPRE_EP/include/ -I$HYPRE_EP_SRC -L$HYPRE_EP/lib -lHYPRE -I$LIB_XML_INCLUDE -lxml2 -g -O2 -xMIC-AVX512 -o 2_ep -fopenmp -DUSE_SIMD


