#!/bin/bash

rm -f ./output.bin

echo executing cpu:
echo
./1_cpu c 32 1 1 1
echo
echo
echo executing gpu original:
echo
./2_cuda_orig c 32 1 1 1
echo
echo

echo executing gpu ep:
echo
./3_cuda_ep c 32 1 1 1

rm -f ./output.bin
echo
echo

