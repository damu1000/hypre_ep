/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for the BoxLoop
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * BoxLoop macros:
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_NEWBOXLOOP_HEADER
#define HYPRE_NEWBOXLOOP_HEADER


//ICCS with openmp: 1, new: 2, adaptive: 3, adaptive with fixed threads: 4, ICCS with custom parallel_for: 5
#define BOXLOOP_VER 5


#ifdef WIN32
#define Pragma(x) __pragma(#x)
#else
#define Pragma(x) _Pragma(#x)
#endif

#if BOXLOOP_VER==3

//added num_threads for adaptive
#ifdef HYPRE_USING_OPENMP
#define HYPRE_BOX_REDUCTION
#define OMP1(tsize) Pragma(omp parallel for num_threads(tsize) private(HYPRE_BOX_PRIVATE) HYPRE_BOX_REDUCTION HYPRE_SMP_SCHEDULE)
#else
#define OMP1(tsize)
#endif

#elif BOXLOOP_VER==4

//added num_threads for adaptive
#ifdef HYPRE_USING_OPENMP
#define HYPRE_BOX_REDUCTION
#define OMP1(tsize) Pragma(omp parallel for private(HYPRE_BOX_PRIVATE) HYPRE_BOX_REDUCTION HYPRE_SMP_SCHEDULE)
#else
#define OMP1(tsize)
#endif

#else

#ifdef HYPRE_USING_OPENMP
#define HYPRE_BOX_REDUCTION
#define OMP1 Pragma(omp parallel for private(HYPRE_BOX_PRIVATE) HYPRE_BOX_REDUCTION HYPRE_SMP_SCHEDULE)
#else
#define OMP1
#endif

#endif


#if (BOXLOOP_VER && defined(HYPRE_USING_OPENMP))
#error "Do not give --with-openmp option while configuring if BOXLOOP_VER is set to 5 (using custom parallel_for)"
#endif

#if (BOXLOOP_VER==1 || BOXLOOP_VER==5)

/********************************************************************************************************************************************

 ICCS loops

*********************************************************************************************************************************************/


typedef struct hypre_Box_info_struct{
	int stride[3], start;
}hypre_Box_info;


#define zypre_CalcStride(boxinfo, dbox, start_a, stride_a)\
		boxinfo.start = hypre_BoxIndexRank(dbox, start_a);\
		boxinfo.stride[0] =  stride_a[0];	\
		boxinfo.stride[1] =  hypre_max(0, ((dbox)->imax)[0] - ((dbox)->imin)[0] + 1)*stride_a[1] -	\
							 loop_dim[0]*stride_a[0]; \
		boxinfo.stride[2] =  hypre_max(0, ((dbox)->imax)[0] - ((dbox)->imin)[0] + 1)*hypre_max(0, ((dbox)->imax)[1] - ((dbox)->imin)[1] + 1)*stride_a[2] - \
							 loop_dim[1]*hypre_max(0, ((dbox)->imax)[0] - ((dbox)->imin)[0] + 1)*stride_a[1];

#define zypre_CalcStrideBasic(boxinfo, stride_a)\
		boxinfo.start = 0;\
		boxinfo.stride[0] =  stride_a[0];	\
		boxinfo.stride[1] =  stride_a[1] - loop_dim[0]*stride_a[0]; \
		boxinfo.stride[2] =  stride_a[2] - loop_dim[1]*stride_a[1];


#if BOXLOOP_VER==1
//use simple openmp for 1
#define executeLoop()\
	if(loop_dim[2]>1){\
		Pragma(omp parallel for HYPRE_BOX_REDUCTION HYPRE_SMP_SCHEDULE) \
		for(int hypre___k=0; hypre___k < loop_dim[2]; hypre___k++) \
			f(hypre___k);\
	}\
	else if(loop_dim[2]==1){\
		f(0);\
	}

#define executeLoopReduction(reducesum)\
	if(loop_dim[2]>1){\
		Pragma(omp parallel for HYPRE_BOX_REDUCTION HYPRE_SMP_SCHEDULE) \
		for(int hypre___k=0; hypre___k < loop_dim[2]; hypre___k++) \
			f(hypre___k, reducesum);\
	}\
	else if(loop_dim[2]==1){\
		f(0, reducesum);\
	}

#elif BOXLOOP_VER==5
//use custom par_for for 5

extern "C++"{
#include<functional>
extern void custom_parallel_for(int b, int e, std::function<void(int)> f);


#define executeLoop()\
	if(loop_dim[2]>1)\
		custom_parallel_for(0, loop_dim[2], f);\
	else if(loop_dim[2]==1)\
		f(0);

//#define executeLoopReduction(reducesum)\
//	if(loop_dim[2]>1)\
//		custom_parallel_reduce(0, loop_dim[2], f, reducesum);\
//	else if(loop_dim[2]==1)\
//		f(0, reducesum);

//use serial reduction for now. Used during setup. Changed struct_innerproduce.c not to use reduction loop. Dirty but works for now
#define executeLoopReduction(reducesum)\
	for(int i=0; i<loop_dim[2]; i++)\
		f(i, reducesum);


#endif


#define zypre_newBoxLoop0Begin(ndim, loop_size)\
{\
	int *loop_dim = loop_size; \
	auto f = [&](int hypre___k)\
	{\
		int hypre___i, hypre___j;\
    	for(hypre___j=0; hypre___j < loop_dim[1]; hypre___j++)\
		{\
    		for(hypre___i=0; hypre___i < loop_dim[0]; hypre___i++)\
			{\


#define zypre_newBoxLoop0End()\
			}\
		}\
	};\
	executeLoop();\
}


#define zypre_newBoxLoop1Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1)\
{\
	int *loop_dim = loop_size;\
	hypre_Box_info boxinfo1;\
	zypre_CalcStride(boxinfo1, dbox1, start1, stride1);\
	auto f = [&](int hypre___k)\
	{\
		int hypre___i, hypre___j;\
		int i1 = boxinfo1.start + hypre___k * (boxinfo1.stride[2] + loop_dim[1] * boxinfo1.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo1.stride[0]);\
		for(hypre___j=0; hypre___j < loop_dim[1]; hypre___j++)\
		{\
    		for(hypre___i=0; hypre___i < loop_dim[0]; hypre___i++)\
			{

#define zypre_newBoxLoop1End(i1)\
				i1 += boxinfo1.stride[0];\
			}\
			i1 += boxinfo1.stride[1];\
		}\
	};\
	executeLoop();\
}



#define zypre_newBoxLoop1BeginSimd(ndim, loop_size,\
                            dbox1, start1, stride1, i1)\
{\
	int *loop_dim = loop_size;\
	hypre_Box_info boxinfo1;\
	zypre_CalcStride(boxinfo1, dbox1, start1, stride1);\
	auto f = [&](int hypre___k)\
	{\
		int hypre___i, hypre___j;\
		int i1 = boxinfo1.start + hypre___k * (boxinfo1.stride[2] + loop_dim[1] * boxinfo1.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo1.stride[0]);\
		for(hypre___j=0; hypre___j < loop_dim[1]; hypre___j++)\
		{\
    		_Pragma("omp simd")\
    		for(hypre___i=0; hypre___i < loop_dim[0]; hypre___i++)\
			{

#define zypre_newBoxLoop1EndSimd(i1)\
				i1++;\
			}\
			i1 += boxinfo1.stride[1];\
		}\
	};\
	executeLoop();\
}


#define zypre_newBoxLoop2Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1, \
							dbox2, start2, stride2, i2) \
{\
	int *loop_dim = loop_size;\
	hypre_Box_info boxinfo1, boxinfo2;\
	zypre_CalcStride(boxinfo1, dbox1, start1, stride1);	\
	zypre_CalcStride(boxinfo2, dbox2, start2, stride2); \
	auto f = [&](int hypre___k)\
	{\
		int hypre___i, hypre___j;\
		int i1 = boxinfo1.start + hypre___k * (boxinfo1.stride[2] + loop_dim[1] * boxinfo1.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo1.stride[0]);\
		int i2 = boxinfo2.start + hypre___k * (boxinfo2.stride[2] + loop_dim[1] * boxinfo2.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo2.stride[0]);\
    	for(hypre___j=0; hypre___j < loop_dim[1]; hypre___j++)\
		{\
    		for(hypre___i=0; hypre___i < loop_dim[0]; hypre___i++)\
			{



#define zypre_newBoxLoop2End(i1, i2)\
				i1 += boxinfo1.stride[0];\
				i2 += boxinfo2.stride[0];\
			}\
			i1 += boxinfo1.stride[1];\
			i2 += boxinfo2.stride[1];\
		}\
	};\
	executeLoop();\
}


#define zypre_newBoxLoop2BeginSimd(ndim, loop_size,\
                            dbox1, start1, stride1, i1, \
							dbox2, start2, stride2, i2) \
{\
	int *loop_dim = loop_size;\
	hypre_Box_info boxinfo1, boxinfo2;\
	zypre_CalcStride(boxinfo1, dbox1, start1, stride1);	\
	zypre_CalcStride(boxinfo2, dbox2, start2, stride2); \
	auto f = [&](int hypre___k)\
	{\
		int hypre___i, hypre___j;\
		int i1 = boxinfo1.start + hypre___k * (boxinfo1.stride[2] + loop_dim[1] * boxinfo1.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo1.stride[0]);\
		int i2 = boxinfo2.start + hypre___k * (boxinfo2.stride[2] + loop_dim[1] * boxinfo2.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo2.stride[0]);\
    	for(hypre___j=0; hypre___j < loop_dim[1]; hypre___j++)\
		{\
    		_Pragma("omp simd")\
    		for(hypre___i=0; hypre___i < loop_dim[0]; hypre___i++)\
			{



#define zypre_newBoxLoop2EndSimd(i1, i2)\
				i1++;\
				i2++;\
			}\
			i1 += boxinfo1.stride[1];\
			i2 += boxinfo2.stride[1];\
		}\
	};\
	executeLoop();\
}



#define zypre_newBasicBoxLoop2Begin(ndim, loop_size,                          \
        stride1, i1,                              \
        stride2, i2) \
{\
	int *loop_dim = loop_size;\
	hypre_Box_info boxinfo1, boxinfo2;\
	zypre_CalcStrideBasic(boxinfo1, stride1);	\
	zypre_CalcStrideBasic(boxinfo2, stride2); \
	auto f = [&](int hypre___k)\
	{\
		int hypre___i, hypre___j;\
		int i1 = boxinfo1.start + hypre___k * (boxinfo1.stride[2] + loop_dim[1] * boxinfo1.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo1.stride[0]);\
		int i2 = boxinfo2.start + hypre___k * (boxinfo2.stride[2] + loop_dim[1] * boxinfo2.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo2.stride[0]);\
    	for(hypre___j=0; hypre___j < loop_dim[1]; hypre___j++)\
		{\
    		for(hypre___i=0; hypre___i < loop_dim[0]; hypre___i++)\
			{


#define zypre_newBasicBoxLoop2End(i1, i2)\
				i1 += boxinfo1.stride[0];\
				i2 += boxinfo2.stride[0];\
			}\
			i1 += boxinfo1.stride[1];\
			i2 += boxinfo2.stride[1];\
		}\
	};\
	for(int hypre___k=0; hypre___k < loop_dim[2]; hypre___k++) \
		f(hypre___k);\
}


#define zypre_newBasicBoxLoop2BeginSimd(ndim, loop_size,                          \
        stride1, i1,                              \
        stride2, i2) \
{\
	int *loop_dim = loop_size;\
	hypre_Box_info boxinfo1, boxinfo2;\
	zypre_CalcStrideBasic(boxinfo1, stride1);	\
	zypre_CalcStrideBasic(boxinfo2, stride2); \
	auto f = [&](int hypre___k)\
	{\
		int hypre___i, hypre___j;\
		int i1 = boxinfo1.start + hypre___k * (boxinfo1.stride[2] + loop_dim[1] * boxinfo1.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo1.stride[0]);\
		int i2 = boxinfo2.start + hypre___k * (boxinfo2.stride[2] + loop_dim[1] * boxinfo2.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo2.stride[0]);\
    	for(hypre___j=0; hypre___j < loop_dim[1]; hypre___j++)\
		{\
    		_Pragma("omp simd")\
    		for(hypre___i=0; hypre___i < loop_dim[0]; hypre___i++)\
			{


#define zypre_newBasicBoxLoop2EndSimd(i1, i2)\
				i1++;\
				i2++;\
			}\
			i1 += boxinfo1.stride[1];\
			i2 += boxinfo2.stride[1];\
		}\
	};\
	for(int hypre___k=0; hypre___k < loop_dim[2]; hypre___k++) \
		f(hypre___k);\
}


#define hypre_BoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, \
                                                      dbox2, start2, stride2, i2, reducesum) \
{\
	int *loop_dim = loop_size;\
	hypre_Box_info boxinfo1, boxinfo2;\
	zypre_CalcStride(boxinfo1, dbox1, start1, stride1);	\
	zypre_CalcStride(boxinfo2, dbox2, start2, stride2); \
	auto f = [&](int hypre___k, decltype(reducesum)& reducesum)\
	{\
		int hypre___i, hypre___j;\
		int i1 = boxinfo1.start + hypre___k * (boxinfo1.stride[2] + loop_dim[1] * boxinfo1.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo1.stride[0]);\
		int i2 = boxinfo2.start + hypre___k * (boxinfo2.stride[2] + loop_dim[1] * boxinfo2.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo2.stride[0]);\
    	for(hypre___j=0; hypre___j < loop_dim[1]; hypre___j++)\
		{\
    		for(hypre___i=0; hypre___i < loop_dim[0]; hypre___i++)\
			{



#define hypre_BoxLoop2ReductionEnd(i1, i2, reducesum)                           \
				i1 += boxinfo1.stride[0];\
				i2 += boxinfo2.stride[0];\
			}\
			i1 += boxinfo1.stride[1];\
			i2 += boxinfo2.stride[1];\
		}\
	};\
	executeLoopReduction(reducesum);\
}


#define hypre_BoxLoop2ReductionBeginSimd(ndim, loop_size, dbox1, start1, stride1, i1, \
                                                      dbox2, start2, stride2, i2, reducesum) \
{\
	int *loop_dim = loop_size;\
	hypre_Box_info boxinfo1, boxinfo2;\
	zypre_CalcStride(boxinfo1, dbox1, start1, stride1);	\
	zypre_CalcStride(boxinfo2, dbox2, start2, stride2); \
	auto f = [&](int hypre___k, decltype(reducesum)& reducesum)\
	{\
		int hypre___i, hypre___j;\
		int i1 = boxinfo1.start + hypre___k * (boxinfo1.stride[2] + loop_dim[1] * boxinfo1.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo1.stride[0]);\
		int i2 = boxinfo2.start + hypre___k * (boxinfo2.stride[2] + loop_dim[1] * boxinfo2.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo2.stride[0]);\
    	for(hypre___j=0; hypre___j < loop_dim[1]; hypre___j++)\
		{\
    		Pragma(omp simd HYPRE_BOX_REDUCTION)\
    		for(hypre___i=0; hypre___i < loop_dim[0]; hypre___i++)\
			{



#define hypre_BoxLoop2ReductionEndSimd(i1, i2, reducesum)                           \
				i1++;\
				i2++;\
			}\
			i1 += boxinfo1.stride[1];\
			i2 += boxinfo2.stride[1];\
		}\
	};\
	executeLoopReduction(reducesum);\
}


#define zypre_newBoxLoop3Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1, \
							dbox2, start2, stride2, i2,  \
							dbox3, start3, stride3, i3) \
{\
	int *loop_dim = loop_size;\
	hypre_Box_info boxinfo1, boxinfo2, boxinfo3;\
	zypre_CalcStride(boxinfo1, dbox1, start1, stride1);	\
	zypre_CalcStride(boxinfo2, dbox2, start2, stride2);\
	zypre_CalcStride(boxinfo3, dbox3, start3, stride3); \
	auto f = [&](int hypre___k)\
	{\
		int hypre___i, hypre___j;\
		int i1 = boxinfo1.start + hypre___k * (boxinfo1.stride[2] + loop_dim[1] * boxinfo1.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo1.stride[0]);\
		int i2 = boxinfo2.start + hypre___k * (boxinfo2.stride[2] + loop_dim[1] * boxinfo2.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo2.stride[0]);\
		int i3 = boxinfo3.start + hypre___k * (boxinfo3.stride[2] + loop_dim[1] * boxinfo3.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo3.stride[0]); \
    	for(hypre___j=0; hypre___j < loop_dim[1]; hypre___j++)\
		{\
    		for(hypre___i=0; hypre___i < loop_dim[0]; hypre___i++)\
			{

#define zypre_newBoxLoop3End(i1, i2, i3)\
				i1 += boxinfo1.stride[0];\
				i2 += boxinfo2.stride[0];\
				i3 += boxinfo3.stride[0];\
			}\
			i1 += boxinfo1.stride[1];\
			i2 += boxinfo2.stride[1];\
			i3 += boxinfo3.stride[1];\
		}\
	};\
	executeLoop();\
}




#define zypre_newBoxLoop3BeginSimd(ndim, loop_size,\
                            dbox1, start1, stride1, i1, \
							dbox2, start2, stride2, i2,  \
							dbox3, start3, stride3, i3) \
{\
	int *loop_dim = loop_size;\
	hypre_Box_info boxinfo1, boxinfo2, boxinfo3;\
	zypre_CalcStride(boxinfo1, dbox1, start1, stride1);	\
	zypre_CalcStride(boxinfo2, dbox2, start2, stride2);\
	zypre_CalcStride(boxinfo3, dbox3, start3, stride3); \
	auto f = [&](int hypre___k)\
	{\
		int hypre___i, hypre___j;\
		int i1 = boxinfo1.start + hypre___k * (boxinfo1.stride[2] + loop_dim[1] * boxinfo1.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo1.stride[0]);\
		int i2 = boxinfo2.start + hypre___k * (boxinfo2.stride[2] + loop_dim[1] * boxinfo2.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo2.stride[0]);\
		int i3 = boxinfo3.start + hypre___k * (boxinfo3.stride[2] + loop_dim[1] * boxinfo3.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo3.stride[0]); \
    	for(hypre___j=0; hypre___j < loop_dim[1]; hypre___j++)\
		{\
    		_Pragma("omp simd")\
    		for(hypre___i=0; hypre___i < loop_dim[0]; hypre___i++)\
			{

#define zypre_newBoxLoop3EndSimd(i1, i2, i3)\
				i1++;\
				i2++;\
				i3++;\
			}\
			i1 += boxinfo1.stride[1];\
			i2 += boxinfo2.stride[1];\
			i3 += boxinfo3.stride[1];\
		}\
	};\
	executeLoop();\
}



#define zypre_newBoxLoop4Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1, \
							dbox2, start2, stride2, i2,  \
							dbox3, start3, stride3, i3,  \
							dbox4, start4, stride4, i4) \
{\
	int *loop_dim = loop_size;\
	hypre_Box_info boxinfo1, boxinfo2, boxinfo3, boxinfo4;\
	zypre_CalcStride(boxinfo1, dbox1, start1, stride1);	\
	zypre_CalcStride(boxinfo2, dbox2, start2, stride2);\
	zypre_CalcStride(boxinfo3, dbox3, start3, stride3);\
	zypre_CalcStride(boxinfo4, dbox4, start4, stride4);\
	auto f = [&](int hypre___k)\
	{\
		int hypre___i, hypre___j;\
		int i1 = boxinfo1.start + hypre___k * (boxinfo1.stride[2] + loop_dim[1] * boxinfo1.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo1.stride[0]);\
		int i2 = boxinfo2.start + hypre___k * (boxinfo2.stride[2] + loop_dim[1] * boxinfo2.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo2.stride[0]);\
		int i3 = boxinfo3.start + hypre___k * (boxinfo3.stride[2] + loop_dim[1] * boxinfo3.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo3.stride[0]); \
		int i4 = boxinfo4.start + hypre___k * (boxinfo4.stride[2] + loop_dim[1] * boxinfo4.stride[1] + loop_dim[0] * loop_dim[1] * boxinfo4.stride[0]); \
    	for(hypre___j=0; hypre___j < loop_dim[1]; hypre___j++)\
		{\
    		for(hypre___i=0; hypre___i < loop_dim[0]; hypre___i++)\
			{\


#define zypre_newBoxLoop4End(i1, i2, i3, i4)\
				i1 += boxinfo1.stride[0];\
				i2 += boxinfo2.stride[0];\
				i3 += boxinfo3.stride[0];\
				i4 += boxinfo4.stride[0];\
			}\
			i1 += boxinfo1.stride[1];\
			i2 += boxinfo2.stride[1];\
			i3 += boxinfo3.stride[1];\
			i4 += boxinfo4.stride[1];\
		}\
		i1 += boxinfo1.stride[2];\
		i2 += boxinfo2.stride[2];\
		i3 += boxinfo3.stride[2];\
		i4 += boxinfo4.stride[2];\
	};\
	executeLoop();\
}
#if BOXLOOP_VER==5
};
#endif

/********************************************************************************************************************************************

 ICCS loops end

*********************************************************************************************************************************************/

#elif BOXLOOP_VER==2

/********************************************************************************************************************************************

 New version

*********************************************************************************************************************************************/


typedef struct hypre_Boxloop_struct
{
   HYPRE_Int lsize0,lsize1,lsize2;
   HYPRE_Int strides0,strides1,strides2;
   HYPRE_Int bstart0,bstart1,bstart2;
   HYPRE_Int bsize0,bsize1,bsize2;
} hypre_Boxloop;

#define executeLoop()                                                         \
   extern thread_local int hypre_min_workload;                                             \
   if(hypre__tot > hypre_min_workload){                                       \
      OMP1                                                                    \
      for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
         boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i);       \
   }                                                                          \
   else{                                                                      \
      for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
          boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i);      \
   }                                                                          \

#define zypre_newBoxLoop0Begin(ndim, loop_size)                               \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
	  int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      zypre_BoxLoopSet();                                                     \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop0End()                                                \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop();                                                             \
}

#define zypre_newBoxLoop1Begin(ndim, loop_size,                               \
                               dbox1, start1, stride1, i1)                    \
{                                                                             \
   HYPRE_Int i1;                                                              \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
	  int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop1End(i1)                                              \
            i1 += hypre__i0inc1;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop();                                                             \
}


#define zypre_newBoxLoop2Begin(ndim, loop_size,                               \
                               dbox1, start1, stride1, i1,                    \
                               dbox2, start2, stride2, i2)                    \
{                                                                             \
   HYPRE_Int i1, i2;                                                          \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
	  int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2;                                                       \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop2End(i1, i2)                                          \
            i1 += hypre__i0inc1;                                              \
            i2 += hypre__i0inc2;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop();                                                             \
}


#define zypre_newBoxLoop3Begin(ndim, loop_size,                               \
                               dbox1, start1, stride1, i1,                    \
                               dbox2, start2, stride2, i2,                    \
                               dbox3, start3, stride3, i3)                    \
{                                                                             \
   HYPRE_Int i1, i2, i3;                                                      \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopDeclareK(3);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
	  int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2, i3;                                                   \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      zypre_BoxLoopSetK(3, i3);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop3End(i1, i2, i3)                                      \
            i1 += hypre__i0inc1;                                              \
            i2 += hypre__i0inc2;                                              \
            i3 += hypre__i0inc3;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         i3 += hypre__ikinc3[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop();                                                             \
}

#define zypre_newBoxLoop4Begin(ndim, loop_size,                               \
                            dbox1, start1, stride1, i1,                       \
                            dbox2, start2, stride2, i2,                       \
                            dbox3, start3, stride3, i3,                       \
                            dbox4, start4, stride4, i4)                       \
{                                                                             \
   HYPRE_Int i1, i2, i3, i4;                                                  \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopDeclareK(3);                                                  \
   zypre_BoxLoopDeclareK(4);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);                         \
   zypre_BoxLoopInitK(4, dbox4, start4, stride4, i4);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
	  int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2, i3, i4;                                               \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      zypre_BoxLoopSetK(3, i3);                                               \
      zypre_BoxLoopSetK(4, i4);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop4End(i1, i2, i3, i4)                                  \
            i1 += hypre__i0inc1;                                              \
            i2 += hypre__i0inc2;                                              \
            i3 += hypre__i0inc3;                                              \
            i4 += hypre__i0inc4;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         i3 += hypre__ikinc3[hypre__d];                                       \
         i4 += hypre__ikinc4[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop();                                                             \
}

//removed omp1 from basic loop. Used only in struct_communication.c. Moved omp to outer loop
#define zypre_newBasicBoxLoop2Begin(ndim, loop_size,                          \
                                    stride1, i1,                              \
                                    stride2, i2)                              \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BasicBoxLoopInitK(1, stride1);                                       \
   zypre_BasicBoxLoopInitK(2, stride2);                                       \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
	  int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2;                                                       \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBasicBoxLoop2End(i1, i2)                                          \
            i1 += hypre__i0inc1;                                              \
            i2 += hypre__i0inc2;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
        boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i); \
}

#define zypre_newBasicBoxLoop2BeginSimd(ndim, loop_size,                      \
                                    stride1, i1,                              \
                                    stride2, i2)                              \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BasicBoxLoopInitK(1, stride1);                                       \
   zypre_BasicBoxLoopInitK(2, stride2);                                       \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
	  int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2;                                                       \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
    	 _Pragma("omp simd")                                                  \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBasicBoxLoop2EndSimd(i1, i2)                                 \
            i1++;                                                             \
            i2++;                                                             \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
        boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i); \
}


#define zypre_newBoxLoop1BeginSimd(ndim, loop_size,                           \
                               dbox1, start1, stride1, i1)                    \
{                                                                             \
   HYPRE_Int i1;                                                              \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
	  int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1;                                                           \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         _Pragma("omp simd")                                                  \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {


#define zypre_newBoxLoop1EndSimd(i1)                                          \
            i1++;                                                             \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop();                                                             \
}

#define zypre_newBoxLoop2BeginSimd(ndim, loop_size,                           \
                               dbox1, start1, stride1, i1,                    \
                               dbox2, start2, stride2, i2)                    \
{                                                                             \
   HYPRE_Int i1, i2;                                                          \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
	  int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2;                                                       \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         _Pragma("omp simd")                                                  \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop2EndSimd(i1, i2)                                      \
            i1++;                                                             \
            i2++;                                                             \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop();                                                             \
}

#define zypre_newBoxLoop3BeginSimd(ndim, loop_size,                           \
                               dbox1, start1, stride1, i1,                    \
                               dbox2, start2, stride2, i2,                    \
                               dbox3, start3, stride3, i3)                    \
{                                                                             \
   HYPRE_Int i1, i2, i3;                                                      \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopDeclareK(3);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
	  int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2, i3;                                                   \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      zypre_BoxLoopSetK(3, i3);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         _Pragma("omp simd")                                                  \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop3EndSimd(i1, i2, i3)                                  \
            i1++;                                                             \
            i2++;                                                             \
            i3++;                                                             \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         i3 += hypre__ikinc3[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop();                                                             \
}


#define hypre_BoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, \
                                                      dbox2, start2, stride2, i2, reducesum) \
{                                                                               \
	 HYPRE_Int i1, i2;                                                          \
	 zypre_BoxLoopDeclare();                                                    \
	 zypre_BoxLoopDeclareK(1);                                                  \
	 zypre_BoxLoopDeclareK(2);                                                  \
	 zypre_BoxLoopInit(ndim, loop_size);                                        \
	 zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
	 zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
	 auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1, decltype(reducesum)& reducesum)        \
	 {                                                                          \
        int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
		HYPRE_Int i1, i2;                                                       \
		zypre_BoxLoopSet();                                                     \
		zypre_BoxLoopSetK(1, i1);                                               \
		zypre_BoxLoopSetK(2, i2);                                               \
		for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
		{                                                                       \
		   for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
		   {

#define hypre_BoxLoop2ReductionEnd(i1, i2, reducesum)                           \
              i1 += hypre__i0inc1;                                              \
              i2 += hypre__i0inc2;                                              \
           }                                                                    \
           zypre_BoxLoopInc1();                                                 \
           i1 += hypre__ikinc1[hypre__d];                                       \
           i2 += hypre__ikinc2[hypre__d];                                       \
           zypre_BoxLoopInc2();                                                 \
        }                                                                       \
     };                                                                         \
     extern thread_local int hypre_min_workload;                                             \
     if(hypre__tot > hypre_min_workload){                                       \
        OMP1                                                                    \
        for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
           boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i, reducesum);       \
     }                                                                          \
     else{                                                                      \
        for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
            boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i, reducesum);      \
     }                                                                          \
}


#define hypre_BoxLoop2ReductionBeginSimd(ndim, loop_size, dbox1, start1, stride1, i1, \
                                                      dbox2, start2, stride2, i2, reducesum) \
{                                                                               \
	 HYPRE_Int i1, i2;                                                          \
	 zypre_BoxLoopDeclare();                                                    \
	 zypre_BoxLoopDeclareK(1);                                                  \
	 zypre_BoxLoopDeclareK(2);                                                  \
	 zypre_BoxLoopInit(ndim, loop_size);                                        \
	 zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
	 zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
	 auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1, decltype(reducesum)& reducesum)        \
	 {                                                                          \
        int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
		HYPRE_Int i1, i2;                                                       \
		zypre_BoxLoopSet();                                                     \
		zypre_BoxLoopSetK(1, i1);                                               \
		zypre_BoxLoopSetK(2, i2);                                               \
		for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
		{                                                                       \
		   Pragma(omp simd HYPRE_BOX_REDUCTION)                                                      \
		   for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
		   {

#define hypre_BoxLoop2ReductionEndSimd(i1, i2, reducesum)                           \
              i1++;                                              \
              i2++;                                              \
           }                                                                    \
           zypre_BoxLoopInc1();                                                 \
           i1 += hypre__ikinc1[hypre__d];                                       \
           i2 += hypre__ikinc2[hypre__d];                                       \
           zypre_BoxLoopInc2();                                                 \
        }                                                                       \
     };                                                                         \
     extern thread_local int hypre_min_workload;                                             \
     if(hypre__tot > hypre_min_workload){                                       \
        OMP1                                                                    \
        for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
           boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i, reducesum);       \
     }                                                                          \
     else{                                                                      \
        for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
            boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i, reducesum);      \
     }                                                                          \
}



/********************************************************************************************************************************************

 New version

*********************************************************************************************************************************************/

#elif (BOXLOOP_VER==3 || BOXLOOP_VER==4)


/********************************************************************************************************************************************

 Adaptive parallelism

*********************************************************************************************************************************************/

//for KNL
#define SIMD_LEN 8

//for sandy bridge
//#define SIMD_LEN 4

typedef struct hypre_Boxloop_struct
{
   HYPRE_Int lsize0,lsize1,lsize2;
   HYPRE_Int strides0,strides1,strides2;
   HYPRE_Int bstart0,bstart1,bstart2;
   HYPRE_Int bsize0,bsize1,bsize2;
} hypre_Boxloop;

#define executeLoop(SIMDL)                                                    \
   HYPRE_Int hypre__end_line=__LINE__;                                        \
   HYPRE_Int hypre__lines=hypre__end_line - hypre__start_line - 3;            \
   HYPRE_Int hypre__work = (hypre__tot*hypre__lines + SIMDL - 1) / SIMDL;     \
   extern thread_local int hypre_min_workload;                                \
   extern int gteam_size;                                                     \
   if(hypre__work > hypre_min_workload){                                      \
      HYPRE_Int threads = hypre__work / hypre_min_workload;                   \
      threads = (threads < gteam_size) ? threads : gteam_size;                \
      OMP1(threads)                                                           \
      for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++){\
         /*printf("%s:%d parallel hypre__tot %d, hypre__lines %d, hypre__work %d, threads %d, ompthreads %d\n", __FILE__, __LINE__ , hypre__tot, hypre__lines, hypre__work, threads, omp_get_num_threads());*/ \
         boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i);       \
      }                                                                       \
   }                                                                          \
   else{                                                                      \
      /*printf("%s:%d serial hypre__tot %d, hypre__lines %d, hypre__work %d, threads %d, ompthreads %d\n", __FILE__, __LINE__ , hypre__tot, hypre__lines, hypre__work, 1, 1);*/ \
      for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
          boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i);      \
   }


#define executeReductionLoop(reducesum, SIMDL)                                \
   HYPRE_Int hypre__end_line=__LINE__;                                        \
   HYPRE_Int hypre__lines=hypre__end_line - hypre__start_line - 3;            \
   HYPRE_Int hypre__work = (hypre__tot*hypre__lines + SIMDL - 1) / SIMDL;     \
   extern thread_local int hypre_min_workload;                                \
   extern int gteam_size;                                                     \
   if(hypre__work > hypre_min_workload){                                      \
      HYPRE_Int threads = hypre__work / hypre_min_workload;                   \
      threads = (threads < gteam_size) ? threads : gteam_size;                \
      OMP1(threads)                                                           \
      for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++){\
         /*printf("%s:%d parallel hypre__tot %d, hypre__lines %d, hypre__work %d, threads %d, ompthreads %d\n", __FILE__, __LINE__ , hypre__tot, hypre__lines, hypre__work, threads, omp_get_num_threads()); */\
         boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i, reducesum);       \
      }                                                                       \
   }                                                                          \
   else{                                                                      \
      /*printf("%s:%d serial hypre__tot %d, hypre__lines %d, hypre__work %d, threads %d, ompthreads %d\n", __FILE__, __LINE__ , hypre__tot, hypre__lines, hypre__work, 1, 1);*/ \
      for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
          boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i, reducesum);      \
   }


#define zypre_newBoxLoop0Begin(ndim, loop_size)                               \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
      int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      zypre_BoxLoopSet();                                                     \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop0End()                                                \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop(1);                                                            \
}

#define zypre_newBoxLoop1Begin(ndim, loop_size,                               \
                               dbox1, start1, stride1, i1)                    \
{                                                                             \
   HYPRE_Int i1;                                                              \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
      int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop1End(i1)                                              \
            i1 += hypre__i0inc1;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop(1);                                                            \
}


#define zypre_newBoxLoop2Begin(ndim, loop_size,                               \
                               dbox1, start1, stride1, i1,                    \
                               dbox2, start2, stride2, i2)                    \
{                                                                             \
   HYPRE_Int i1, i2;                                                          \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
      int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2;                                                       \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop2End(i1, i2)                                          \
            i1 += hypre__i0inc1;                                              \
            i2 += hypre__i0inc2;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop(1);                                                            \
}


#define zypre_newBoxLoop3Begin(ndim, loop_size,                               \
                               dbox1, start1, stride1, i1,                    \
                               dbox2, start2, stride2, i2,                    \
                               dbox3, start3, stride3, i3)                    \
{                                                                             \
   HYPRE_Int i1, i2, i3;                                                      \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopDeclareK(3);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
      int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2, i3;                                                   \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      zypre_BoxLoopSetK(3, i3);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop3End(i1, i2, i3)                                      \
            i1 += hypre__i0inc1;                                              \
            i2 += hypre__i0inc2;                                              \
            i3 += hypre__i0inc3;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         i3 += hypre__ikinc3[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop(1);                                                            \
}

#define zypre_newBoxLoop4Begin(ndim, loop_size,                               \
                            dbox1, start1, stride1, i1,                       \
                            dbox2, start2, stride2, i2,                       \
                            dbox3, start3, stride3, i3,                       \
                            dbox4, start4, stride4, i4)                       \
{                                                                             \
   HYPRE_Int i1, i2, i3, i4;                                                  \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopDeclareK(3);                                                  \
   zypre_BoxLoopDeclareK(4);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);                         \
   zypre_BoxLoopInitK(4, dbox4, start4, stride4, i4);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
      int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2, i3, i4;                                               \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      zypre_BoxLoopSetK(3, i3);                                               \
      zypre_BoxLoopSetK(4, i4);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop4End(i1, i2, i3, i4)                                  \
            i1 += hypre__i0inc1;                                              \
            i2 += hypre__i0inc2;                                              \
            i3 += hypre__i0inc3;                                              \
            i4 += hypre__i0inc4;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         i3 += hypre__ikinc3[hypre__d];                                       \
         i4 += hypre__ikinc4[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop(1);                                                            \
}

//removed omp1 from basic loop. Used only in struct_communication.c. Moved omp to outer loop
#define zypre_newBasicBoxLoop2Begin(ndim, loop_size,                          \
                                    stride1, i1,                              \
                                    stride2, i2)                              \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BasicBoxLoopInitK(1, stride1);                                       \
   zypre_BasicBoxLoopInitK(2, stride2);                                       \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
      int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2;                                                       \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBasicBoxLoop2End(i1, i2)                                     \
            i1 += hypre__i0inc1;                                              \
            i2 += hypre__i0inc2;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
        boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i); \
}

#define zypre_newBoxLoop1BeginSimd(ndim, loop_size,                           \
                               dbox1, start1, stride1, i1)                    \
{                                                                             \
   HYPRE_Int i1;                                                              \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
      int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1;                                                           \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         _Pragma("omp simd")                                                  \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {


#define zypre_newBoxLoop1EndSimd(i1)                                          \
            i1++;                                                             \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop(SIMD_LEN);                                                     \
}

#define zypre_newBoxLoop2BeginSimd(ndim, loop_size,                           \
                               dbox1, start1, stride1, i1,                    \
                               dbox2, start2, stride2, i2)                    \
{                                                                             \
   HYPRE_Int i1, i2;                                                          \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
      int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2;                                                       \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         _Pragma("omp simd")                                                  \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop2EndSimd(i1, i2)                                      \
            i1++;                                                             \
            i2++;                                                             \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop(SIMD_LEN);                                                     \
}

#define zypre_newBoxLoop3BeginSimd(ndim, loop_size,                           \
                               dbox1, start1, stride1, i1,                    \
                               dbox2, start2, stride2, i2,                    \
                               dbox3, start3, stride3, i3)                    \
{                                                                             \
   HYPRE_Int i1, i2, i3;                                                      \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopDeclareK(3);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);                         \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
      int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2, i3;                                                   \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      zypre_BoxLoopSetK(3, i3);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         _Pragma("omp simd")                                                  \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop3EndSimd(i1, i2, i3)                                  \
            i1++;                                                             \
            i2++;                                                             \
            i3++;                                                             \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         i3 += hypre__ikinc3[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   executeLoop(SIMD_LEN);                                                     \
}

//removed omp1 from basic loop. Used only in struct_communication.c. Moved omp to outer loop
#define zypre_newBasicBoxLoop2BeginSimd(ndim, loop_size,                      \
                                    stride1, i1,                              \
                                    stride2, i2)                              \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BasicBoxLoopInitK(1, stride1);                                       \
   zypre_BasicBoxLoopInitK(2, stride2);                                       \
   auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1)        \
   {                                                                          \
      int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
      HYPRE_Int i1, i2;                                                       \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
    	 _Pragma("omp simd")                                                  \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBasicBoxLoop2EndSimd(i1, i2)                                 \
            i1++;                                                             \
            i2++;                                                             \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   };                                                                         \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
        boxLoopF(hypre__block, hypre__IN, hypre__JN, hypre__I, hypre__J, hypre__d, hypre__i); \
}


#define hypre_BoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, \
                                                      dbox2, start2, stride2, i2, reducesum) \
{                                                                               \
     HYPRE_Int i1, i2;                                                          \
     zypre_BoxLoopDeclare();                                                    \
     zypre_BoxLoopDeclareK(1);                                                  \
     zypre_BoxLoopDeclareK(2);                                                  \
     zypre_BoxLoopInit(ndim, loop_size);                                        \
     zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
     zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
     auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1, decltype(reducesum)& reducesum)        \
     {                                                                          \
        int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
        HYPRE_Int i1, i2;                                                       \
        zypre_BoxLoopSet();                                                     \
        zypre_BoxLoopSetK(1, i1);                                               \
        zypre_BoxLoopSetK(2, i2);                                               \
        for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
        {                                                                       \
           for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
           {

#define hypre_BoxLoop2ReductionEnd(i1, i2, reducesum)                           \
              i1 += hypre__i0inc1;                                              \
              i2 += hypre__i0inc2;                                              \
           }                                                                    \
           zypre_BoxLoopInc1();                                                 \
           i1 += hypre__ikinc1[hypre__d];                                       \
           i2 += hypre__ikinc2[hypre__d];                                       \
           zypre_BoxLoopInc2();                                                 \
        }                                                                       \
     };                                                                         \
     executeReductionLoop(reducesum, 1);                                        \
}


#define hypre_BoxLoop2ReductionBeginSimd(ndim, loop_size, dbox1, start1, stride1, i1, \
                                                      dbox2, start2, stride2, i2, reducesum) \
{                                                                               \
     HYPRE_Int i1, i2;                                                          \
     zypre_BoxLoopDeclare();                                                    \
     zypre_BoxLoopDeclareK(1);                                                  \
     zypre_BoxLoopDeclareK(2);                                                  \
     zypre_BoxLoopInit(ndim, loop_size);                                        \
     zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
     zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
     auto boxLoopF = [&] (int hypre__block, int hypre__IN, int hypre__JN, int hypre__I, int hypre__J, int hypre__d, int *hypre__i1, decltype(reducesum)& reducesum)        \
     {                                                                          \
        int hypre__i[] = {hypre__i1[0], hypre__i1[1], hypre__i1[2], hypre__i1[3]}; \
        HYPRE_Int i1, i2;                                                       \
        zypre_BoxLoopSet();                                                     \
        zypre_BoxLoopSetK(1, i1);                                               \
        zypre_BoxLoopSetK(2, i2);                                               \
        for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
        {                                                                       \
           Pragma(omp simd HYPRE_BOX_REDUCTION)                                 \
           for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
           {

#define hypre_BoxLoop2ReductionEndSimd(i1, i2, reducesum)                       \
              i1++;                                                             \
              i2++;                                                             \
           }                                                                    \
           zypre_BoxLoopInc1();                                                 \
           i1 += hypre__ikinc1[hypre__d];                                       \
           i2 += hypre__ikinc2[hypre__d];                                       \
           zypre_BoxLoopInc2();                                                 \
        }                                                                       \
     };                                                                         \
     executeReductionLoop(reducesum, SIMD_LEN);                                 \
}

/********************************************************************************************************************************************

 Adaptive parallelism loops ends

*********************************************************************************************************************************************/
#endif




/********************************************************************************************************************************************

 Common code

*********************************************************************************************************************************************/

#define hypre_LoopBegin(size,idx)                                             \
{                                                                             \
   HYPRE_Int idx;                                                             \
   for (idx = 0;idx < size;idx ++)                                            \
   {

#define hypre_LoopEnd()                                                       \
  }                                                                           \
}



//Not changing hypre_BoxLoop1ReductionBegin for now. Its used only during setup.
#define hypre_BoxLoop1ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, reducesum) \
{                                                                             \
   HYPRE_Int i1;                                                              \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   /*OMP1    */                                                                   \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
   {                                                                          \
      HYPRE_Int i1;                                                           \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {


#define hypre_BoxLoop1ReductionEnd(i1, reducesum)                             \
            i1 += hypre__i0inc1;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}

/* Reduction */

//#define hypre_BoxLoop1ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, reducesum) \
//        hypre_BoxLoop1Begin(ndim, loop_size, dbox1, start1, stride1, i1)
//
//#define hypre_BoxLoop1ReductionEnd(i1, reducesum) \
//        hypre_BoxLoop1End(i1)
//
//#define hypre_BoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, \
//                                                      dbox2, start2, stride2, i2, reducesum) \
//        hypre_BoxLoop2Begin(ndim, loop_size, dbox1, start1, stride1, i1, \
//                                             dbox2, start2, stride2, i2)
//
//#define hypre_BoxLoop2ReductionEnd(i1, i2, reducesum) \
//        hypre_BoxLoop2End(i1, i2)



#define hypre_newBoxLoopGetIndex zypre_BoxLoopGetIndex
#define hypre_BoxLoopGetIndex    zypre_BoxLoopGetIndex
#define hypre_BoxLoopSetOneBlock zypre_BoxLoopSetOneBlock
#define hypre_BoxLoopBlock       zypre_BoxLoopBlock
#define hypre_BoxLoop0Begin      zypre_newBoxLoop0Begin
#define hypre_BoxLoop0End        zypre_newBoxLoop0End
#define hypre_BoxLoop1Begin      zypre_newBoxLoop1Begin
#define hypre_BoxLoop1End        zypre_newBoxLoop1End
#define hypre_BoxLoop2Begin      zypre_newBoxLoop2Begin
#define hypre_BoxLoop2End        zypre_newBoxLoop2End
#define hypre_BoxLoop3Begin      zypre_newBoxLoop3Begin
#define hypre_BoxLoop3End        zypre_newBoxLoop3End
#define hypre_BoxLoop4Begin      zypre_newBoxLoop4Begin
#define hypre_BoxLoop4End        zypre_newBoxLoop4End
#define hypre_BasicBoxLoop2Begin zypre_newBasicBoxLoop2Begin
#define hypre_BasicBoxLoop2End   zypre_newBasicBoxLoop2End

#ifdef USE_SIMD
#define hypre_BoxLoop1BeginSimd      zypre_newBoxLoop1BeginSimd
#define hypre_BoxLoop1EndSimd        zypre_newBoxLoop1EndSimd
#define hypre_BoxLoop2BeginSimd      zypre_newBoxLoop2BeginSimd
#define hypre_BoxLoop2EndSimd        zypre_newBoxLoop2EndSimd
#define hypre_BoxLoop3BeginSimd      zypre_newBoxLoop3BeginSimd
#define hypre_BoxLoop3EndSimd        zypre_newBoxLoop3EndSimd
#define hypre_BasicBoxLoop2BeginSimd zypre_newBasicBoxLoop2BeginSimd
#define hypre_BasicBoxLoop2EndSimd   zypre_newBasicBoxLoop2EndSimd
#else
#define hypre_BoxLoop1BeginSimd      zypre_newBoxLoop1Begin
#define hypre_BoxLoop1EndSimd        zypre_newBoxLoop1End
#define hypre_BoxLoop2BeginSimd      zypre_newBoxLoop2Begin
#define hypre_BoxLoop2EndSimd        zypre_newBoxLoop2End
#define hypre_BoxLoop3BeginSimd      zypre_newBoxLoop3Begin
#define hypre_BoxLoop3EndSimd        zypre_newBoxLoop3End
#define hypre_BasicBoxLoop2BeginSimd zypre_newBasicBoxLoop2Begin
#define hypre_BasicBoxLoop2EndSimd   zypre_newBasicBoxLoop2End
#endif


#endif
