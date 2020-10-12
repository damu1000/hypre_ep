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
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include <utility>

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputeInfoCreate( hypre_CommInfo       *comm_info,
                         hypre_BoxArrayArray  *indt_boxes,
                         hypre_BoxArrayArray  *dept_boxes,
                         hypre_ComputeInfo   **compute_info_ptr )
{
   hypre_ComputeInfo  *compute_info;

   compute_info = hypre_TAlloc(hypre_ComputeInfo,  1, HYPRE_MEMORY_HOST);

   hypre_ComputeInfoCommInfo(compute_info)  = comm_info;
   hypre_ComputeInfoIndtBoxes(compute_info) = indt_boxes;
   hypre_ComputeInfoDeptBoxes(compute_info) = dept_boxes;

   hypre_SetIndex(hypre_ComputeInfoStride(compute_info), 1);

   *compute_info_ptr = compute_info;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputeInfoProjectSend( hypre_ComputeInfo  *compute_info,
                              hypre_Index         index,
                              hypre_Index         stride )
{
   hypre_CommInfoProjectSend(hypre_ComputeInfoCommInfo(compute_info),
                             index, stride);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputeInfoProjectRecv( hypre_ComputeInfo  *compute_info,
                              hypre_Index         index,
                              hypre_Index         stride )
{
   hypre_CommInfoProjectRecv(hypre_ComputeInfoCommInfo(compute_info),
                             index, stride);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputeInfoProjectComp( hypre_ComputeInfo  *compute_info,
                              hypre_Index         index,
                              hypre_Index         stride )
{
   hypre_ProjectBoxArrayArray(hypre_ComputeInfoIndtBoxes(compute_info),
                              index, stride);
   hypre_ProjectBoxArrayArray(hypre_ComputeInfoDeptBoxes(compute_info),
                              index, stride);
   hypre_CopyIndex(stride, hypre_ComputeInfoStride(compute_info));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputeInfoDestroy( hypre_ComputeInfo  *compute_info )
{
   hypre_TFree(compute_info, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return descriptions of communications and computations patterns for
 * a given grid-stencil computation.  If HYPRE\_OVERLAP\_COMM\_COMP is
 * defined, then the patterns are computed to allow for overlapping
 * communications and computations.  The default is no overlap.
 *
 * Note: This routine assumes that the grid boxes do not overlap.
 *--------------------------------------------------------------------------*/

extern __thread int g_overlap_comm;



HYPRE_Int
hypre_CreateMap( hypre_StructGrid      *grid,
		         hypre_ComputeInfo     *compute_info,
				 hypre_CommPkg         *comm_pkg)
{
   hypre_CommInfo          *comm_info = hypre_ComputeInfoCommInfo(compute_info);
   HYPRE_Int                ndim = hypre_StructGridNDim(grid);
   hypre_BoxArray          *boxes;
   HYPRE_Int                i, myid, num_cboxes;
   hypre_BoxArrayArray     *recv_boxes;
   hypre_BoxArray          *recv_box_array;
   HYPRE_Int              **recv_procs;
   MPI_Comm                 comm;
   HYPRE_Int               *ext_deps;
   hypre_Box            * rbox;

   comm   = hypre_StructGridComm(grid);
   hypre_MPI_Comm_rank(comm, &myid);
   /*------------------------------------------------------
    * Extract needed grid info
    *------------------------------------------------------*/

   boxes = hypre_StructGridBoxes(grid);

   /*------------------------------------------------------
    * Get communication info
    *------------------------------------------------------*/

   recv_boxes = hypre_CommInfoRecvBoxes(comm_info);
   recv_procs = hypre_CommInfoRecvProcesses(comm_info);

   /*------------------------------------------------------
    * Create proc to box map and count external dependencies
    *------------------------------------------------------*/

   ext_deps   = hypre_CTAlloc(HYPRE_Int, hypre_BoxArraySize(boxes), HYPRE_MEMORY_HOST);

   for(int i=0; i<hypre_BoxArraySize(boxes); i++)
	   ext_deps[i] = 0;

   hypre_ForBoxI(i, boxes)
   {
	  //iterate over all the recv_procs for box i and create proc to box mapping and count number of external dependencies.
	  recv_box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
	  num_cboxes = hypre_BoxArraySize(recv_box_array);
	  for (int m = 0; m < num_cboxes; m++){
		  if(recv_procs[i][m]!=myid){ //external dependency. Set independent to 0 and break
			  rbox = hypre_BoxArrayBox(recv_box_array, m);
			  auto range = hypre_CommPkgProcToBoxMap(comm_pkg).equal_range(recv_procs[i][m]);
			  int found = 0;
			  for (auto b = range.first; b != range.second; ++b){
				  int boxid = b->second;
				  if(i==boxid){
					  found=1;
					  break;
				  }
			  }

			  if(found==0 && hypre_BoxVolume(rbox) != 0){//do not add dup boxes. Do not add the box if recv volume is 0.
				  hypre_CommPkgProcToBoxMap(comm_pkg).insert(std::make_pair(recv_procs[i][m], i));
				  ext_deps[i]++;	//count num of extl deps
			  }
		  }
	  }
   }


   //There can be few boxes in dept_boxes array with incoming volume 0 (after coarsening??)
   //As a result, extl deps for such boxes might become 0. Move these boxes to indt_boxes.
   hypre_BoxArrayArray *dept_boxes = hypre_ComputeInfoDeptBoxes(compute_info);
   hypre_BoxArrayArray *indt_boxes = hypre_ComputeInfoIndtBoxes(compute_info);
   hypre_BoxArray      *indt_boxes_array, *dept_boxes_array;

   hypre_ForBoxArrayI(i, dept_boxes)
   {
	   if(ext_deps[i]==0){
		   indt_boxes_array = hypre_BoxArrayArrayBoxArray(indt_boxes, i);
		   dept_boxes_array = hypre_BoxArrayArrayBoxArray(dept_boxes, i);
		   if(hypre_BoxArraySize(dept_boxes_array)>0){
			   hypre_BoxArraySetSize(indt_boxes_array, 1);
			   hypre_CopyBox(hypre_BoxArrayBox(dept_boxes_array, 0), hypre_BoxArrayBox(indt_boxes_array, 0));
			   hypre_BoxArraySetSize(dept_boxes_array, 0);
		   }
	   }
   }

   hypre_CommPkgExtDeps(comm_pkg) = ext_deps;
   return hypre_error_flag;
}



HYPRE_Int
hypre_CreateComputeInfo( hypre_StructGrid      *grid,
                         hypre_StructStencil   *stencil,
                         hypre_ComputeInfo    **compute_info_ptr )
{
   HYPRE_Int                ndim = hypre_StructGridNDim(grid);
   hypre_CommInfo          *comm_info;
   hypre_BoxArrayArray     *indt_boxes;
   hypre_BoxArrayArray     *dept_boxes;

   hypre_BoxArray          *boxes;

   hypre_BoxArray          *cbox_array;
   hypre_Box               *cbox, *box;

   HYPRE_Int                i, num_cboxes, myid;
   hypre_BoxArrayArray     *recv_boxes;
   hypre_BoxArray          *recv_box_array;
   HYPRE_Int              **recv_procs;
   MPI_Comm                 comm;

   comm   = hypre_StructGridComm(grid);
   hypre_MPI_Comm_rank(comm, &myid);
   /*------------------------------------------------------
    * Extract needed grid info
    *------------------------------------------------------*/

   boxes = hypre_StructGridBoxes(grid);

   /*------------------------------------------------------
    * Get communication info
    *------------------------------------------------------*/

   hypre_CreateCommInfoFromStencil(grid, stencil, &comm_info);
   recv_boxes = hypre_CommInfoRecvBoxes(comm_info);
   recv_procs = hypre_CommInfoRecvProcesses(comm_info);

   /*------------------------------------------------------
    * Set up the independent & dependent boxes
    *------------------------------------------------------*/
   indt_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes), ndim);
   dept_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes), ndim);

   hypre_ForBoxI(i, boxes)
   {
	  box = hypre_BoxArrayBox(boxes, i); //box is a local box

	  //iterate over all the recv_procs for box i. If all recv_procs == myid, then no MPI dependency. Add into independent boxes
	  recv_box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
	  num_cboxes = hypre_BoxArraySize(recv_box_array);
	  int independent = 1;
	  for (int m = 0; m < num_cboxes; m++){
		  if(recv_procs[i][m]!=myid){ //external dependency. Set independent to 0 and break
			  independent = 0;
			  break;
		  }
	  }

	  if(independent==1 /*&& g_overlap_comm==1*/)//add into independent array
		  cbox_array = hypre_BoxArrayArrayBoxArray(indt_boxes, i);
	  else //add into dependent array
		  cbox_array = hypre_BoxArrayArrayBoxArray(dept_boxes, i);

	  hypre_BoxArraySetSize(cbox_array, 1);
	  cbox = hypre_BoxArrayBox(cbox_array, 0);
	  hypre_CopyBox(hypre_BoxArrayBox(boxes, i), cbox);
   }

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   hypre_ComputeInfoCreate(comm_info, indt_boxes, dept_boxes,
                           compute_info_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Create a computation package from a grid-based description of a
 * communication-computation pattern.
 *
 * Note: The input boxes and processes are destroyed.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputePkgCreate( hypre_ComputeInfo     *compute_info,
                        hypre_BoxArray        *data_space,
                        HYPRE_Int              num_values,
                        hypre_StructGrid      *grid,
                        hypre_ComputePkg     **compute_pkg_ptr )
{
   hypre_ComputePkg  *compute_pkg;
   hypre_CommPkg     *comm_pkg;

   compute_pkg = hypre_CTAlloc(hypre_ComputePkg,  1, HYPRE_MEMORY_HOST);

   hypre_CommPkgCreate(hypre_ComputeInfoCommInfo(compute_info),
                       data_space, data_space, num_values, NULL, 0,
                       hypre_StructGridComm(grid), &comm_pkg);

   hypre_CreateMap(grid, compute_info, comm_pkg);
   hypre_CommPkgNumOfBoxes(comm_pkg)   = hypre_BoxArraySize(hypre_StructGridBoxes(grid));

   hypre_CommInfoDestroy(hypre_ComputeInfoCommInfo(compute_info));
   hypre_ComputePkgCommPkg(compute_pkg) = comm_pkg;

   hypre_ComputePkgIndtBoxes(compute_pkg) = 
      hypre_ComputeInfoIndtBoxes(compute_info);
   hypre_ComputePkgDeptBoxes(compute_pkg) =
      hypre_ComputeInfoDeptBoxes(compute_info);
   hypre_CopyIndex(hypre_ComputeInfoStride(compute_info),
                   hypre_ComputePkgStride(compute_pkg));

   hypre_StructGridRef(grid, &hypre_ComputePkgGrid(compute_pkg));
   hypre_ComputePkgDataSpace(compute_pkg) = data_space;
   hypre_ComputePkgNumValues(compute_pkg) = num_values;

   hypre_BoxArrayArray     *rolling_dept_boxes;
   hypre_BoxArray          *compute_box_a;
   HYPRE_Int i;

   rolling_dept_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(hypre_ComputePkgIndtBoxes(compute_pkg)), hypre_CommPkgNDim(comm_pkg));
   hypre_ComputePkgRollingDeptBoxes(compute_pkg) = rolling_dept_boxes;
   hypre_ForBoxArrayI(i, rolling_dept_boxes)
   {
      compute_box_a = hypre_BoxArrayArrayBoxArray(rolling_dept_boxes, i);
      hypre_BoxArraySetSize(compute_box_a, 1); //allocate memory for 1 box
      hypre_BoxArraySetSize(compute_box_a, 0); //keep it empty
   }

   hypre_ComputeInfoDestroy(compute_info);

   *compute_pkg_ptr = compute_pkg;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Destroy a computation package.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputePkgDestroy( hypre_ComputePkg *compute_pkg )
{
   if (compute_pkg)
   {
      hypre_CommPkgDestroy(hypre_ComputePkgCommPkg(compute_pkg));

      hypre_BoxArrayArrayDestroy(hypre_ComputePkgIndtBoxes(compute_pkg));
      hypre_BoxArrayArrayDestroy(hypre_ComputePkgDeptBoxes(compute_pkg));

      hypre_StructGridDestroy(hypre_ComputePkgGrid(compute_pkg));

      hypre_TFree(compute_pkg, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Initialize a non-blocking communication exchange.  The independent
 * computations may be done after a call to this routine, to allow for
 * overlap of communications and computations.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_InitializeIndtComputations( hypre_ComputePkg  *compute_pkg,
                                  HYPRE_Complex     *data,
                                  hypre_CommHandle **comm_handle_ptr )
{
   hypre_CommPkg *comm_pkg = hypre_ComputePkgCommPkg(compute_pkg);

   hypre_InitializeCommunication(comm_pkg, data, data, 0, 0, comm_handle_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Finalize a communication exchange.  The dependent computations may
 * be done after a call to this routine.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FinalizeIndtComputations( hypre_CommHandle *comm_handle )
{
   hypre_FinalizeCommunication(comm_handle );

   return hypre_error_flag;
}
