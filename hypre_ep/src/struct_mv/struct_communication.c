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

#include "_hypre_struct_mv.h"
#include <sys/time.h>

#define DEBUG 0

#if DEBUG
char       filename[255];
FILE      *file;
#endif


//#define USE_FUNNELLED_COMM
/*do not define USE_FUNNELLED_COMM here. use -DUSE_FUNNELLED_COMM compiler option instead*/

#ifdef USE_FUNNELLED_COMM
#include<atomic>
#include<omp.h>
#endif

/* These are device buffers needed to do MPI communication 
 * when the computations (BoxLoop) are excuted on device and the host memory
 * are not accessible from device
 * */
HYPRE_THREAD_LOCAL_EP HYPRE_Complex* global_recv_buffer = NULL;
HYPRE_THREAD_LOCAL_EP HYPRE_Complex* global_send_buffer = NULL;
HYPRE_THREAD_LOCAL_EP HYPRE_Int      global_recv_size = 0;
HYPRE_THREAD_LOCAL_EP HYPRE_Int      global_send_size = 0;
thread_local HYPRE_Int commHandleAllocated = 0;

/* this computes a (large enough) size (in doubles) for the message prefix */
#define hypre_CommPrefixSize(ne)                                        \
   ( (((1+ne)*sizeof(HYPRE_Int) + ne*sizeof(hypre_Box))/sizeof(HYPRE_Complex)) + 1 )

/*--------------------------------------------------------------------------
 * Create a communication package.  A grid-based description of a communication
 * exchange is passed in.  This description is then compiled into an
 * intermediate processor-based description of the communication.  The
 * intermediate processor-based description is used directly to pack and unpack
 * buffers during the communications.
 *
 * The 'orders' argument is dimension 'num_transforms' x 'num_values' and should
 * have a one-to-one correspondence with the transform data in 'comm_info'.
 *
 * If 'reverse' is > 0, then the meaning of send/recv is reversed
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommPkgCreate( hypre_CommInfo   *comm_info,
                     hypre_BoxArray   *send_data_space,
                     hypre_BoxArray   *recv_data_space,
                     HYPRE_Int         num_values,
                     HYPRE_Int       **orders,
                     HYPRE_Int         reverse,
                     MPI_Comm          comm,
                     hypre_CommPkg   **comm_pkg_ptr )
{
   HYPRE_Int             ndim = hypre_CommInfoNDim(comm_info);
   hypre_BoxArrayArray  *send_boxes;
   hypre_BoxArrayArray  *recv_boxes;
   hypre_IndexRef        send_stride;
   hypre_IndexRef        recv_stride;
   HYPRE_Int           **send_processes;
   HYPRE_Int           **recv_processes;
   HYPRE_Int           **send_rboxnums;
   hypre_BoxArrayArray  *send_rboxes;

   HYPRE_Int             num_transforms;
   hypre_Index          *coords;
   hypre_Index          *dirs;
   HYPRE_Int           **send_transforms;
   HYPRE_Int           **cp_orders;

   hypre_CommPkg        *comm_pkg;
   hypre_CommType       *comm_types;
   hypre_CommType       *comm_type;
   hypre_CommEntryType  *ct_entries;
   HYPRE_Int            *ct_rem_boxnums;
   hypre_Box            *ct_rem_boxes;
   HYPRE_Int            *comm_boxes_p, *comm_boxes_i, *comm_boxes_j;
   HYPRE_Int             num_boxes, num_entries, num_comms, comm_bufsize;

   hypre_BoxArray       *box_array;
   hypre_Box            *box;
   hypre_BoxArray       *rbox_array;
   hypre_Box            *data_box;
   HYPRE_Int            *data_offsets;
   HYPRE_Int             data_offset;
   hypre_IndexRef        send_coord, send_dir;
   HYPRE_Int            *send_order;

   HYPRE_Int             i, j, k, p, m, size, p_old, my_proc;

   /*------------------------------------------------------
    *------------------------------------------------------*/

   if (reverse > 0)
   {
      /* reverse the meaning of send and recv */
      send_boxes      = hypre_CommInfoRecvBoxes(comm_info);
      recv_boxes      = hypre_CommInfoSendBoxes(comm_info);
      send_stride     = hypre_CommInfoRecvStride(comm_info);
      recv_stride     = hypre_CommInfoSendStride(comm_info);
      send_processes  = hypre_CommInfoRecvProcesses(comm_info);
      recv_processes  = hypre_CommInfoSendProcesses(comm_info);
      send_rboxnums   = hypre_CommInfoRecvRBoxnums(comm_info);
      send_rboxes     = hypre_CommInfoRecvRBoxes(comm_info);
      send_transforms = hypre_CommInfoRecvTransforms(comm_info); /* may be NULL */

      box_array = send_data_space;
      send_data_space = recv_data_space;
      recv_data_space = box_array;
   }
   else
   {
      send_boxes      = hypre_CommInfoSendBoxes(comm_info);
      recv_boxes      = hypre_CommInfoRecvBoxes(comm_info);
      send_stride     = hypre_CommInfoSendStride(comm_info);
      recv_stride     = hypre_CommInfoRecvStride(comm_info);
      send_processes  = hypre_CommInfoSendProcesses(comm_info);
      recv_processes  = hypre_CommInfoRecvProcesses(comm_info);
      send_rboxnums   = hypre_CommInfoSendRBoxnums(comm_info);
      send_rboxes     = hypre_CommInfoSendRBoxes(comm_info);
      send_transforms = hypre_CommInfoSendTransforms(comm_info); /* may be NULL */
   }
   num_transforms = hypre_CommInfoNumTransforms(comm_info);
   coords         = hypre_CommInfoCoords(comm_info); /* may be NULL */
   dirs           = hypre_CommInfoDirs(comm_info);   /* may be NULL */

   hypre_MPI_Comm_rank(comm, &my_proc );

   /*------------------------------------------------------
    * Set up various entries in CommPkg
    *------------------------------------------------------*/

   comm_pkg = hypre_CTAlloc(hypre_CommPkg,  1, HYPRE_MEMORY_HOST);

   hypre_CommPkgComm(comm_pkg)      = comm;
   hypre_CommPkgFirstComm(comm_pkg) = 1;
   hypre_CommPkgNDim(comm_pkg)      = ndim;
   hypre_CommPkgNumValues(comm_pkg) = num_values;
   hypre_CommPkgNumOrders(comm_pkg) = 0;
   hypre_CommPkgOrders(comm_pkg)    = NULL;
   if ( (send_transforms != NULL) && (orders != NULL) )
   {
      hypre_CommPkgNumOrders(comm_pkg) = num_transforms;
      cp_orders = hypre_TAlloc(HYPRE_Int *,  num_transforms, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_transforms; i++)
      {
         cp_orders[i] = hypre_TAlloc(HYPRE_Int,  num_values, HYPRE_MEMORY_HOST);
         for (j = 0; j < num_values; j++)
         {
            cp_orders[i][j] = orders[i][j];
         }
      }
      hypre_CommPkgOrders(comm_pkg) = cp_orders;
   }
   hypre_CopyIndex(send_stride, hypre_CommPkgSendStride(comm_pkg));
   hypre_CopyIndex(recv_stride, hypre_CommPkgRecvStride(comm_pkg));

   /* set identity transform and send_coord/dir/order if needed below */
   hypre_CommPkgIdentityOrder(comm_pkg) = hypre_TAlloc(HYPRE_Int,  num_values, HYPRE_MEMORY_HOST);
   send_coord = hypre_CommPkgIdentityCoord(comm_pkg);
   send_dir   = hypre_CommPkgIdentityDir(comm_pkg);
   send_order = hypre_CommPkgIdentityOrder(comm_pkg);
   for (i = 0; i < ndim; i++)
   {
      hypre_IndexD(send_coord, i) = i;
      hypre_IndexD(send_dir, i) = 1;
   }
   for (i = 0; i < num_values; i++)
   {
      send_order[i] = i;
   }

   /*------------------------------------------------------
    * Set up send CommType information
    *------------------------------------------------------*/

   /* set data_offsets and compute num_boxes, num_entries */
   data_offsets = hypre_TAlloc(HYPRE_Int,  hypre_BoxArraySize(send_data_space), HYPRE_MEMORY_HOST);
   data_offset = 0;
   num_boxes = 0;
   num_entries = 0;
   hypre_ForBoxI(i, send_data_space)
   {
      data_offsets[i] = data_offset;
      data_box = hypre_BoxArrayBox(send_data_space, i);
      data_offset += hypre_BoxVolume(data_box) * num_values;

      /* RDF: This should always be true, but it's not for FAC.  Find out why. */
      if (i < hypre_BoxArrayArraySize(send_boxes))
      {
         box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
         num_boxes += hypre_BoxArraySize(box_array);
         hypre_ForBoxI(j, box_array)
         {
            box = hypre_BoxArrayBox(box_array, j);
            if (hypre_BoxVolume(box) != 0)
            {
               num_entries++;
            }
         }
      }
   }

   /* set up comm_boxes_[pij] */
   comm_boxes_p = hypre_TAlloc(HYPRE_Int,  num_boxes, HYPRE_MEMORY_HOST);
   comm_boxes_i = hypre_TAlloc(HYPRE_Int,  num_boxes, HYPRE_MEMORY_HOST);
   comm_boxes_j = hypre_TAlloc(HYPRE_Int,  num_boxes, HYPRE_MEMORY_HOST);
   num_boxes = 0;
   hypre_ForBoxArrayI(i, send_boxes)
   {
      box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
      hypre_ForBoxI(j, box_array)
      {
         comm_boxes_p[num_boxes] = send_processes[i][j];
         comm_boxes_i[num_boxes] = i;
         comm_boxes_j[num_boxes] = j;
         num_boxes++;
      }
   }
   hypre_qsort3i(comm_boxes_p, comm_boxes_i, comm_boxes_j, 0, num_boxes-1);

   /* compute comm_types */

   /* make sure there is at least 1 comm_type allocated */
   comm_types = hypre_CTAlloc(hypre_CommType,  (num_boxes + 1), HYPRE_MEMORY_HOST);
   ct_entries = hypre_TAlloc(hypre_CommEntryType,  num_entries, HYPRE_MEMORY_HOST);
   ct_rem_boxnums = hypre_TAlloc(HYPRE_Int,  num_entries, HYPRE_MEMORY_HOST);
   ct_rem_boxes = hypre_TAlloc(hypre_Box,  num_entries, HYPRE_MEMORY_HOST);
   hypre_CommPkgEntries(comm_pkg)    = ct_entries;
   hypre_CommPkgRemBoxnums(comm_pkg) = ct_rem_boxnums;
   hypre_CommPkgRemBoxes(comm_pkg)   = ct_rem_boxes;

   p_old = -1;
   num_comms = 0;
   comm_bufsize = 0;
   for (m = 0; m < num_boxes; m++)
   {
      i = comm_boxes_i[m];
      j = comm_boxes_j[m];
      box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
      box = hypre_BoxArrayBox(box_array, j);

      if (hypre_BoxVolume(box) != 0)
      {
         p = comm_boxes_p[m];

         /* start a new comm_type */
         if (p != p_old)
         {
            if (p != my_proc)
            {
               comm_type = &comm_types[num_comms+1];
               num_comms++;
            }
            else
            {
               comm_type = &comm_types[0];
            }
            hypre_CommTypeProc(comm_type)       = p;
            hypre_CommTypeBufsize(comm_type)    = 0;
            hypre_CommTypeNumEntries(comm_type) = 0;
            hypre_CommTypeEntries(comm_type)    = ct_entries;
            hypre_CommTypeRemBoxnums(comm_type) = ct_rem_boxnums;
            hypre_CommTypeRemBoxes(comm_type)   = ct_rem_boxes;
            p_old = p;
         }

         k = hypre_CommTypeNumEntries(comm_type);
         hypre_BoxGetStrideVolume(box, send_stride, &size);
         hypre_CommTypeBufsize(comm_type) += (size*num_values);
         comm_bufsize                     += (size*num_values);
         rbox_array = hypre_BoxArrayArrayBoxArray(send_rboxes, i);
         data_box = hypre_BoxArrayBox(send_data_space, i);
         if (send_transforms != NULL)
         {
            send_coord = coords[send_transforms[i][j]];
            send_dir   = dirs[send_transforms[i][j]];
            if (orders != NULL)
            {
               send_order = cp_orders[send_transforms[i][j]];
            }
         }
         hypre_CommTypeSetEntry(box, send_stride, send_coord, send_dir,
                                send_order, data_box, data_offsets[i],
                                hypre_CommTypeEntry(comm_type, k));
         hypre_CommTypeRemBoxnum(comm_type, k) = send_rboxnums[i][j];
         hypre_CopyBox(hypre_BoxArrayBox(rbox_array, j),
                       hypre_CommTypeRemBox(comm_type, k));
         hypre_CommTypeNumEntries(comm_type) ++;
         ct_entries     ++;
         ct_rem_boxnums ++;
         ct_rem_boxes   ++;
      }
   }

   /* add space for prefix info */
   for (m = 1; m < (num_comms + 1); m++)
   {
      comm_type = &comm_types[m];
      k = hypre_CommTypeNumEntries(comm_type);
      size = hypre_CommPrefixSize(k);
      hypre_CommTypeBufsize(comm_type) += size;
      comm_bufsize                     += size;
   }

   /* set send info in comm_pkg */
   comm_types = hypre_TReAlloc(comm_types,  hypre_CommType,  (num_comms + 1), HYPRE_MEMORY_HOST);
   hypre_CommPkgSendBufsize(comm_pkg)  = comm_bufsize;
   hypre_CommPkgNumSends(comm_pkg)     = num_comms;
   hypre_CommPkgSendTypes(comm_pkg)    = &comm_types[1];
   hypre_CommPkgCopyFromType(comm_pkg) = &comm_types[0];

   /* free up data_offsets */
   hypre_TFree(data_offsets, HYPRE_MEMORY_HOST);

   /*------------------------------------------------------
    * Set up recv CommType information
    *------------------------------------------------------*/

   /* set data_offsets and compute num_boxes */
   data_offsets = hypre_TAlloc(HYPRE_Int,  hypre_BoxArraySize(recv_data_space), HYPRE_MEMORY_HOST);
   data_offset = 0;
   num_boxes = 0;
   hypre_ForBoxI(i, recv_data_space)
   {
      data_offsets[i] = data_offset;
      data_box = hypre_BoxArrayBox(recv_data_space, i);
      data_offset += hypre_BoxVolume(data_box) * num_values;

      /* RDF: This should always be true, but it's not for FAC.  Find out why. */
      if (i < hypre_BoxArrayArraySize(recv_boxes))
      {
         box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
         num_boxes += hypre_BoxArraySize(box_array);
      }
   }
   hypre_CommPkgRecvDataOffsets(comm_pkg) = data_offsets;
   hypre_CommPkgRecvDataSpace(comm_pkg) = hypre_BoxArrayDuplicate(recv_data_space);

   /* set up comm_boxes_[pij] */
   comm_boxes_p = hypre_TReAlloc(comm_boxes_p,  HYPRE_Int,  num_boxes, HYPRE_MEMORY_HOST);
   comm_boxes_i = hypre_TReAlloc(comm_boxes_i,  HYPRE_Int,  num_boxes, HYPRE_MEMORY_HOST);
   comm_boxes_j = hypre_TReAlloc(comm_boxes_j,  HYPRE_Int,  num_boxes, HYPRE_MEMORY_HOST);
   num_boxes = 0;
   hypre_ForBoxArrayI(i, recv_boxes)
   {
      box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
      hypre_ForBoxI(j, box_array)
      {
         comm_boxes_p[num_boxes] = recv_processes[i][j];
         comm_boxes_i[num_boxes] = i;
         comm_boxes_j[num_boxes] = j;
         num_boxes++;
      }
   }
   hypre_qsort3i(comm_boxes_p, comm_boxes_i, comm_boxes_j, 0, num_boxes-1);

   /* compute comm_types */

   /* make sure there is at least 1 comm_type allocated */
   comm_types = hypre_CTAlloc(hypre_CommType,  (num_boxes + 1), HYPRE_MEMORY_HOST);

   p_old = -1;
   num_comms = 0;
   comm_bufsize = 0;
   for (m = 0; m < num_boxes; m++)
   {
      i = comm_boxes_i[m];
      j = comm_boxes_j[m];
      box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
      box = hypre_BoxArrayBox(box_array, j);

      if (hypre_BoxVolume(box) != 0)
      {
         p = comm_boxes_p[m];

         /* start a new comm_type */
         if (p != p_old)
         {
            if (p != my_proc)
            {
               comm_type = &comm_types[num_comms+1];
               num_comms++;
            }
            else
            {
               comm_type = &comm_types[0];
            }
            hypre_CommTypeProc(comm_type)       = p;
            hypre_CommTypeBufsize(comm_type)    = 0;
            hypre_CommTypeNumEntries(comm_type) = 0;
            p_old = p;
         }

         k = hypre_CommTypeNumEntries(comm_type);
         hypre_BoxGetStrideVolume(box, recv_stride, &size);
         hypre_CommTypeBufsize(comm_type) += (size*num_values);
         comm_bufsize                     += (size*num_values);
         hypre_CommTypeNumEntries(comm_type) ++;
      }
   }

   /* add space for prefix info */
   for (m = 1; m < (num_comms + 1); m++)
   {
      comm_type = &comm_types[m];
      k = hypre_CommTypeNumEntries(comm_type);
      size = hypre_CommPrefixSize(k);
      hypre_CommTypeBufsize(comm_type) += size;
      comm_bufsize                     += size;
   }

   /* set recv info in comm_pkg */
   comm_types = hypre_TReAlloc(comm_types,  hypre_CommType,  (num_comms + 1), HYPRE_MEMORY_HOST);
   hypre_CommPkgRecvBufsize(comm_pkg) = comm_bufsize;
   hypre_CommPkgNumRecvs(comm_pkg)    = num_comms;
   hypre_CommPkgRecvTypes(comm_pkg)   = &comm_types[1];
   hypre_CommPkgCopyToType(comm_pkg)  = &comm_types[0];

   /* if CommInfo send/recv boxes don't match, compute a max bufsize */
   if ( !hypre_CommInfoBoxesMatch(comm_info) )
   {
      hypre_CommPkgRecvBufsize(comm_pkg) = 0;
      for (i = 0; i < hypre_CommPkgNumRecvs(comm_pkg); i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i);

         /* subtract off old (incorrect) prefix size */
         num_entries = hypre_CommTypeNumEntries(comm_type);
         hypre_CommTypeBufsize(comm_type) -= hypre_CommPrefixSize(num_entries);

         /* set num_entries to number of grid points and add new prefix size */
         num_entries = hypre_CommTypeBufsize(comm_type);
         hypre_CommTypeNumEntries(comm_type) = num_entries;
         size = hypre_CommPrefixSize(num_entries);
         hypre_CommTypeBufsize(comm_type) += size;
         hypre_CommPkgRecvBufsize(comm_pkg) += hypre_CommTypeBufsize(comm_type);
      }
   }

   /*------------------------------------------------------
    * Debugging stuff - ONLY WORKS FOR 3D
    *------------------------------------------------------*/

#if DEBUG
   {
      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &my_proc);

      hypre_sprintf(filename, "zcommboxes.%05d", my_proc);

      if ((file = fopen(filename, "a")) == NULL)
      {
         hypre_printf("Error: can't open output file %s\n", filename);
         exit(1);
      }

      hypre_fprintf(file, "\n\n============================\n\n");
      hypre_fprintf(file, "SEND boxes:\n\n");

      hypre_fprintf(file, "Stride = (%d,%d,%d)\n",
                    hypre_IndexD(send_stride, 0),
                    hypre_IndexD(send_stride, 1),
                    hypre_IndexD(send_stride, 2));
      hypre_fprintf(file, "BoxArrayArraySize = %d\n",
                    hypre_BoxArrayArraySize(send_boxes));
      hypre_ForBoxArrayI(i, send_boxes)
      {
         box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);

         hypre_fprintf(file, "BoxArraySize = %d\n", hypre_BoxArraySize(box_array));
         hypre_ForBoxI(j, box_array)
         {
            box = hypre_BoxArrayBox(box_array, j);
            hypre_fprintf(file, "(%d,%d): (%d,%d,%d) x (%d,%d,%d)\n",
                          i, j,
                          hypre_BoxIMinD(box, 0),
                          hypre_BoxIMinD(box, 1),
                          hypre_BoxIMinD(box, 2),
                          hypre_BoxIMaxD(box, 0),
                          hypre_BoxIMaxD(box, 1),
                          hypre_BoxIMaxD(box, 2));
            hypre_fprintf(file, "(%d,%d): %d,%d\n",
                          i, j, send_processes[i][j], send_rboxnums[i][j]);
         }
      }

      hypre_fprintf(file, "\n\n============================\n\n");
      hypre_fprintf(file, "RECV boxes:\n\n");

      hypre_fprintf(file, "Stride = (%d,%d,%d)\n",
                    hypre_IndexD(recv_stride, 0),
                    hypre_IndexD(recv_stride, 1),
                    hypre_IndexD(recv_stride, 2));
      hypre_fprintf(file, "BoxArrayArraySize = %d\n",
                    hypre_BoxArrayArraySize(recv_boxes));
      hypre_ForBoxArrayI(i, recv_boxes)
      {
         box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);

         hypre_fprintf(file, "BoxArraySize = %d\n", hypre_BoxArraySize(box_array));
         hypre_ForBoxI(j, box_array)
         {
            box = hypre_BoxArrayBox(box_array, j);
            hypre_fprintf(file, "(%d,%d): (%d,%d,%d) x (%d,%d,%d)\n",
                          i, j,
                          hypre_BoxIMinD(box, 0),
                          hypre_BoxIMinD(box, 1),
                          hypre_BoxIMinD(box, 2),
                          hypre_BoxIMaxD(box, 0),
                          hypre_BoxIMaxD(box, 1),
                          hypre_BoxIMaxD(box, 2));
            hypre_fprintf(file, "(%d,%d): %d\n",
                          i, j, recv_processes[i][j]);
         }
      }

      fflush(file);
      fclose(file);
   }
#endif

#if DEBUG
   {
      hypre_CommEntryType  *comm_entry;
      HYPRE_Int             offset, dim;
      HYPRE_Int            *length;
      HYPRE_Int            *stride;

      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &my_proc);

      hypre_sprintf(filename, "zcommentries.%05d", my_proc);

      if ((file = fopen(filename, "a")) == NULL)
      {
         hypre_printf("Error: can't open output file %s\n", filename);
         exit(1);
      }

      hypre_fprintf(file, "\n\n============================\n\n");
      hypre_fprintf(file, "SEND entries:\n\n");

      hypre_fprintf(file, "num_sends = %d\n", hypre_CommPkgNumSends(comm_pkg));

      comm_types = hypre_CommPkgCopyFromType(comm_pkg);
      for (m = 0; m < (hypre_CommPkgNumSends(comm_pkg) + 1); m++)
      {
         comm_type = &comm_types[m];
         hypre_fprintf(file, "process     = %d\n", hypre_CommTypeProc(comm_type));
         hypre_fprintf(file, "num_entries = %d\n", hypre_CommTypeNumEntries(comm_type));
         for (i = 0; i < hypre_CommTypeNumEntries(comm_type); i++)
         {
            comm_entry = hypre_CommTypeEntry(comm_type, i);
            offset = hypre_CommEntryTypeOffset(comm_entry);
            dim    = hypre_CommEntryTypeDim(comm_entry);
            length = hypre_CommEntryTypeLengthArray(comm_entry);
            stride = hypre_CommEntryTypeStrideArray(comm_entry);
            hypre_fprintf(file, "%d: %d,%d,(%d,%d,%d,%d),(%d,%d,%d,%d)\n",
                          i, offset, dim,
                          length[0], length[1], length[2], length[3],
                          stride[0], stride[1], stride[2], stride[3]);
         }
      }

      hypre_fprintf(file, "\n\n============================\n\n");
      hypre_fprintf(file, "RECV entries:\n\n");

      hypre_fprintf(file, "num_recvs = %d\n", hypre_CommPkgNumRecvs(comm_pkg));

      comm_types = hypre_CommPkgCopyToType(comm_pkg);

      comm_type = &comm_types[0];
      hypre_fprintf(file, "process     = %d\n", hypre_CommTypeProc(comm_type));
      hypre_fprintf(file, "num_entries = %d\n", hypre_CommTypeNumEntries(comm_type));
      for (i = 0; i < hypre_CommTypeNumEntries(comm_type); i++)
      {
         comm_entry = hypre_CommTypeEntry(comm_type, i);
         offset = hypre_CommEntryTypeOffset(comm_entry);
         dim    = hypre_CommEntryTypeDim(comm_entry);
         length = hypre_CommEntryTypeLengthArray(comm_entry);
         stride = hypre_CommEntryTypeStrideArray(comm_entry);
         hypre_fprintf(file, "%d: %d,%d,(%d,%d,%d,%d),(%d,%d,%d,%d)\n",
                       i, offset, dim,
                       length[0], length[1], length[2], length[3],
                       stride[0], stride[1], stride[2], stride[3]);
      }

      for (m = 1; m < (hypre_CommPkgNumRecvs(comm_pkg) + 1); m++)
      {
         comm_type = &comm_types[m];
         hypre_fprintf(file, "process     = %d\n", hypre_CommTypeProc(comm_type));
         hypre_fprintf(file, "num_entries = %d\n", hypre_CommTypeNumEntries(comm_type));
      }

      fflush(file);
      fclose(file);
   }
#endif

   /*------------------------------------------------------
    * Clean up
    *------------------------------------------------------*/

   hypre_TFree(comm_boxes_p, HYPRE_MEMORY_HOST);
   hypre_TFree(comm_boxes_i, HYPRE_MEMORY_HOST);
   hypre_TFree(comm_boxes_j, HYPRE_MEMORY_HOST);

   *comm_pkg_ptr = comm_pkg;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Note that this routine assumes an identity coordinate transform
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommTypeSetEntries( hypre_CommType  *comm_type,
                          HYPRE_Int       *boxnums,
                          hypre_Box       *boxes,
                          hypre_Index      stride,
                          hypre_Index      coord,
                          hypre_Index      dir,
                          HYPRE_Int       *order,
                          hypre_BoxArray  *data_space,
                          HYPRE_Int       *data_offsets )
{
   HYPRE_Int             num_entries = hypre_CommTypeNumEntries(comm_type);
   hypre_CommEntryType  *entries     = hypre_CommTypeEntries(comm_type);
   hypre_Box            *box;
   hypre_Box            *data_box;
   HYPRE_Int             i, j;

   for (j = 0; j < num_entries; j++)
   {
      i = boxnums[j];
      box = &boxes[j];
      data_box = hypre_BoxArrayBox(data_space, i);

      hypre_CommTypeSetEntry(box, stride, coord, dir, order,
                             data_box, data_offsets[i], &entries[j]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommTypeSetEntry( hypre_Box           *box,
                        hypre_Index          stride,
                        hypre_Index          coord,
                        hypre_Index          dir,
                        HYPRE_Int           *order,
                        hypre_Box           *data_box,
                        HYPRE_Int            data_box_offset,
                        hypre_CommEntryType *comm_entry )
{
   HYPRE_Int     dim, ndim = hypre_BoxNDim(box);
   HYPRE_Int     offset;
   HYPRE_Int    *length_array, tmp_length_array[HYPRE_MAXDIM];
   HYPRE_Int    *stride_array, tmp_stride_array[HYPRE_MAXDIM];
   hypre_Index   size;
   HYPRE_Int     i, j;

   length_array = hypre_CommEntryTypeLengthArray(comm_entry);
   stride_array = hypre_CommEntryTypeStrideArray(comm_entry);

   /* initialize offset */
   offset = data_box_offset + hypre_BoxIndexRank(data_box, hypre_BoxIMin(box));

   /* initialize length_array and stride_array */
   hypre_BoxGetStrideSize(box, stride, size);
   for (i = 0; i < ndim; i++)
   {
      length_array[i] = hypre_IndexD(size, i);
      stride_array[i] = hypre_IndexD(stride, i);
      for (j = 0; j < i; j++)
      {
         stride_array[i] *= hypre_BoxSizeD(data_box, j);
      }
   }
   stride_array[ndim] = hypre_BoxVolume(data_box);

   /* make adjustments for dir */
   for (i = 0; i < ndim; i++)
   {
      if (dir[i] < 0)
      {
         offset += (length_array[i] - 1)*stride_array[i];
         stride_array[i] = -stride_array[i];
      }
   }

   /* make adjustments for coord */
   for (i = 0; i < ndim; i++)
   {
      tmp_length_array[i] = length_array[i];
      tmp_stride_array[i] = stride_array[i];
   }
   for (i = 0; i < ndim; i++)
   {
      j = coord[i];
      length_array[j] = tmp_length_array[i];
      stride_array[j] = tmp_stride_array[i];
   }

   /* eliminate dimensions with length_array = 1 */
   dim = ndim;
   i = 0;
   while (i < dim)
   {
      if(length_array[i] == 1)
      {
         for(j = i; j < (dim - 1); j++)
         {
            length_array[j] = length_array[j+1];
            stride_array[j] = stride_array[j+1];
         }
         length_array[dim - 1] = 1;
         stride_array[dim - 1] = 1;
         dim--;
      }
      else
      {
         i++;
      }
   }

#if 0
   /* sort the array according to length_array (largest to smallest) */
   for (i = (dim-1); i > 0; i--)
   {
      for (j = 0; j < i; j++)
      {
         if (length_array[j] < length_array[j+1])
         {
            i_tmp             = length_array[j];
            length_array[j]   = length_array[j+1];
            length_array[j+1] = i_tmp;

            i_tmp             = stride_array[j];
            stride_array[j]   = stride_array[j+1];
            stride_array[j+1] = i_tmp;
         }
      }
   }
#endif

   /* if every len was 1 we need to fix to communicate at least one */
   if(!dim)
   {
      dim = 1;
   }

   hypre_CommEntryTypeOffset(comm_entry) = offset;
   hypre_CommEntryTypeDim(comm_entry) = dim;
   hypre_CommEntryTypeOrder(comm_entry) = order;

   return hypre_error_flag;
}





#ifdef USE_FUNNELLED_COMM

//to be used for funneling the requests through main thread during thread as rank approach.
// declared and allocated in hypre_set_num_threads in /src/utilities/mpistubs.c
typedef struct hypre_CommSendRecvReq
{
	volatile HYPRE_Int          req_status;	//0: new request, 1: posted, 2 completed.
	volatile HYPRE_Int          tag;
	volatile HYPRE_Int          num_recvs, num_sends;
	HYPRE_Complex      **recv_buffers;
	HYPRE_Complex      **send_buffers;
	hypre_CommPkg      *comm_pkg;
	hypre_MPI_Request  *requests;
	hypre_MPI_Status   *status;
	omp_lock_t mutex;

} hypre_CommSendRecvReq;


typedef struct hypre_CommStaging
{
	HYPRE_Complex volatile **staging;
	//omp_lock_t mutex;
} hypre_CommStaging;

volatile hypre_CommStaging *g_stg;


std::atomic<int> g_new_comm_req;
volatile hypre_CommSendRecvReq * g_comm_array;
volatile int g_funneled_comm_spin=1;
extern __thread int g_thread_id;
extern int g_num_of_threads;


void allocate_funneled_comm()
{
////	atomic_store_explicit (&g_new_comm_req, 0, memory_order_seq_cst);
	g_new_comm_req = 0;
	g_funneled_comm_spin = 1;
	g_comm_array = (hypre_CommSendRecvReq *) malloc(sizeof(hypre_CommSendRecvReq)*g_num_of_threads);
	g_stg = (volatile hypre_CommStaging *) malloc(sizeof(hypre_CommStaging)*g_num_of_threads);
	int i;
	for(i=0; i<g_num_of_threads; i++)
	{
		g_comm_array[i].req_status=-1;
		omp_init_lock(const_cast<omp_lock_t*>(&g_comm_array[i].mutex));

		g_stg[i].staging = (HYPRE_Complex volatile **) malloc(sizeof(HYPRE_Complex *)*g_num_of_threads);
		//omp_init_lock(&g_stg[i].mutex);

		int j;
		for(j=0; j<g_num_of_threads; j++)
			g_stg[i].staging[j] = NULL;
	}
}

void deallocate_funneled_comm()
{
	//printf("inside deallocate_funneled_comm\n");
	g_funneled_comm_spin = 0;

}
void funnelled_comm()
{
	int real_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &real_rank);	//this will give real rank, not fake rank manipulated in mpistubs.c

	while(g_funneled_comm_spin)
	{

		int t=0;

		//int placed_requests = 0;
		for(t=0; t<g_num_of_threads; t++)
		{
			if(g_comm_array[t].req_status==0 )//new request
			{
				//printf("new request for %d\n", t);
				int i=0, j = 0;
				omp_set_lock(const_cast<omp_lock_t*>(&g_comm_array[t].mutex));
				g_thread_id = t;
				MPI_Comm comm = hypre_CommPkgComm(g_comm_array[t].comm_pkg);

				for(i = 0; i < g_comm_array[t].num_recvs; i++)
				{
					hypre_CommType *comm_type = hypre_CommPkgRecvType(g_comm_array[t].comm_pkg, i);
					int src = hypre_CommTypeProc(comm_type);
					int real_src = src / g_num_of_threads;

					if(real_rank != real_src)	//if intra-thread comm
					{
						hypre_MPI_Irecv(g_comm_array[t].recv_buffers[i],
								hypre_CommTypeBufsize(comm_type)*sizeof(HYPRE_Complex),
								hypre_MPI_BYTE, hypre_CommTypeProc(comm_type),
								g_comm_array[t].tag, comm, &g_comm_array[t].requests[j++]);
						if ( hypre_CommPkgFirstComm(g_comm_array[t].comm_pkg) )
						{
							int size = hypre_CommPrefixSize(hypre_CommTypeNumEntries(comm_type));
							hypre_CommTypeBufsize(comm_type)   -= size;
							hypre_CommPkgRecvBufsize(g_comm_array[t].comm_pkg) -= size;
						}
					}
				}

				g_comm_array[t].num_recvs = j;

				for(i = 0; i < g_comm_array[t].num_sends; i++)
				{
					hypre_CommType *comm_type = hypre_CommPkgSendType(g_comm_array[t].comm_pkg, i);
					int dest = hypre_CommTypeProc(comm_type);
					int real_dest = dest / g_num_of_threads;
					if(real_rank != real_dest)	//if intra-thread comm
					{
						hypre_MPI_Isend(g_comm_array[t].send_buffers[i],
								hypre_CommTypeBufsize(comm_type)*sizeof(HYPRE_Complex),
								hypre_MPI_BYTE, hypre_CommTypeProc(comm_type),
								g_comm_array[t].tag, comm, &g_comm_array[t].requests[j++]);
						if ( hypre_CommPkgFirstComm(g_comm_array[t].comm_pkg) )
						{
							int size = hypre_CommPrefixSize(hypre_CommTypeNumEntries(comm_type));
							hypre_CommTypeBufsize(comm_type)   -= size;
							hypre_CommPkgSendBufsize(g_comm_array[t].comm_pkg) -= size;
						}
					}
				}

				g_comm_array[t].num_sends = j - g_comm_array[t].num_recvs;
				g_comm_array[t].req_status=1;
				//omp_unset_lock(&g_comm_array[t].mutex);
				//printf("requests placed for thread %d\n", t);
			}
		}


		//while there are no new requests, call mpi test to progress requests.
		//int t;
		for(t=0; t<g_num_of_threads; t++)
		{
			if(g_comm_array[t].req_status==1 /*&& g_comm_array[t].status != NULL  && g_comm_array[t].requests != NULL*/)	//request posted and status buffer is allocated
			{
				g_thread_id = t;
				int flag;
				//printf("testing for thread %d\n", t);
				MPI_Testall(g_comm_array[t].num_recvs + g_comm_array[t].num_sends, g_comm_array[t].requests, &flag, g_comm_array[t].status);
				//printf("tested for thread %d\n", t);
				if(flag)
				{
					//printf("comm over for %d\n", t);
					//omp_set_lock(&g_comm_array[t].mutex);
					g_comm_array[t].req_status = 2; //completed
					omp_unset_lock(const_cast<omp_lock_t*>(&g_comm_array[t].mutex));
				}

			}
		}
	}


	int i;
	for(i=0; i<g_num_of_threads; i++)
	{
		omp_destroy_lock(const_cast<omp_lock_t*>(&g_comm_array[i].mutex));
		//omp_destroy_lock(&g_stg[i].mutex);
		free(g_stg[i].staging);
	}

	if(g_stg) free((void*)g_stg);
	if(g_comm_array) free((void*)g_comm_array);
}


#endif



/*--------------------------------------------------------------------------
 * Initialize a non-blocking communication exchange.
 *
 * The communication buffers are created, the send buffer is manually
 * packed, and the communication requests are posted.
 *
 * Different "actions" are possible when the buffer data is unpacked:
 *   action = 0    - copy the data over existing values in memory
 *   action = 1    - add the data to existing values in memory
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_InitializeCommunication( hypre_CommPkg     *comm_pkg,
                               HYPRE_Complex     *send_data,
                               HYPRE_Complex     *recv_data,
                               HYPRE_Int          action,
                               HYPRE_Int          tag,
                               hypre_CommHandle **comm_handle_ptr )
{
   hypre_CommHandle    *comm_handle;

   HYPRE_Int            ndim       = hypre_CommPkgNDim(comm_pkg);
   HYPRE_Int            num_values = hypre_CommPkgNumValues(comm_pkg);
   HYPRE_Int            num_sends  = hypre_CommPkgNumSends(comm_pkg);
   HYPRE_Int            num_recvs  = hypre_CommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm       = hypre_CommPkgComm(comm_pkg);

   HYPRE_Int            num_requests;
   hypre_MPI_Request   *requests;
   hypre_MPI_Status    *status;
   HYPRE_Complex      **send_buffers;
   HYPRE_Complex      **recv_buffers;

   HYPRE_Complex      **send_buffers_data;
   HYPRE_Complex      **recv_buffers_data;

   hypre_CommType      *comm_type, *from_type, *to_type;
   hypre_CommEntryType *comm_entry;
   HYPRE_Int            num_entries;

   HYPRE_Int           *length_array;
   HYPRE_Int           *stride_array, unitst_array[HYPRE_MAXDIM+1];
   HYPRE_Int           *order;

   HYPRE_Complex       *dptr, *kptr, *lptr;
   HYPRE_Int           *qptr;

   HYPRE_Int            i, j, d, ll, k;
   HYPRE_Int            size;
   HYPRE_Int  *interthread_requests; //similar to MPI_Request. Store recv req handle for inter-thread comm.
   HYPRE_Int  *mpi_to_i, *interthread_to_i; //mapping to num_recvs index. Needed to get comm_type. Find out a better way.
   HYPRE_Int   num_mpi_sends, num_mpi_recvs, num_inter_thread_sends, num_inter_thread_recvs;

   /*--------------------------------------------------------------------
    * allocate requests and status
    *--------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests = hypre_CTAlloc(hypre_MPI_Request,  num_requests, HYPRE_MEMORY_HOST);
   status   = hypre_CTAlloc(hypre_MPI_Status,  num_requests, HYPRE_MEMORY_HOST);

   interthread_requests = hypre_CTAlloc(HYPRE_Int,  num_requests, HYPRE_MEMORY_HOST);
   mpi_to_i = hypre_CTAlloc(HYPRE_Int,  num_requests, HYPRE_MEMORY_HOST);
   interthread_to_i = hypre_CTAlloc(HYPRE_Int,  num_requests, HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------
    * allocate buffers
    *--------------------------------------------------------------------*/

   /* allocate send buffers */
   send_buffers = hypre_TAlloc(HYPRE_Complex *,  num_sends, HYPRE_MEMORY_HOST);
   if (num_sends > 0)
   {
      size = hypre_CommPkgSendBufsize(comm_pkg);
      send_buffers[0] =  hypre_TAlloc(HYPRE_Complex,  size, HYPRE_MEMORY_HOST);
      for (i = 1; i < num_sends; i++)
      {
         comm_type = hypre_CommPkgSendType(comm_pkg, i-1);
         size = hypre_CommTypeBufsize(comm_type);
         send_buffers[i] = send_buffers[i-1] + size;
      }
   }

   /* Prepare send buffers: allocate device buffer */
   HYPRE_Int alloc_dev_buffer = 0;
   /* In the case of running on device and cannot access host memory from device */
#if (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)) && (HYPRE_MEMORY_HOST_ACT != HYPRE_MEMORY_SHARED)
#if defined(HYPRE_USING_RAJA) || defined(HYPRE_USING_KOKKOS)
   alloc_dev_buffer = 1;
#elif defined(HYPRE_USING_CUDA)
   alloc_dev_buffer = (hypre_exec_policy == HYPRE_MEMORY_DEVICE);
#elif defined(HYPRE_USING_DEVICE_OPENMP)
   alloc_dev_buffer = hypre__global_offload;
#endif
#endif

   if (alloc_dev_buffer)
   {
      send_buffers_data = hypre_TAlloc(HYPRE_Complex *, num_sends,HYPRE_MEMORY_HOST);
      if (num_sends > 0)
      {
	 size = hypre_CommPkgSendBufsize(comm_pkg);
	 if (size > global_send_size)
	 {
	    if (global_send_size > 0)
	    {
	       hypre_TFree(global_send_buffer,HYPRE_MEMORY_DEVICE);
	    }
	    global_send_buffer = hypre_CTAlloc(HYPRE_Complex, 5*size,HYPRE_MEMORY_DEVICE);
	    global_send_size   = 5*size;
	 }
	 send_buffers_data[0] = global_send_buffer;
	 for (i = 1; i < num_sends; i++)
	 {
	    comm_type = hypre_CommPkgSendType(comm_pkg, i-1);
	    size = hypre_CommTypeBufsize(comm_type);
	    send_buffers_data[i] = send_buffers_data[i-1] + size;
	 }
      }
   }
   else
   {
      send_buffers_data = send_buffers;
   }

   /* allocate recv buffers */
   recv_buffers = hypre_TAlloc(HYPRE_Complex *,  num_recvs, HYPRE_MEMORY_HOST);
   if (num_recvs > 0)
   {
      size = hypre_CommPkgRecvBufsize(comm_pkg);
      recv_buffers[0] =  hypre_TAlloc(HYPRE_Complex,  size, HYPRE_MEMORY_HOST);
      for (i = 1; i < num_recvs; i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i-1);
         size = hypre_CommTypeBufsize(comm_type);
         recv_buffers[i] = recv_buffers[i-1] + size;
      }
   }

   /* Prepare recv buffers */
   if (alloc_dev_buffer)
   {    
      recv_buffers_data = hypre_TAlloc(HYPRE_Complex *, num_recvs,HYPRE_MEMORY_HOST);
      if (num_recvs > 0)
      {
	 size = hypre_CommPkgRecvBufsize(comm_pkg);

	 if (size > global_recv_size)
	 {
	    if (global_recv_size > 0)
	    {
	       hypre_TFree(global_recv_buffer,HYPRE_MEMORY_DEVICE);
	    }
	    global_recv_buffer = hypre_CTAlloc(HYPRE_Complex, 5*size,HYPRE_MEMORY_DEVICE);
	    global_recv_size   = 5*size;
	 }
	 recv_buffers_data[0] = global_recv_buffer;
	 for (i = 1; i < num_recvs; i++)
	 {
	    comm_type = hypre_CommPkgRecvType(comm_pkg, i-1);
	    size = hypre_CommTypeBufsize(comm_type);
	    recv_buffers_data[i] = recv_buffers_data[i-1] + size;
	 }
      }
   }
   else
   {
      recv_buffers_data = recv_buffers;
   }

   /*--------------------------------------------------------------------
    * pack send buffers
    *--------------------------------------------------------------------*/

#if BOXLOOP_VER==5
   extern int gteam_size;
   custom_parallel_for(0, num_sends, [&](int i)
#else
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for
#endif
   for (int i = 0; i < num_sends; i++)
#endif
   {
	  hypre_CommType      *comm_type;
	  hypre_CommEntryType *comm_entry;
	  HYPRE_Int num_entries;
	  HYPRE_Complex       *dptr, *kptr, *lptr;
      HYPRE_Int           *qptr, *length_array, *stride_array, unitst_array[HYPRE_MAXDIM+1], *order;
	  HYPRE_Int            j, d, ll, size;

      comm_type = hypre_CommPkgSendType(comm_pkg, i);
      num_entries = hypre_CommTypeNumEntries(comm_type);

      dptr = (HYPRE_Complex *) send_buffers_data[i];
      if ( hypre_CommPkgFirstComm(comm_pkg) )
      {
         dptr += hypre_CommPrefixSize(num_entries);
      }

      for (j = 0; j < num_entries; j++)
      {
         comm_entry = hypre_CommTypeEntry(comm_type, j);
         length_array = hypre_CommEntryTypeLengthArray(comm_entry);
         stride_array = hypre_CommEntryTypeStrideArray(comm_entry);
         order = hypre_CommEntryTypeOrder(comm_entry);
         unitst_array[0] = 1;
         for (d = 1; d <= ndim; d++)
         {
            unitst_array[d] = unitst_array[d-1]*length_array[d-1];
         }

         lptr = send_data + hypre_CommEntryTypeOffset(comm_entry);
         for (ll = 0; ll < num_values; ll++)
         {
            if (order[ll] > -1)
            {
               kptr = lptr + order[ll]*stride_array[ndim];
	       
               if(stride_array[0]==1){
#define DEVICE_VAR is_device_ptr(dptr,kptr)
				   hypre_BasicBoxLoop2BeginSimd(ndim, length_array,
											stride_array, ki,
											unitst_array, di);
				   {
					  dptr[di] = kptr[ki];
				   }
				   hypre_BasicBoxLoop2EndSimd(ki, di);
#undef DEVICE_VAR
               }
               else{
#define DEVICE_VAR is_device_ptr(dptr,kptr)
				   hypre_BasicBoxLoop2Begin(ndim, length_array,
											stride_array, ki,
											unitst_array, di);
				   {
					  dptr[di] = kptr[ki];
				   }
				   hypre_BasicBoxLoop2End(ki, di);
#undef DEVICE_VAR
               }

               dptr += unitst_array[ndim];
            }
            else
            {
               size = 1;
               for (d = 0; d < ndim; d++)
               {
                  size *= length_array[d];
               }

               hypre_Memset(dptr, 0, size*sizeof(HYPRE_Complex), HYPRE_MEMORY_DEVICE);

               dptr += size;
            }
         }
      }
   }
#if BOXLOOP_VER==5
   , gteam_size );
#endif

   /* Copy buffer data from Device to Host */
   if (num_sends > 0 && alloc_dev_buffer)
   {
      HYPRE_Complex  *dptr_host;
      size = hypre_CommPkgSendBufsize(comm_pkg);
      dptr_host = (HYPRE_Complex *) send_buffers[0];
      dptr      = (HYPRE_Complex *) send_buffers_data[0];
      hypre_TMemcpy(dptr_host,dptr,HYPRE_Complex,size,HYPRE_MEMORY_HOST,HYPRE_MEMORY_DEVICE);
   }

   for (i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);
      num_entries = hypre_CommTypeNumEntries(comm_type);

      dptr = (HYPRE_Complex *) send_buffers[i];
      if ( hypre_CommPkgFirstComm(comm_pkg) )
      {
         qptr = (HYPRE_Int *) send_buffers[i];
         *qptr = num_entries;
         qptr ++;
         hypre_TMemcpy(qptr,  hypre_CommTypeRemBoxnums(comm_type), 
               HYPRE_Int, num_entries, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         qptr += num_entries;
         hypre_TMemcpy(qptr,  hypre_CommTypeRemBoxes(comm_type), 
               hypre_Box, num_entries, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         hypre_CommTypeRemBoxnums(comm_type) = NULL;
         hypre_CommTypeRemBoxes(comm_type)   = NULL;
      }
   }

   /*--------------------------------------------------------------------
    * post receives and initiate sends
    *--------------------------------------------------------------------*/

   //interchanged send recv sequence. Post send first so that there is higher probability of
   //sends being available when recvs of interthread comm (in hypre_mpi_ep_helper.h) are called.
   //interthread comm return MPI_REQUEST_NULL into MPI_Request. decrement number of requests
   //in that case and do not update "requests" buffer

#ifdef USE_FUNNELLED_COMM

   int real_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &real_rank);	//this will give real rank, not fake rank manipulated in mpistubs.c

   extern __thread double _hypre_comm_time;

   struct timeval  tv1, tv2;
   gettimeofday(&tv1, NULL);


   omp_set_lock(const_cast<omp_lock_t*>(&g_comm_array[g_thread_id].mutex));
   g_comm_array[g_thread_id].recv_buffers = recv_buffers;
   g_comm_array[g_thread_id].send_buffers = send_buffers;
   g_comm_array[g_thread_id].comm_pkg=comm_pkg;
   g_comm_array[g_thread_id].requests = requests;
   g_comm_array[g_thread_id].status = status;
   g_comm_array[g_thread_id].num_recvs = num_recvs;
   g_comm_array[g_thread_id].num_sends = num_sends;
   g_comm_array[g_thread_id].tag = tag;
   g_comm_array[g_thread_id].req_status = 0;
   //atomic_fetch_add_explicit (&g_new_comm_req, 1, memory_order_seq_cst);
   omp_unset_lock(const_cast<omp_lock_t*>(&g_comm_array[g_thread_id].mutex));

   //share local sends with other threads. Need to use temporary staging, because sender can deallocate send_buffers as soon as its own comm is completed. Copying does not add overhead as threads are anyway waiting to complete comm
   for(i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);
	  int dest = hypre_CommTypeProc(comm_type);
	  int real_dest = dest / g_num_of_threads;
	  if(real_rank == real_dest)	//if intra-thread comm
	  {
		  int dest_thread_id = dest % g_num_of_threads;

		  int data_size = hypre_CommTypeBufsize(comm_type)*sizeof(HYPRE_Complex);
		  if(data_size>0)
		  {
			  HYPRE_Complex * temp = (HYPRE_Complex *) malloc(data_size);
			  memcpy(temp, send_buffers[i], data_size );

			  while(g_stg[g_thread_id].staging[dest_thread_id] != NULL); ///wait till previous comm is completed by receiver thread

			  //omp_set_lock(&g_stg[g_thread_id].mutex);
			  g_stg[g_thread_id].staging[dest_thread_id] = temp;
			  //omp_unset_lock(&g_stg[g_thread_id].mutex);
		  }
		  else
			  printf("thread id %d, bytes: %d\n", g_thread_id, data_size);

		  if ( hypre_CommPkgFirstComm(comm_pkg) )
		  {
			 size = hypre_CommPrefixSize(hypre_CommTypeNumEntries(comm_type));
			 hypre_CommTypeBufsize(comm_type)   -= size;
			 hypre_CommPkgSendBufsize(comm_pkg) -= size;
		  }
	  }
   }

   //receive local values from other threads.
   for(i = 0; i < num_recvs; i++)
   {
      comm_type = hypre_CommPkgRecvType(comm_pkg, i);

	  int src = hypre_CommTypeProc(comm_type);
	  int real_src = src / g_num_of_threads;

	  if(real_rank == real_src)	//if intra-thread comm
	  {
		  int src_thread_id = src % g_num_of_threads;
		  if(hypre_CommTypeBufsize(comm_type)*sizeof(HYPRE_Complex) > 0)
		  {
			  while(g_stg[src_thread_id].staging[g_thread_id]==NULL){asm("pause");}

			  //omp_set_lock(&g_stg[src_thread_id].mutex);
			  memcpy(recv_buffers[i], const_cast<HYPRE_Complex *>(g_stg[src_thread_id].staging[g_thread_id]), hypre_CommTypeBufsize(comm_type)*sizeof(HYPRE_Complex) );
			  free((void *) g_stg[src_thread_id].staging[g_thread_id]);
			  g_stg[src_thread_id].staging[g_thread_id] = NULL;
			  //omp_unset_lock(&g_stg[src_thread_id].mutex);
		  }
		  else
			  printf("recv thread id %d, bytes: %d\n", g_thread_id, hypre_CommTypeBufsize(comm_type)*sizeof(HYPRE_Complex));

	      if ( hypre_CommPkgFirstComm(comm_pkg) )
	      {
	         size = hypre_CommPrefixSize(hypre_CommTypeNumEntries(comm_type));
	         hypre_CommTypeBufsize(comm_type)   -= size;
	         hypre_CommPkgRecvBufsize(comm_pkg) -= size;
	      }
	  }

   }

   gettimeofday(&tv2, NULL);
   _hypre_comm_time += (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);

   //printf("back to thread\n");
#else

   j = 0, k=0;

   for(i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);
      hypre_MPI_Isend(send_buffers[i],
                      hypre_CommTypeBufsize(comm_type)*sizeof(HYPRE_Complex),
                      hypre_MPI_BYTE, hypre_CommTypeProc(comm_type),
                      tag, comm, &requests[j]);

      if(requests[j] != MPI_REQUEST_NULL){//this is mpi comm.
    	  j++; //requests[j] is valid. So increment j.
      }
      else{
    	  num_requests--;
      }

      if ( hypre_CommPkgFirstComm(comm_pkg) )
      {
         size = hypre_CommPrefixSize(hypre_CommTypeNumEntries(comm_type));
         hypre_CommTypeBufsize(comm_type)   -= size;
         hypre_CommPkgSendBufsize(comm_pkg) -= size;
      }
   }

   num_mpi_sends = j;

   for(i = 0; i < num_recvs; i++)
   {
      comm_type = hypre_CommPkgRecvType(comm_pkg, i);
      hypre_MPI_Irecv(recv_buffers[i],
                      hypre_CommTypeBufsize(comm_type)*sizeof(HYPRE_Complex),
                      hypre_MPI_BYTE, hypre_CommTypeProc(comm_type),
                      tag, comm, &requests[j]);

      if(requests[j] != MPI_REQUEST_NULL){//this is mpi comm.
    	  mpi_to_i[j-num_mpi_sends] = i;
    	  j++; //requests[j] is valid. So increment j.
      }
      else{
    	  interthread_requests[k] = hypre_CommTypeProc(comm_type);
		  interthread_to_i[k] = i;
		  k++;
    	  num_requests--;
      }

      if ( hypre_CommPkgFirstComm(comm_pkg) )
      {
         size = hypre_CommPrefixSize(hypre_CommTypeNumEntries(comm_type));
         hypre_CommTypeBufsize(comm_type)   -= size;
         hypre_CommPkgRecvBufsize(comm_pkg) -= size;
      }
   }

   num_inter_thread_recvs = k;
   num_mpi_recvs = j - num_mpi_sends;


   //copy number of external dependencies
   HYPRE_Int *ext_deps;
   ext_deps = hypre_TAlloc(HYPRE_Int,  hypre_CommPkgNumOfBoxes(comm_pkg), HYPRE_MEMORY_HOST);
   for(int i=0; i<hypre_CommPkgNumOfBoxes(comm_pkg); i++)
	   ext_deps[i] = hypre_CommPkgExtDeps(comm_pkg)[i];

#endif
   /*--------------------------------------------------------------------
    * set up CopyToType and exchange local data
    *--------------------------------------------------------------------*/

   if ( hypre_CommPkgFirstComm(comm_pkg) )
   {
      from_type = hypre_CommPkgCopyFromType(comm_pkg);
      to_type   = hypre_CommPkgCopyToType(comm_pkg);
      num_entries = hypre_CommTypeNumEntries(from_type);
      hypre_CommTypeNumEntries(to_type) = num_entries;
      hypre_CommTypeEntries(to_type) =
         hypre_TAlloc(hypre_CommEntryType,  num_entries, HYPRE_MEMORY_HOST);
      hypre_CommTypeSetEntries(to_type,
                               hypre_CommTypeRemBoxnums(from_type),
                               hypre_CommTypeRemBoxes(from_type),
                               hypre_CommPkgRecvStride(comm_pkg),
                               hypre_CommPkgIdentityCoord(comm_pkg),
                               hypre_CommPkgIdentityDir(comm_pkg),
                               hypre_CommPkgIdentityOrder(comm_pkg),
                               hypre_CommPkgRecvDataSpace(comm_pkg),
                               hypre_CommPkgRecvDataOffsets(comm_pkg));
      hypre_TFree(hypre_CommPkgRemBoxnums(comm_pkg), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_CommPkgRemBoxes(comm_pkg), HYPRE_MEMORY_HOST);
   }

   hypre_ExchangeLocalData(comm_pkg, send_data, recv_data, action);

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = hypre_TAlloc(hypre_CommHandle,  1, HYPRE_MEMORY_HOST);

   commHandleAllocated = 1;
   hypre_CommHandleCommPkg(comm_handle)     = comm_pkg;
   hypre_CommHandleSendData(comm_handle)    = send_data;
   hypre_CommHandleRecvData(comm_handle)    = recv_data;
   hypre_CommHandleNumRequests(comm_handle) = num_requests;
   hypre_CommHandleRequests(comm_handle)    = requests;
   hypre_CommHandleStatus(comm_handle)      = status;
   hypre_CommHandleSendBuffers(comm_handle) = send_buffers;
   hypre_CommHandleRecvBuffers(comm_handle) = recv_buffers;
   hypre_CommHandleAction(comm_handle)      = action;
   hypre_CommHandleSendBuffersDevice(comm_handle) = send_buffers_data;
   hypre_CommHandleRecvBuffersDevice(comm_handle) = recv_buffers_data;

   hypre_CommHandleInterThreadReq(comm_handle)  =    interthread_requests;
   hypre_CommHandleMPIToI(comm_handle)          =    mpi_to_i;
   hypre_CommHandleInterThreadToI(comm_handle)  =    interthread_to_i;
   hypre_CommHandleNumMPISends(comm_handle)     =    num_mpi_sends;
   hypre_CommHandleNumMPIRecvs(comm_handle)     =    num_mpi_recvs;
//   hypre_CommHandleNumInterThreadSends(comm_handle)  = num_inter_thread_sends;
   hypre_CommHandleNumInterThreadRecvs(comm_handle)  = num_inter_thread_recvs;
   hypre_CommHandleExtDeps(comm_handle) = ext_deps;
   hypre_CommHandleIndices(comm_handle) = hypre_TAlloc(HYPRE_Int,  num_recvs, HYPRE_MEMORY_HOST);

   *comm_handle_ptr = comm_handle;

   return hypre_error_flag;
}

void unpackBuffer(hypre_CommHandle *comm_handle, int i)
{
	HYPRE_Int            action       = hypre_CommHandleAction(comm_handle);
	hypre_CommType      *comm_type;
	hypre_CommPkg       *comm_pkg     = hypre_CommHandleCommPkg(comm_handle);
	hypre_CommEntryType *comm_entry;
	HYPRE_Int num_entries;
	HYPRE_Complex       *dptr, *kptr, *lptr;
	HYPRE_Int           *qptr, *length_array, *stride_array, unitst_array[HYPRE_MAXDIM+1];
	HYPRE_Int            j, d, ll;
	HYPRE_Int            ndim         = hypre_CommPkgNDim(comm_pkg);
	HYPRE_Int            num_values   = hypre_CommPkgNumValues(comm_pkg);
	HYPRE_Complex      **recv_buffers_data = hypre_CommHandleRecvBuffersDevice(comm_handle);

    comm_type = hypre_CommPkgRecvType(comm_pkg, i);
    num_entries = hypre_CommTypeNumEntries(comm_type);

    dptr = (HYPRE_Complex *) recv_buffers_data[i];

    if ( hypre_CommPkgFirstComm(comm_pkg) )
    {
       dptr += hypre_CommPrefixSize(num_entries);
    }

    for (j = 0; j < num_entries; j++)
    {
       comm_entry = hypre_CommTypeEntry(comm_type, j);
       length_array = hypre_CommEntryTypeLengthArray(comm_entry);
       stride_array = hypre_CommEntryTypeStrideArray(comm_entry);
       unitst_array[0] = 1;
       for (d = 1; d <= ndim; d++)
       {
          unitst_array[d] = unitst_array[d-1]*length_array[d-1];
       }

       lptr = hypre_CommHandleRecvData(comm_handle) +
          hypre_CommEntryTypeOffset(comm_entry);
       for (ll = 0; ll < num_values; ll++)
       {
          kptr = lptr + ll*stride_array[ndim];

          if (action > 0)
          {
          	if(stride_array[0]==1){
#define DEVICE_VAR is_device_ptr(kptr,dptr)
				   hypre_BasicBoxLoop2BeginSimd(ndim, length_array,
											stride_array, ki,
											unitst_array, di);
				   {
					  kptr[ki] += dptr[di];
				   }
				   hypre_BasicBoxLoop2EndSimdParallel(ki, di);
#undef DEVICE_VAR
          	}
          	else{
#define DEVICE_VAR is_device_ptr(kptr,dptr)
				   hypre_BasicBoxLoop2Begin(ndim, length_array,
											stride_array, ki,
											unitst_array, di);
				   {
					  kptr[ki] += dptr[di];
				   }
				   hypre_BasicBoxLoop2EndParallel(ki, di);
#undef DEVICE_VAR
          	}
          }
          else
          {
          	if(stride_array[0]==1){
#define DEVICE_VAR is_device_ptr(kptr,dptr)
				   hypre_BasicBoxLoop2BeginSimd(ndim, length_array,
											stride_array, ki,
											unitst_array, di);
				   {
					  kptr[ki] = dptr[di];
				   }
				   hypre_BasicBoxLoop2EndSimdParallel(ki, di);
#undef DEVICE_VAR
          	}
          	else{
#define DEVICE_VAR is_device_ptr(kptr,dptr)
				   hypre_BasicBoxLoop2Begin(ndim, length_array,
											stride_array, ki,
											unitst_array, di);
				   {
					  kptr[ki] = dptr[di];
				   }
				   hypre_BasicBoxLoop2EndParallel(ki, di);
#undef DEVICE_VAR
          	}
          }

          dptr += unitst_array[ndim];
       }
    }
}

void freeCommHandle(hypre_CommHandle *comm_handle)
{
	hypre_CommPkg       *comm_pkg     = hypre_CommHandleCommPkg(comm_handle);
	HYPRE_Complex      **send_buffers_data = hypre_CommHandleSendBuffersDevice(comm_handle);
	HYPRE_Complex      **recv_buffers_data = hypre_CommHandleRecvBuffersDevice(comm_handle);
	HYPRE_Int            num_sends    = hypre_CommPkgNumSends(comm_pkg);
	HYPRE_Int            num_recvs    = hypre_CommPkgNumRecvs(comm_pkg);
	HYPRE_Complex      **send_buffers = hypre_CommHandleSendBuffers(comm_handle);
	HYPRE_Complex      **recv_buffers = hypre_CommHandleRecvBuffers(comm_handle);

	hypre_TFree(hypre_CommHandleIndices(comm_handle), HYPRE_MEMORY_HOST);
	hypre_TFree(hypre_CommHandleInterThreadReq(comm_handle), HYPRE_MEMORY_HOST);
	hypre_TFree(hypre_CommHandleMPIToI(comm_handle), HYPRE_MEMORY_HOST);
	hypre_TFree(hypre_CommHandleInterThreadToI(comm_handle), HYPRE_MEMORY_HOST);
	hypre_TFree(hypre_CommHandleExtDeps(comm_handle), HYPRE_MEMORY_HOST);
	hypre_TFree(hypre_CommHandleRequests(comm_handle), HYPRE_MEMORY_HOST);
	hypre_TFree(hypre_CommHandleStatus(comm_handle), HYPRE_MEMORY_HOST);
	if (num_sends > 0)
	{
		hypre_TFree(send_buffers[0], HYPRE_MEMORY_HOST);
	}
	if (num_recvs > 0)
	{
		hypre_TFree(recv_buffers[0], HYPRE_MEMORY_HOST);
	}

	hypre_TFree(comm_handle, HYPRE_MEMORY_HOST);

	if (send_buffers == send_buffers_data)
	{
		hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
	}
	else
	{
		hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
		hypre_TFree(send_buffers_data, HYPRE_MEMORY_HOST);
	}
	if (recv_buffers == recv_buffers_data)
	{
		hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
	}
	else
	{
		hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
		hypre_TFree(recv_buffers_data, HYPRE_MEMORY_HOST);
	}

}

extern __thread int g_overlap_comm;

HYPRE_Int
hypre_FinalizeOverlappedCommunication( hypre_CommHandle *comm_handle,    hypre_ComputePkg *compute_pkg, HYPRE_Int *completed_ptr)
{
#ifdef HYPRE_USING_CUDA
	printf("Overlapped comm not yet supported in CUDA. Copy and update code for H2D transfer from hypre_FinalizeCommunication here. Look for 'if (num_recvs > 0 && alloc_dev_buffer)' and 'hypre_TMemcpy'\n");
	exit(1);
#endif

	HYPRE_Int completed=0;
	hypre_CommPkg       *comm_pkg     = hypre_CommHandleCommPkg(comm_handle);


	hypre_BoxArrayArray     *rolling_dept_boxes, *dept_boxes;
	hypre_BoxArray          *rd_box_a, *d_box_a;
	hypre_Box               *dbox, *rbox;
	HYPRE_Int i;

	rolling_dept_boxes = hypre_ComputePkgRollingDeptBoxes(compute_pkg);
	dept_boxes =  hypre_ComputePkgDeptBoxes(compute_pkg);

	if ( hypre_CommPkgFirstComm(comm_pkg) || g_overlap_comm!=1){	//call hypre_FinalizeCommunication as it is if its a first comm or comm is NOT overlapped
		HYPRE_Int retVal = hypre_FinalizeCommunication(comm_handle);

		//add all boxes into dependent boxes to rolling boxes

		hypre_ForBoxArrayI(i, rolling_dept_boxes)
		{
			rd_box_a = hypre_BoxArrayArrayBoxArray(rolling_dept_boxes, i);
			d_box_a  = hypre_BoxArrayArrayBoxArray(dept_boxes, i);
			hypre_BoxArraySetSize(rd_box_a, 0);	//reset the box

			if(hypre_BoxArraySize(d_box_a)>0){//if there is a dependent box, copy it to rolling boxes
				hypre_BoxArraySetSize(rd_box_a, 1);
				rbox = hypre_BoxArrayBox(rd_box_a, 0);
				dbox = hypre_BoxArrayBox(d_box_a, 0);
				hypre_CopyBox(dbox, rbox);
			}
		}
		*completed_ptr = 1;
		return retVal;
	}

	int readyboxes = 0;
	HYPRE_Int *ext_deps = hypre_CommHandleExtDeps(comm_handle);
	HYPRE_Int outcount;
	HYPRE_Int *lindices = hypre_CommHandleIndices(comm_handle);
	HYPRE_Int *index_to_i = hypre_CommHandleInterThreadToI(comm_handle); //default to interthread comm. set to mpi if its mpi;

	//reset rolling boxes
	hypre_ForBoxArrayI(i, rolling_dept_boxes)
	{
		rd_box_a = hypre_BoxArrayArrayBoxArray(rolling_dept_boxes, i);
		hypre_BoxArraySetSize(rd_box_a, 0);
	}

	while(readyboxes==0 && completed==0){//loop until at least one box is ready or comm is completed.
		outcount=0;
		HYPRE_Int inter_thread_comm_completed = irWaitsome(hypre_CommHandleNumInterThreadRecvs(comm_handle),
														   hypre_CommHandleInterThreadReq(comm_handle),
														   &outcount, lindices);

		index_to_i = hypre_CommHandleInterThreadToI(comm_handle);

		if(inter_thread_comm_completed && outcount==0){//call mpi waitsome if inter_thread_comm_completed and there is no "new" comm in this round.

			if(hypre_CommHandleNumMPIRecvs(comm_handle)>0){
				MPI_Waitsome(hypre_CommHandleNumMPIRecvs(comm_handle), //check for mpi recvs only.
							 hypre_CommHandleRequests(comm_handle) + hypre_CommHandleNumMPISends(comm_handle), //add num of MPI sends to skip send requests
							 &outcount, lindices,
							 hypre_CommHandleStatus(comm_handle) + hypre_CommHandleNumMPISends(comm_handle) );
			}
			index_to_i = hypre_CommHandleMPIToI(comm_handle);

			//"completed" flag based on  MPI_Waitsome
			if(outcount == MPI_UNDEFINED || outcount == 0)
				completed = 1; //This is the last comm.
		}

		//copy boxes for all completed comms to rolling boxes
		for(int id=0; id<outcount; id++){
			int i=index_to_i[lindices[id]]; //this "i" is for num_of_recvs. Not to be confused with box ids in grid

			unpackBuffer(comm_handle, i);	//unpack received data

			//add corresponding boxes to dependent boxes list

			//get comm type and proc
			hypre_CommType *comm_type = hypre_CommPkgRecvType(comm_pkg, i);
			HYPRE_Int proc = hypre_CommTypeProc(comm_type);

			//get box ids from proc_to_box map
			auto range = hypre_CommPkgProcToBoxMap(comm_pkg).equal_range(proc);

			//copy boxes to rolling boxes
			for (auto b = range.first; b != range.second; ++b){
				int boxid = b->second;
				ext_deps[boxid]--;	//decrement extl dependency count

				if(ext_deps[boxid]==0){//add box into rolling box array only if extl dependcy count is 0.
					rd_box_a = hypre_BoxArrayArrayBoxArray(rolling_dept_boxes, boxid);
					d_box_a  = hypre_BoxArrayArrayBoxArray(dept_boxes, boxid);
					hypre_BoxArraySetSize(rd_box_a, 1);
					rbox = hypre_BoxArrayBox(rd_box_a, 0);
					dbox = hypre_BoxArrayBox(d_box_a, 0);
					hypre_CopyBox(dbox, rbox);
					readyboxes++;
				}
			}
		}
	}

	*completed_ptr = completed;
	return hypre_error_flag;
}

//wait for sends to complete
void hypre_FinalizeSends( hypre_CommHandle *comm_handle)
{
	if(commHandleAllocated){
		irWaitSend();
		if(hypre_CommHandleNumMPISends(comm_handle)>0)
			hypre_MPI_Waitall(hypre_CommHandleNumMPISends(comm_handle),
					hypre_CommHandleRequests(comm_handle),
					hypre_CommHandleStatus(comm_handle));
		freeCommHandle(comm_handle);
		commHandleAllocated = 0;
	}
}
/*--------------------------------------------------------------------------
 * Finalize a communication exchange.  This routine blocks until all
 * of the communication requests are completed.
 *
 * The communication requests are completed, and the receive buffer is
 * manually unpacked.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FinalizeCommunication( hypre_CommHandle *comm_handle )
{
   hypre_CommPkg       *comm_pkg     = hypre_CommHandleCommPkg(comm_handle);
   HYPRE_Complex      **recv_buffers = hypre_CommHandleRecvBuffers(comm_handle);

   HYPRE_Int            num_recvs    = hypre_CommPkgNumRecvs(comm_pkg);

   hypre_CommType      *comm_type;
   HYPRE_Int            num_entries;

   HYPRE_Complex       *dptr;
   HYPRE_Int           *qptr;

   HYPRE_Int           *boxnums;
   hypre_Box           *boxes;

   HYPRE_Int            i, j, d, ll;

   HYPRE_Complex      **recv_buffers_data = hypre_CommHandleRecvBuffersDevice(comm_handle);

   /*--------------------------------------------------------------------
    * finish communications
    *--------------------------------------------------------------------*/

#ifdef USE_FUNNELLED_COMM

      extern __thread double _hypre_comm_time;

	  struct timeval  tv1, tv2;
	  gettimeofday(&tv1, NULL);

	  while(g_comm_array[g_thread_id].req_status != 2){asm("pause");}

	  gettimeofday(&tv2, NULL);
	  _hypre_comm_time += (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);



#else

   //if (hypre_CommHandleNumRequests(comm_handle)) //moved this condition within hypre_MPI_Waitall because there could be interthread comm
   {
      hypre_MPI_Waitall(hypre_CommHandleNumRequests(comm_handle),
                        hypre_CommHandleRequests(comm_handle),
                        hypre_CommHandleStatus(comm_handle));
   }
#endif

   /*--------------------------------------------------------------------
    * if FirstComm, unpack prefix information and set 'num_entries' and
    * 'entries' for RecvType
    *--------------------------------------------------------------------*/

   if ( hypre_CommPkgFirstComm(comm_pkg) )
   {
      hypre_CommEntryType  *ct_entries;

      num_entries = 0;
      for (i = 0; i < num_recvs; i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i);

         qptr = (HYPRE_Int *) recv_buffers[i];
         hypre_CommTypeNumEntries(comm_type) = *qptr;
         num_entries += hypre_CommTypeNumEntries(comm_type);
      }

      /* allocate CommType entries 'ct_entries' */
      ct_entries = hypre_TAlloc(hypre_CommEntryType,  num_entries, HYPRE_MEMORY_HOST);

      /* unpack prefix information and set RecvType entries */
      for (i = 0; i < num_recvs; i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i);
         hypre_CommTypeEntries(comm_type) = ct_entries;
         ct_entries += hypre_CommTypeNumEntries(comm_type);

         qptr = (HYPRE_Int *) recv_buffers[i];
         num_entries = *qptr;
         qptr ++;
         boxnums = qptr;
         qptr += num_entries;
         boxes = (hypre_Box *) qptr;
         hypre_CommTypeSetEntries(comm_type, boxnums, boxes,
                                  hypre_CommPkgRecvStride(comm_pkg),
                                  hypre_CommPkgIdentityCoord(comm_pkg),
                                  hypre_CommPkgIdentityDir(comm_pkg),
                                  hypre_CommPkgIdentityOrder(comm_pkg),
                                  hypre_CommPkgRecvDataSpace(comm_pkg),
                                  hypre_CommPkgRecvDataOffsets(comm_pkg));
      }
   }

   /*--------------------------------------------------------------------
    * unpack receive buffer data
    *--------------------------------------------------------------------*/

   /* Copy buffer data from Host to Device */
   HYPRE_Int alloc_dev_buffer = 0;
   /* In the case of running on device and cannot access host memory from device */
#if (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)) && (HYPRE_MEMORY_HOST_ACT != HYPRE_MEMORY_SHARED)
#if defined(HYPRE_USING_RAJA) || defined(HYPRE_USING_KOKKOS)
   alloc_dev_buffer = 1;
#elif defined(HYPRE_USING_CUDA)
   alloc_dev_buffer = (hypre_exec_policy == HYPRE_MEMORY_DEVICE);
#elif defined(HYPRE_USING_DEVICE_OPENMP)
   alloc_dev_buffer = hypre__global_offload;
#endif
#endif

   if (num_recvs > 0 && alloc_dev_buffer)
   {
      HYPRE_Complex  *dptr_host;
      HYPRE_Int       size;
      size = 0;
      for (i = 0; i < num_recvs; i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i);
         num_entries = hypre_CommTypeNumEntries(comm_type);
         size += hypre_CommTypeBufsize(comm_type);
         if ( hypre_CommPkgFirstComm(comm_pkg) )
         {
            size += hypre_CommPrefixSize(num_entries);
         }
      }
      dptr_host = (HYPRE_Complex *) recv_buffers[0];
      dptr      = (HYPRE_Complex *) recv_buffers_data[0];

      hypre_TMemcpy( dptr, dptr_host, HYPRE_Complex, size, 
                     HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST );
   }

//#if BOXLOOP_VER==5
//   extern int gteam_size;
//   custom_parallel_for(0, num_recvs, [&](int i)
//#else
//#ifdef HYPRE_USING_OPENMP
//#pragma omp parallel for
//#endif
   for (int i = 0; i < num_recvs; i++)
//#endif
   {
	   unpackBuffer(comm_handle, i);
   }

//#if BOXLOOP_VER==5
//   , gteam_size );
//#endif
   /*--------------------------------------------------------------------
    * turn off first communication indicator
    *--------------------------------------------------------------------*/

   hypre_CommPkgFirstComm(comm_pkg) = 0;
   commHandleAllocated = 0;

   /*--------------------------------------------------------------------
    * Free up communication handle
    *--------------------------------------------------------------------*/
   freeCommHandle(comm_handle);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Execute local data exchanges.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ExchangeLocalData( hypre_CommPkg *comm_pkg,
                         HYPRE_Complex *send_data,
                         HYPRE_Complex *recv_data,
                         HYPRE_Int      action )
{
   HYPRE_Int            ndim       = hypre_CommPkgNDim(comm_pkg);
   HYPRE_Int            num_values = hypre_CommPkgNumValues(comm_pkg);
   hypre_CommType      *copy_fr_type;
   hypre_CommType      *copy_to_type;
   hypre_CommEntryType *copy_fr_entry;
   hypre_CommEntryType *copy_to_entry;

   HYPRE_Complex       *fr_dp;
   HYPRE_Int           *fr_stride_array;
   HYPRE_Complex       *to_dp;
   HYPRE_Int           *to_stride_array;
   HYPRE_Complex       *fr_dpl, *to_dpl;

   HYPRE_Int           *length_array;
   HYPRE_Int            i, ll;

   HYPRE_Int           *order;

   /*--------------------------------------------------------------------
    * copy local data
    *--------------------------------------------------------------------*/

   copy_fr_type = hypre_CommPkgCopyFromType(comm_pkg);
   copy_to_type = hypre_CommPkgCopyToType(comm_pkg);

#if BOXLOOP_VER==5
   extern int gteam_size;
   custom_parallel_for(0, hypre_CommTypeNumEntries(copy_fr_type), [&](int i)
#else
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for
#endif
   for (int i = 0; i < hypre_CommTypeNumEntries(copy_fr_type); i++)
#endif
   {
	  hypre_CommEntryType *copy_fr_entry = hypre_CommTypeEntry(copy_fr_type, i);
	  hypre_CommEntryType *copy_to_entry = hypre_CommTypeEntry(copy_to_type, i);

	  HYPRE_Complex *fr_dp = send_data + hypre_CommEntryTypeOffset(copy_fr_entry);
	  HYPRE_Complex *to_dp = recv_data + hypre_CommEntryTypeOffset(copy_to_entry);

      /* copy data only when necessary */
      if (to_dp != fr_dp)
      {
    	 HYPRE_Int *length_array = hypre_CommEntryTypeLengthArray(copy_fr_entry);

    	 HYPRE_Int *fr_stride_array = hypre_CommEntryTypeStrideArray(copy_fr_entry);
    	 HYPRE_Int *to_stride_array = hypre_CommEntryTypeStrideArray(copy_to_entry);
    	 HYPRE_Int *order = hypre_CommEntryTypeOrder(copy_fr_entry);

         for (HYPRE_Int ll = 0; ll < num_values; ll++)
         {
            if (order[ll] > -1)
            {
               HYPRE_Complex *fr_dpl = fr_dp + (order[ll])*fr_stride_array[ndim];
               HYPRE_Complex *to_dpl = to_dp + (      ll )*to_stride_array[ndim];

               if (action > 0)
               {
            	   if(fr_stride_array[0]==1 && to_stride_array[0]==1){
#define DEVICE_VAR is_device_ptr(to_dpl,fr_dpl)
					 hypre_BasicBoxLoop2BeginSimd(ndim, length_array,
											  fr_stride_array, fi,
											  to_stride_array, ti);
					  {
						 /* add the data to existing values in memory */
						 to_dpl[ti] += fr_dpl[fi];
					  }
					  hypre_BasicBoxLoop2EndSimd(fi, ti);
#undef DEVICE_VAR
            	   }
            	   else{
#define DEVICE_VAR is_device_ptr(to_dpl,fr_dpl)
					 hypre_BasicBoxLoop2Begin(ndim, length_array,
											  fr_stride_array, fi,
											  to_stride_array, ti);
					  {
						 /* add the data to existing values in memory */
						 to_dpl[ti] += fr_dpl[fi];
					  }
					  hypre_BasicBoxLoop2End(fi, ti);
#undef DEVICE_VAR
            	   }
               }
               else
               {


            	   if(fr_stride_array[0]==1 && to_stride_array[0]==1){
#define DEVICE_VAR is_device_ptr(to_dpl,fr_dpl)
					  hypre_BasicBoxLoop2BeginSimd(ndim, length_array,
											   fr_stride_array, fi,
											   to_stride_array, ti);
					  {
						 /* copy the data over existing values in memory */
						 to_dpl[ti] = fr_dpl[fi];
					  }
					  hypre_BasicBoxLoop2EndSimd(fi, ti);
#undef DEVICE_VAR
            	   }
            	   else{
#define DEVICE_VAR is_device_ptr(to_dpl,fr_dpl)
					  hypre_BasicBoxLoop2Begin(ndim, length_array,
											   fr_stride_array, fi,
											   to_stride_array, ti);
					  {
						 /* copy the data over existing values in memory */
						 to_dpl[ti] = fr_dpl[fi];
					  }
					  hypre_BasicBoxLoop2End(fi, ti);
#undef DEVICE_VAR
            	   }



               }
            }
         }
      }
   }
#if BOXLOOP_VER==5
   , gteam_size );
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommPkgDestroy( hypre_CommPkg *comm_pkg )
{
   hypre_CommType  *comm_type;
   HYPRE_Int      **orders;
   HYPRE_Int        i;

   if (comm_pkg)
   {
	   hypre_TFree( hypre_CommPkgExtDeps(comm_pkg), HYPRE_MEMORY_HOST );
      /* note that entries are allocated in two stages for To/Recv */
      if (hypre_CommPkgNumRecvs(comm_pkg) > 0)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, 0);
         hypre_TFree(hypre_CommTypeEntries(comm_type), HYPRE_MEMORY_HOST);
      }
      comm_type = hypre_CommPkgCopyToType(comm_pkg);
      hypre_TFree(hypre_CommTypeEntries(comm_type), HYPRE_MEMORY_HOST);
      hypre_TFree(comm_type, HYPRE_MEMORY_HOST);

      comm_type = hypre_CommPkgCopyFromType(comm_pkg);
      hypre_TFree(comm_type, HYPRE_MEMORY_HOST);

      hypre_TFree(hypre_CommPkgEntries(comm_pkg), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_CommPkgRemBoxnums(comm_pkg), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_CommPkgRemBoxes(comm_pkg), HYPRE_MEMORY_HOST);

      hypre_TFree(hypre_CommPkgRecvDataOffsets(comm_pkg), HYPRE_MEMORY_HOST);
      hypre_BoxArrayDestroy(hypre_CommPkgRecvDataSpace(comm_pkg));

      orders = hypre_CommPkgOrders(comm_pkg);
      for (i = 0; i < hypre_CommPkgNumOrders(comm_pkg); i++)
      {
         hypre_TFree(orders[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(orders, HYPRE_MEMORY_HOST);

      hypre_TFree(hypre_CommPkgIdentityOrder(comm_pkg), HYPRE_MEMORY_HOST);

      hypre_TFree(comm_pkg, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

