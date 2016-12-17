//
// Created by lezi on 11/1/16.
//

#ifndef SPARSE_OPT_PARALLEL_SINGLE_MACHINE_TOPK_H
#define SPARSE_OPT_PARALLEL_SINGLE_MACHINE_TOPK_H



#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <stddef.h>
#include <Eigen/Dense>
using namespace Eigen;

/* Define correct type depending on platform */
#if defined(_MSC_VER) || defined(__BORLANDC__)
typedef unsigned __int64 ulong64;
#elif defined(_LCC)
typedef long long long64;
typedef unsigned long long ulong64;
#else
typedef unsigned long long ulong64;
#endif

/* Global variables, used to avoid stacking them during recusive call since
   they do not change */


#define MIDPOINT 0
#define MEDIAN3 1
#define MEDIANMEDIANS 2

/* Pivot Strategy, use one of the above */
#define PIVOT MIDPOINT
void getTopK(VectorXd *srcVect, int k, int *index, double *topK);

/*************************************************************************/
/*Find the index of the Median of the elements
of array that occur at every "shift" positions.*/
int findMedianIndex(int left, int right, int shift);

/*Computes the median of each group of 5 elements and stores
  it as the first element of the group (left). Recursively does this
  till there is only one group and hence only one Median */
/*************************************************************************/
int findMedianOfMedians(int left, int right);

/*Computes the median of three points (left,right,and mid) */
int findMedianThree(int left, int right);

/* A quiet NaN is represented by any bit pattern
   between X'7FF80000 00000000' and X'7FFFFFFF FFFFFFFF' or
   between X'FFF80000 00000000' and X'FFFFFFFF FFFFFFFF'. */
#define NANmask 0x7ff8000000000000
#define ISNAN(x) ((*(ulong64*)(&x) & NANmask)  == NANmask)
#define MINF 0xfff0000000000000

/* Partitioning the list around the pivot NaN.
   After runing, at exit we obtain pindex satisfied:
   l[left]...l[index] are regular numbers (might include Inf)
   l[index+1] ... l[right] are NaN
   where l[i] := list[pos[i]] for all i */
int partNaN(int left, int right);

/* Partitioning the list around pivot pivotValue := l[pivotIndex];
   After runing, at exit we obtain:
   l[left]...l[index-1] > pivotValue >= l[index] ... l[right]
   where l[i] := list[pos[i]] for all i */
int partition(int left, int right, int pivotIndex);

/* Partitioning the list around pivot 0;
 * After runing, at exit we obtain:
   l[left]...l[index-1] > 0 >= l[index] ... l[right]
   where l[i] := list[pos[i]] for all i
   Note: at return, index might be larger than right (if all elements are
         strictly greater than zero) */
int part0(int left, int right);

/* Recursive engine (partial quicksort) */
void findFirstK(int left, int right);

/* Create the result contains k largest values */
void MinMaxResult(int k, int p0, int nz, int kout, double*);

void LocResult(int k, int p0, int nz, int kout, int*);

/* FindSPzero, find the location of zeros in sparse matrix */
void FindSPzero(const double* S, int nz, double* I, double *J);


/* Create the result contains the location of k smallest values
 for sparse matrix */
void SpLocResult(int k, int p0, int nz, int kout,
                 const double* S, double** I, double** J);

/* Gateway of maxkmex */

#endif //SPARSE_OPT_PARALLEL_SINGLE_MACHINE_TOPK_H
