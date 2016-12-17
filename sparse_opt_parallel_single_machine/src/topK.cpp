#include "topK.h"
int k;
int *pos;
double *list;
/* Gateway of maxkmex */
void getTopK(VectorXd* srcVec, int kval,
             int *topKIndex, double *topK) {

    int l, i, kout, nelem, p0, nz, nansplit;

    nelem = srcVec->size();
    /* Get the number of elements of the list of subindexes */
    l = nelem;

    /* Get the number of elements of the list of subindexes */
    //if (mxGetM(prhs[1])!=1 || mxGetN(prhs[1])!=1)
    //    mexErrMsgTxt("MAXKMEX: Second input K must be a scalar.");

    //if (mxGetClassID(prhs[1]) != mxDOUBLE_CLASS)
    //    mexErrMsgTxt("MAXKMEX: Second input K must be a double.");
    k = kval;
    kout = k;
    if (k <= 0)
    {
        printf("k must be larger than 0");
        return;
    }

    /* Get a data pointer */
    list = srcVec->data();

    /* Clip k */
    if (k>l) k=l;

    /* Clip kout */
    if (kout>nelem) kout=nelem;

    /* Clean programming */
    pos=NULL;

    nansplit = l-1;
    /* Work for non-empty array */
    if (l>0) {
        /* Vector of index */
        pos = (int *)malloc(sizeof(int) * l);
        //pos = mxMalloc(sizeof(mwSize)*l);
        if (pos==NULL)
        {
            printf("Out of memory.");
            return;
        }
        /* Initialize the array of position (zero-based index) */
        for (i=0; i<l; i++) pos[i]=i;


        /* Call the recursive engine */
        k--; /* because we work on zero-based */
        //nansplit = partNaN(0, l-1); /* Push NaN at the end */
        //findFirstK(0, l-1);
        if (k<nansplit && nansplit>=0)
            findFirstK(0, nansplit);

        /* ++ to Restore one-based Matlab index */
        p0 = ++k;
    } /* if (l>0) */
    else p0 = 0;

    /* Number of implicite zero in (sparse) */
    nz = nelem-l;
    /* Create the Matrix result (first output) */
    MinMaxResult(k, p0, nz, kout, topK);

    /* Create the Matrix position (second output) */

    LocResult(k, p0, nz, kout, topKIndex);


    /* Free the array of position */
    if (pos) free(pos);
    pos = NULL; /* clean programming */

    return;

} /* Gateway of maxkmex.c */
/* Create the result contains the locatio of k largest values */
void LocResult(int k, int p0, int nz,
               int kout, int *Result)
{
    int i;
    int dims[2];
    //mxArray* Result;
    int *data;

    dims[0] = kout; dims[1] = 1;
    //Result = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    //if (Result == NULL)
    //    mexErrMsgTxt("Out of memory.");
    data = Result;

    /* index of positive part */
    for (i=0; i<p0; i++) data[i]=pos[i]; /* one-based indexing */

    if (nz>kout-p0)
        nz = kout-p0;
    /* Fill nz zeros */
    memset((void*)(data+p0), 0, sizeof(int)*nz);

    /* index of negative part */
    for (i=p0+nz; i<kout; i++) data[i]=pos[i-nz];

    return;
} /* LocResult */


/*Find the index of the Median of the elements
of array that occur at every "shift" positions.*/
int findMedianIndex(int left, int right, int shift)
{
    int tmp, groups, k;
    double maxValue;
    int *pi, *pj, *pk, *pright, *pmaxIndex;

    groups = (right-left)/shift + 1;
    pk = pos + (k = left + (groups/2)*shift);
    pright = pos + right;
    for (pi=pos+left; pi<=pk; pi+= shift)
    {
        pmaxIndex = pi;
        maxValue = list[*pmaxIndex];

        for (pj=pi; pj<=pright; pj+=shift)
            if (list[*pj]>maxValue) /* Comparison */
                maxValue = list[*(pmaxIndex=pj)];
        /* Swap pos[i] with pos[maxIndex] */
        tmp = *pi;
        *pi = *pmaxIndex;
        *pmaxIndex = tmp;
    }

    return k;

} /* findMedianIndex */


/*************************************************************************/
/*Computes the median of three points (left,right,and mid) */
int findMedianThree(int left, int right)
{
    double vleft, vright, vmid;
    int mid;

    if (left==right) return left;

    vleft = list[pos[left]];
    vright = list[pos[right]];
    vmid = list[pos[mid = (left+right+1)/2]];

    if (vleft<vright)
    {
        if (vmid>vright)
            return right;
        else if (vmid<vleft)
            return left;
        else
            return mid;

    } else { /* (vleft>=vright) */

        if (vmid>vleft)
            return left;
        else if (vmid<vright)
            return right;
        else
            return mid;

    }
} /* findMedianThree */

/*Computes the median of each group of 5 elements and stores
  it as the first element of the group (left). Recursively does this
  till there is only one group and hence only one Median */
int findMedianOfMedians(int left, int right)
{
    int i, shift, step, tmp;
    int endIndex, medianIndex;

    if (left==right) return left;

    shift = 1;
    while (shift <= (right-left))
    {
        step=shift*5;
        for (i=left; i<=right; i+=step)
        {
            if ((endIndex=i+step-1)>=right)
                endIndex=right;
            medianIndex = findMedianIndex(i, endIndex, shift);
            /* Swap pos[i] with pos[medianIndex] */
            tmp = pos[i];
            pos[i] = pos[medianIndex];
            pos[medianIndex] = tmp;
        }
        shift = step;
    }
    return left;
} /* findMedianOfMedians */


/* Recursive engine (partial quicksort) */
void findFirstK(int left, int right) {

    int pivotIndex;

    if (right > left) {

#if (PIVOT==MEDIANMEDIANS)
        pivotIndex = findMedianOfMedians(left, right);
#elif (PIVOT==MEDIAN3)
        pivotIndex = findMedianThree(left, right);
#else /* MIDPOINT */
        pivotIndex = (left+right+1)/2;
#endif

        pivotIndex = partition(left, right, pivotIndex);
        if (pivotIndex > k)
            findFirstK(left, pivotIndex-1);
        else if (pivotIndex < k)
            findFirstK(pivotIndex+1, right);
    }

    return;
} /* findFirstK *


/* Partitioning the list around pivot pivotValue := l[pivotIndex];
   After runing, at exit we obtain: 
   l[left]...l[index-1] > pivotValue >= l[index] ... l[right]
   where l[i] := list[pos[i]] for all i */
int partition(int left, int right, int pivotIndex) {

    double pivotValue;
    int *pindex, *pi, *pright;
    int tmp;

    pright=pos+right;
    pindex=pos+pivotIndex;
    pivotValue = list[tmp = *pindex];
    /* Swap pos[pivotIndex] with pos[right] */
    *pindex = *pright;
    *pright = tmp;

    pindex=pos+left;
    for (pi=pindex; pi<pright; pi++)
        /* Compare with pivotValue */
        if (list[*pi] > pivotValue) {
            /* if larger; Swap pos[index] with pos[i] */
            tmp = *pi;
            *pi = *pindex;
            *(pindex++) = tmp;
        }

    /* Swap pos[index] with pos[right] */
    tmp = *pindex;
    *pindex = *pright;
    *pright = tmp;

    return (pindex-pos); /* Pointer arithmetic */
} /* Partition *


/* Create the result contains k largest values */
void MinMaxResult(int k, int p0, int nz,
                  int kout, double *Result)
{
    int i;
    int dims[2];
    //mxArray* Result;
    double *data;

    /* Create the Matrix result (first output) */
    dims[0] = kout; dims[1] = 1;
    //Result = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    //if (Result == NULL)
    //   mexErrMsgTxt("Out of memory.");
    data = Result;
    /* copy positive part (p0) */
    for (i=0; i<p0; i++) data[i]=list[pos[i]];

    if (nz>kout-p0)
        nz = kout-p0;
    /* Fill nz zeros */
    memset((void*)(data+p0), 0, sizeof(double)*nz);

    /* copy negative part (kout - (p0+nz)) */
    for (i=p0+nz; i<kout; i++) data[i]=list[pos[i-nz]];

    return;
    //return Result;
} /* MinMaxResult */

int partNaN(int left, int right) {

    int *pleft, *pright, tmp;
    int *pfirst;

    pfirst = pleft = pos+left;
    pright = pos+right;

    for (;;) {
        while ((pleft<pright))
            pleft++;
        while ((pleft<pright))
            pright--;
        if (pleft<pright) {
            /* Swap left and right */
            tmp = *pleft;
            *pleft = *pright;
            *pright = tmp;
            pleft++, pright--;
        }
        else {
            if (pright>=pfirst && ISNAN(list[*pright]))
                pright--;
            return (pright-pos);
        }
    } /* for-loop */
} /* partNaN */

