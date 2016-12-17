//
// Created by lezi on 10/29/16.
//

#ifndef SPARSE_OPT_PARALLEL_SINGLE_MACHINE_COMMON_H
#define SPARSE_OPT_PARALLEL_SINGLE_MACHINE_COMMON_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#ifdef __cplusplus
extern "C" {
#endif

using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;


struct server_parameter
{
    int numWorker;
    int beta;
    int maxIter;
    double lambda;
    int dim;
    int sparseDim;
    double epsacc;
    int num_data_total;
};

struct worker_parameter
{
    int maxIter;
    double epsacc;
    int sapBlock;
    double eta1;
    double eta2;
    int dim;
    int sparseDim;
    double lambda;
    int ndata;
};

struct MiniBatch
{
    int num; //num of data
    int dataDim;
    SpMat X;
    VectorXd Y;
    VectorXd alpha;
};

struct Batch_thred
{
    int num_mBatches;
    vector<MiniBatch> mBatches;
};

double squarehingeloss(SpMat * w, vector<Batch_thred>* data,  double lambda, int total_data_n);
double squarehingeloss_dense(VectorXd * w, vector<Batch_thred>* data,  double lambda, int N);

void duallossSquaredhinMatrixXdge();

VectorXd gradientDescentDualSquaredloss(worker_parameter wparam, VectorXd * subalphaprev, VectorXd * W_prev, double eta1, SpMat * miniBatchX, VectorXd * miniBatchY);

void svm_set_print_string_function(void (*print_func)(const char *));

vector<Batch_thred> loadData(int num_thred, int mBatch_sz, int dataDim, char* datalist);


#ifdef __cplusplus
}
#endif


#endif //SPARSE_OPT_PARALLEL_SINGLE_MACHINE_COMMON_H