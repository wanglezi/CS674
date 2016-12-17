//
// Created by lezi on 10/29/16.
//
#include <iostream>
#include <omp.h>
#include "common.h"
#include "solve.h"
#include "topK.h"

void maxkmex(VectorXd * model_W, int k, int * idx, double *val){
    for(int i = 0 ; i < k ; i++){
        idx[i] = i;
        val[i] = i;
    }
}


void server_run(vector<VectorXd> * sub_W, int k, int dim, int num_workers, SpMat * model_W,server_parameter sparam, vector<Batch_thred> * data) {
    VectorXd model_W_dense;
    model_W_dense = VectorXd::Zero(dim);
    //model_W_dense = VectorXd(* model_W);
    int *idx = new int[k];
    double *val = new double[k];
    //averaging
    for (int i = 0; i < num_workers; i++) {
        model_W_dense = model_W_dense + (*sub_W)[i];
    }

    model_W_dense = model_W_dense / num_workers;

    //sync_min
    /*ouble loss = 10000;
    int wid = 0;
    clock_t loss_st = clock();
    for (int i = 0; i < num_workers; i++) {
        double loss_tmp = squarehingeloss_dense(&(*sub_W)[i], data, sparam.lambda, sparam.num_data_total);
        if (loss_tmp < loss) {
            model_W_dense = (*sub_W)[i];
            loss = loss_tmp;
            wid = i;
        }
    }
    double loss_t = (clock()- loss_st)/ CLOCKS_PER_SEC;
    cout<<"wid:"<<wid<<" time used to compute loss:"<< loss_t<<" s"<<endl;*/


    VectorXd model_W_dense_abs;
    model_W_dense_abs = model_W_dense.cwiseAbs();
    getTopK(&model_W_dense_abs, k, idx, val);
    vector<T> tripletList; //(k);
    tripletList.reserve(k);
    SpMat model_W_temp(dim, 1);
    for (int i = 0; i < k; i++) {
        int index = idx[i];
        tripletList.push_back(T(idx[i], 0, model_W_dense[index]));
        //tripletList[i] = T(idx[i], 0, model_W_dense[index]);
    }
    model_W_temp.setFromTriplets(tripletList.begin(), tripletList.end());
    *model_W = model_W_temp;
}

void worker_run(worker_parameter wparam, Batch_thred * thread_data, VectorXd * sub_W, VectorXd * subalpha, VectorXd * subXalpha, SpMat * W_prev ){
    int mBatch_idx = -1;
    VectorXd temp(wparam.dim);
    temp = * W_prev;
    VectorXd subalphanew(wparam.sapBlock);
    MatrixXd subXalpha_update(1,wparam.dim);
    VectorXd deta(wparam.sapBlock);
    VectorXd miniBatchY(wparam.sapBlock);
    SpMat miniBatchX(wparam.sapBlock,wparam.dim);
    vector<MiniBatch> * miniBatches;
    MiniBatch * miniBatch;
    SparseMatrix<double>X_transpose(wparam.dim,wparam.sapBlock);
    double normalizor = wparam.lambda+wparam.eta2;
    //temp = VectorXd(*W_prev);
    for(int i = 0; i < wparam.maxIter; i++){
        mBatch_idx++;
        if (mBatch_idx > thread_data->num_mBatches - 1){
            mBatch_idx = 0;
        }
        miniBatches = &(thread_data->mBatches);
        miniBatch = &(*miniBatches)[mBatch_idx];
        miniBatchX = miniBatch->X;
        miniBatchY = miniBatch->Y;
        *subalpha = miniBatch->alpha;
        //cout<<"alpha: "<<subalphaprev[0]<<endl;
        subalphanew = gradientDescentDualSquaredloss(wparam, subalpha, &temp, wparam.eta1, &miniBatchX, &miniBatchY); ///////
        miniBatch->alpha = subalphanew;

        deta = subalphanew - *subalpha;
        //X_transpose = miniBatchX.transpose();
        subXalpha_update =  (deta.transpose() * miniBatchX); // so slow !!!!!!!!!!!!!!!!!!!!!!
        *subXalpha = (*subXalpha) + subXalpha_update.transpose() /wparam.ndata; //wparam.sapBlock;
        temp = 0.5 * (2 * wparam.eta2 * temp - (*subXalpha))/ normalizor;
    }
    *sub_W = temp;
}


void ParallelSDAL0Method1(server_parameter sparam, worker_parameter wparam, int num_workers, vector<Batch_thred> * data, SpMat * model_W, clock_t * time_extra)
{
    printf("solve funtion\n");
    clock_t solv_st = clock();
    int s_itersize = sparam.maxIter;
    int w_itersize = wparam.maxIter;
    int sparsedim = sparam.sparseDim;
    int dim = sparam.dim;
    double loss_prev = 10;
    double loss = 1;
    vector<VectorXd> sub_W(sparam.numWorker);
    vector<VectorXd> subXalpha(sparam.numWorker);
    vector<VectorXd> subalpha(sparam.numWorker);
    //initialize with 0
    for(int i = 0; i < num_workers; i++){
        sub_W[i] = VectorXd::Zero(wparam.dim);
        subXalpha[i] = VectorXd::Zero(wparam.dim);
        subalpha[i] = VectorXd::Zero(wparam.sapBlock);
    }

    for (int i = 0; i < s_itersize; i++){
        #pragma omp parallel for num_threads(num_workers)
        for(int j = 0 ; j < num_workers; j++){
            Batch_thred thread_data = (*data)[j];
            worker_run(wparam, &thread_data, &(sub_W[j]),&(subalpha[j]),&(subXalpha[j]), model_W);
        }

        //update model_W according to the subW given by workers
        server_run(&sub_W,sparsedim, dim, num_workers, model_W, sparam, data);

        //calculate the loss
        clock_t st = clock();
        loss = squarehingeloss(model_W,data,sparam.lambda,sparam.num_data_total);
        *time_extra = (*time_extra) + clock() - st;
        //cout<<"Iter"<<i<<":  loss is "<<loss <<"  ("<<*time_extra<<")"<< "deta:"<<loss_prev - loss<<" "<<sparam.epsacc<<endl;

        clock_t iter_t = clock();
        double iter_time = (iter_t  - solv_st - *time_extra )/ CLOCKS_PER_SEC;
        cout<<"Iter"<<i<<":  loss is "<<loss <<"  time: "<<iter_time<<" s"<<endl;

        /*if(loss_prev - loss < sparam.epsacc){
            cout<<"converge!"<<endl;
            break;
        }
        else {
            loss_prev = loss;
        }*/
    }
}