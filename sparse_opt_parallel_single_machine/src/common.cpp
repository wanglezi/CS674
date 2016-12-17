//
// Created by lezi on 10/29/16.
//
#include "common.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
using namespace std;

using namespace Eigen;
typedef Eigen::Triplet<double> T;


static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
    int len;

    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}


static void print_string_stdout(const char *s)
{
    fputs(s,stdout);
    fflush(stdout);
}
static void (*svm_print_string) (const char *) = &print_string_stdout;

void svm_set_print_string_function(void (*print_func)(const char *))
{
    if(print_func == NULL)
        svm_print_string = &print_string_stdout;
    else
        svm_print_string = print_func;
}

/*check whether data are correct*/
void check_file(string file_list, int numWorker, int mBatch_sz) {
    ifstream inFile(file_list);
    if (!inFile) {
        std::cout << "Failed to open file" << std::endl;
        exit(0);
    }
    cout<<"datalist:"<< file_list<<endl;
    long n_thread = count(std::istreambuf_iterator<char>(inFile),
                          std::istreambuf_iterator<char>(), '\n');
    if (n_thread != numWorker){
        cout<<"worker number does not match!"<<n_thread<<" vs. "<<numWorker<<endl;
        exit(1);
    }
    inFile.close();
}
vector<Batch_thred> loadData(int num_worker, int mBatch_sz, int dataDim, char* datalist)
{
    vector<Batch_thred> data(num_worker);
    check_file(datalist,num_worker,mBatch_sz);
    char thread_data[1024];
    char *endptr;
    char *idx, *val, *label;
    int inst_max_index, num_mBatches;
    max_line_len = 1024;
    line = Malloc(char,max_line_len);

    ifstream infile(datalist);
    for(int i = 0; i < num_worker; i++) {
        Batch_thred data_w;
        infile >> thread_data;
        cout<<"loading data for worker_"<<i<<endl;// get one worker file
        FILE *fp = fopen(thread_data,"r");
        if(fp == NULL)
        {
            fprintf(stderr,"can't open input file %s\n",thread_data);
            exit(1);
        }
        int num_data = 0;
        while(readline(fp)!=NULL)
        {
            char *p = strtok(line," \t"); // label

            // features
            while(1) {
                p = strtok(NULL, " \t");
                if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                    break;
            }
            ++num_data;
        }
        rewind(fp);

        num_mBatches = floor(num_data / mBatch_sz); // count how many mBatches
        //cout<<"num_mBatches: "<<num_mBatches<<endl;
        vector<MiniBatch> mBatches;
        for (int i = 0; i < num_mBatches; i++) {
            MiniBatch miniBatch;
            miniBatch.dataDim = dataDim;
            miniBatch.num = mBatch_sz;
            SpMat xtemp(mBatch_sz, dataDim); //sparse matrix for X in one batch
            vector<T> tripletList;
            VectorXd ytemp(mBatch_sz);//labels for Y in one batch
            int x_elements = 0;
            for (int j = 0; j < mBatch_sz; j++) {
                ///* assign matrix */
                inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
                readline(fp);
                label = strtok(line," \t\n");
                if(label == NULL){
                    cout<<"empty line!"<<endl;
                    exit(1);
                }

                //assign label Y[i]
                ytemp[j]= strtod(label,&endptr);
                if(endptr == label || *endptr != '\0') {
                    cout << "something wrong with the line" << endl;
                    exit(1);
                }

                //parsing X
                while(1) {
                    idx = strtok(NULL, ":");
                    val = strtok(NULL, " \t");

                    if (val == NULL)
                        break;

                    errno = 0;
                    int index = (int) strtol(idx, &endptr, 10);
                    index = index - 1;
                    if (endptr == idx || errno != 0 || *endptr != '\0' || index <= inst_max_index) {
                        cout << "something wrong with the line parsing" << endl;
                        exit(1);
                    } else {
                        inst_max_index = index;
                    }

                    errno = 0;
                    double value = strtod(val, &endptr);
                    if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr))) {
                        cout << "something wrong with the value\n";
                        exit(1);
                    }
                    ++x_elements;
                    tripletList.push_back(T(j,index,value));
                }

            }

            xtemp.setFromTriplets(tripletList.begin(), tripletList.end());
            //xtemp.setFromTriplets(tripletList_temp.begin(), tripletList_temp.end());
            //cout<<"finish assigning X and Y"<<endl;
            miniBatch.X = xtemp;
            miniBatch.Y = ytemp;
            miniBatch.alpha = VectorXd::Zero(mBatch_sz);
            mBatches.push_back(miniBatch);
        }
        //cout<<"finish parsing data for worker_"<<i<<", num_mBatches:"<<num_mBatches<<endl;
        data_w.num_mBatches = num_mBatches;
        data_w.mBatches = mBatches;
        //data.push_back(data_w);
        data[i] = data_w;
       // cout<<data_w.num_mBatches<<endl;
       // cout<<data[0].num_mBatches<<endl;
        fclose(fp);
    }
    int total_num = data[0].num_mBatches * num_worker * mBatch_sz;
    cout<<"total data:"<<total_num<<endl;
    infile.close();
    return data;
}



VectorXd gradientDescentDualSquaredloss(worker_parameter wparam ,VectorXd * subalphaprev, VectorXd * W_prev, double eta1, SpMat * miniBatchX, VectorXd * miniBatchY){
    int sapBlock = wparam.sapBlock;
    VectorXd grad(sapBlock);
    VectorXd XW(sapBlock);
    XW = (*miniBatchX) * (*W_prev);
    grad =  XW  - (*miniBatchY) - (*subalphaprev)*0.5;
    VectorXd subalphanew(sapBlock);
    subalphanew = (*subalphaprev) + eta1 * grad;
    for(int i = 0 ; i < sapBlock; i++)
        subalphanew[i] = ((*miniBatchY)[i] * subalphanew[i] > 0) ? 0 : subalphanew[i];
    return subalphanew;
}


double squarehingeloss(SpMat * w, vector<Batch_thred>* data,  double lambda, int N){
    int num_w = (*data).size();
    VectorXd hinge;
    hinge = VectorXd::Zero(N);
    int cnt = 0;
    for(int i = 0; i < num_w; i++){
        vector<MiniBatch> miniBatches = (*data)[i].mBatches;
        int num_mBatches = miniBatches.size();
        for(int j = 0; j < num_mBatches; j++){
            SpMat miniBatchX = miniBatches[j].X;
            VectorXd miniBatchY = miniBatches[j].Y;
            int mBatches_sz = miniBatches[j].num;
            SpMat XW(mBatches_sz, 1);
            XW = (miniBatchX) * (*w);
            VectorXd hinge_temp(mBatches_sz);
            hinge_temp = VectorXd(XW);
            for(int k = 0; k < mBatches_sz; k++){
                double temp = 1 - miniBatchY[k]*hinge_temp[k];
                hinge[cnt] = ( temp > 0) ? temp : 0;
                cnt++;
            }
        }
    }
    double w_norm = (*w).squaredNorm();
    double hinge_norm = hinge.squaredNorm();
    double loss = hinge_norm/N + lambda * w_norm;
    return loss;
}

double squarehingeloss_dense(VectorXd * w, vector<Batch_thred>* data,  double lambda, int N){
    int num_w = (*data).size();
    VectorXd hinge;
    hinge = VectorXd::Zero(N);
    int cnt = 0;
    for(int i = 0; i < num_w; i++){
        vector<MiniBatch> miniBatches = (*data)[i].mBatches;
        int num_mBatches = miniBatches.size();
        for(int j = 0; j < num_mBatches; j++){
            SpMat miniBatchX = miniBatches[j].X;
            VectorXd miniBatchY = miniBatches[j].Y;
            int mBatches_sz = miniBatches[j].num;
            SpMat XW(mBatches_sz, 1);
            XW = (miniBatchX) * (*w);
            VectorXd hinge_temp(mBatches_sz);
            hinge_temp = VectorXd(XW);
            for(int k = 0; k < mBatches_sz; k++){
                double temp = 1 - miniBatchY[k]*hinge_temp[k];
                hinge[cnt] = ( temp > 0) ? temp : 0;
                cnt++;
            }
        }
    }
    double w_norm = (*w).squaredNorm();
    double hinge_norm = hinge.squaredNorm();
    double loss = hinge_norm/N + lambda * w_norm;
    return loss;
}