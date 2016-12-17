#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include "common.h"
#include "include/solve.h"

using namespace std;

server_parameter sparam;
worker_parameter wparam;

void exit_with_help()
{
    printf(
            "Usage:\n"
            "-k the fixed sparse dimension\n"
            "-l lambda\n"
            "-n num_worker number of workers(threads)\n"
    );

    exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
    int i;
    void (*print_func)(const char*) = NULL;	// default printing to stdout

    //default value for server;
    sparam.beta = 1;
    sparam.maxIter = 100;
    sparam.epsacc =  0.0001;
    sparam.num_data_total = 480000;

    //default value for worker
    wparam.maxIter = 20;
    wparam.epsacc = 0.0005;
    wparam.sapBlock = 3000;
    wparam.eta1 = 0.01;
    wparam.eta2 = 0.000001;

    for(i=1;i<argc;i++) {
        if (argv[i][0] != '-') break;
        if (++i >= argc)
            exit_with_help();
        switch (argv[i - 1][1]) {
            case 'd':
                sparam.dim = atoi(argv[i]);
                wparam.dim = atoi(argv[i]);
                break;
            case 'k':
                sparam.sparseDim = atoi(argv[i]);
                wparam.sparseDim = atoi(argv[i]);
                printf("sparse DIM is %d\n", wparam.sparseDim);
                break;
            case 'l':
                sparam.lambda = atof(argv[i]);
                wparam.lambda = atof(argv[i]);
                printf("lambda is %f\n", sparam.lambda);
                break;
            case 'n':
                sparam.numWorker = atoi(argv[i]);
                printf("number of workers is %d\n", sparam.numWorker);
                break;
            case 'N':
                sparam.num_data_total = atoi(argv[i]);
                printf("number of data in total is %d\n", sparam.num_data_total);
                break;
            case 'm':
                wparam.ndata = atoi(argv[i]);
                printf("number of data for each worker is %d\n", wparam.ndata);
                break;
            default:
                fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
                exit_with_help();
        }
    }

    svm_set_print_string_function(print_func);

    // determine filenames
    if(i>=argc)
        exit_with_help();

    strcpy(input_file_name, argv[i]);

    if(i<argc-1) {
        strcpy(model_file_name, argv[i + 1]);
        cout<<"model file: "<<model_file_name<<endl;
    }
    else
    {
        char *p = strrchr(argv[i],'/');
        if(p==NULL)
            p = argv[i];
        else
            ++p;
        sprintf(model_file_name,"%s.model",p);
        cout<<"model file: "<<model_file_name<<endl;
    }
}

int main(int argc, char * argv[]) {
    char input_file_name[1024];
    char model_file_name[1024];
    parse_command_line(argc, argv, input_file_name,model_file_name);
    std::cout << "Hello, World!" << std::endl;
    int num_worker = sparam.numWorker; // num_threds
    int mBatch_sz = wparam.sapBlock;
    int dataDim = wparam.dim;
    SpMat model_w(dataDim,1);
    model_w.setZero();
    vector<Batch_thred> data;
    cout<<"loading data...."<<endl;
    clock_t begin_time = clock();
    data = loadData(num_worker, mBatch_sz,dataDim,input_file_name);
    double dur = ( clock () - begin_time ) /  CLOCKS_PER_SEC;
    cout<<"data is ready!  It takes:"<< dur<< "s."<<endl;
    cout<<"solving...."<<endl;
    begin_time = clock();
    clock_t time_extra = 0;
    ParallelSDAL0Method1(sparam,wparam,num_worker,&data,&model_w,&time_extra);
    dur = ( clock () - begin_time ) /  CLOCKS_PER_SEC;
    double dur_extra = time_extra / CLOCKS_PER_SEC;
    cout<<"finished! Solving the problem takes total"<<dur<<"s.(Extra Time: "<<dur_extra<<"s)"<<endl;
    return 0;
}


