//
// Created by lezi on 10/29/16.
//
#ifndef SPARSE_OPT_PARALLEL_SINGLE_MACHINE_SOLVE_H
#define SPARSE_OPT_PARALLEL_SINGLE_MACHINE_SOLVE_H

#include "common.h"
#include <Eigen/Dense>
#ifdef __cplusplus
extern "C" {
#endif

void ParallelSDAL0Method1(server_parameter sparam, worker_parameter wparam, int num_workers, vector<Batch_thred> *data, SpMat * model_W, clock_t * time_extra);

#ifdef __cplusplus
}
#endif

#endif //SPARSE_OPT_PARALLEL_SINGLE_MACHINE_SOLVE_H
