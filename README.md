# CS674
You can use these codes to reproduce the results that I included in the paper. You are welcomed to modify the code, e.g tuning the paramters, to verify your own idea.

ksparse_singleMachine is for DIHT algorithm (centralized optimization).
sparse_opt_parallel_single_machine is for D-DIHT (distributed version on eight workers).

Before running the code, you need to download the data for each and put the data in "./data" folder. Data can be downloaded by the following links:
data for sparse_opt_parallel_single_machine:
https://www.dropbox.com/sh/uaspmpfjgz448si/AADHTe05QJPcjdrbTVv32FXKa?dl=0
it contains "datalist" and "miniBatches", which should be put in ./sparse_opt_parallel_single_machine/data folder

data for ksparse_singleMachine:
https://www.dropbox.com/sh/1qkaaz42mq0w0v5/AAA3O7LHZHdtj1rgd543LGeba?dl=0
it contains "data_all.txt" which should be stored in ./ksparse_singleMachine/data folder

run the binary code on command line:
DIHT:
./ksparse_singleMachine -n 1 -d 47236  -k 10000 -l 0.00001  -N 480000 -m 480000  /home/lezi/ClionProjects/ksparse_singleMachine/datalist  model

D-DIHT:
sparse_opt_parallel_single_machine -d 47236 -k 2000 -l 0.00001 -n 8 -N 480000 -m 60000 /home/lezi/ClionProjects/sparse_opt_parallel_single_machine/data/datalist model

The code is implemented by c++ under unbuntu14.04 and it relizes on eigen library.
I am keeping maintain it. If you have any question, please email me: lw462@cs.rutgers.edu
