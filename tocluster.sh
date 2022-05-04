#!/bin/bash

scp -P 2222 -r ./run_model_v4.py devdokimov@cluster.hpc.hse.ru:/home/devdokimov/TPC-FastSim/
scp -P 2222 -r ./models devdokimov@cluster.hpc.hse.ru:/home/devdokimov/TPC-FastSim/
scp -P 2222 -r ./metrics devdokimov@cluster.hpc.hse.ru:/home/devdokimov/TPC-FastSim/