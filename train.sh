#!/bin/bash

docker run --rm -v $(pwd)/work_dir:/root -it spp_lstm python3 train.py ++training.epochs=100 ++stock_code="3092.T"
