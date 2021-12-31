#!/bin/bash

docker run --rm -v $(pwd)/work_dir:/root/stock_price_prediction/output_dir -it spp_lstm python3 predict.py ++stock_code="3092.T" ++eval.model_dir=/root/stock_price_prediction/output_dir/my_model
