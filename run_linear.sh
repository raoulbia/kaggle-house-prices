#!/bin/bash -x

python3 src/main.py \
    --dataset houses \
    --train-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-house-prices/local-data/house-price-train-clean.csv \
    --test-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-house-prices/local-data/house-price-test-clean.csv \
    --results-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-house-prices/local-data/house-price-results.csv \
    --test-size 0.3 \
    --num-iters 1000 \
    --cost-history-plot -1 \
    --learning-curve 1 \
    --learning-rate 0.001 \
    --reg-param 0.1 \
