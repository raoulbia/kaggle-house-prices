#!/bin/bash -x

# kernprof -lv src/main.py \
time python3 src/main.py \
    --dataset houses \
    --train-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-house-prices/local-data/house-price-train-clean.csv \
    --test-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-house-prices/local-data/house-price-test-clean.csv \
    --results-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-house-prices/local-data/house-price-results.csv \
    --test-size 0.3 \
    --num-iters 1500 \
    --basic -1 \
    --learning-curve -1 \
    --validation-curve 1 \
    --_alpha 0.3 \
    --_lambda 1.2 \
