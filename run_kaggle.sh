#!/bin/bash -x

# kernprof -lv src/main_kaggle.py \
time python3 src/main_kaggle.py \
    --train-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-house-prices/local-data/house-price-train-clean.csv \
    --test-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-house-prices/local-data/house-price-test-clean.csv \
    --results-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-house-prices/local-data/house-price-results.csv \
    --num-iters 1500 \
    --learn-hyperparamters 1 \
    --_alpha 0.03 \
    --_lambda 0.0001 \

