#!/bin/sh

rm output/results.csv

python project_main.py -i data/soybean-large.data -r 0 -c 0 -l 0

python project_main.py -i data/echocardiogram.data -l 1 --skip-columns=0,10

python project_main.py -i data/house-votes-84.data -l 0

python project_main.py -i data/bridges.data.version1.data -l 9 --skip-columns=0,2

python project_main.py -i data/water-treatment.data --skip-columns=0

python project_main.py -i data/hepatitis.data -l 0

python project_main.py -i data/iris.data -r 0,1.01,0.1 -c 0,8,1 --seed=0
