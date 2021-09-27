#!/bin/bash
kaggle competitions download -c competitive-data-science-predict-future-sales -p data/src/
unzip  data/src/competitive-data-science-predict-future-sales.zip -d data/src/
rm -v data/src/*.zip