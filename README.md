# SIFT1B extract

Some python files to extract data from SIFT1B dataset (http://corpus-texmex.irisa.fr/) into `.npy` chunk files

Inspired by https://github.com/milvus-io/bootcamp/tree/master/benchmark_test

## How to use

1. Download the SIFT1B dataset (may take a long time, around 100GB to download):
```shell
mkdir dataset-1B
cd dataset-1B
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
```
2. Extract the dataset (requires 150GB additional)
```shell
gzip -kd bigann_query.bvecs.gz
tar -xf bigann_gnd.tar.gz
gzip -kd bigann_base.bvecs.gz
```

3. Run the script (may need adjustments to some parameters)
```shell
python main.py
```
