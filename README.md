# Timeseries Neural Mutual Information Fields (TNMIFs)

Requirements:

pytorch-cuda=11.7

tensorboard=2.15.0

tinycudann=1.7

xarray=2023.1.0

dask=2023.11.0

Install requirements:
```
conda activate env
conda install --file requirements.txt
```
Randomly sampling reference and query positions and computing ground truths:
```
python mi_sampler.py --source_path path/to/folder/of/.nc/files
```

Run `TrainingScript.py`:
```
python TrainingScript.py
```
