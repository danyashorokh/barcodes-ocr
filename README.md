## Modelling

Barcodes text recognition.

### Dataset

![](./assets/barcodes.png)

Download the dataset from [here](https://disk.yandex.ru/d/nk-h0vv20EZvzg)

### Pipeline preparing

1. Create and activate environment
    ```
    python3 -m venv /path/to/new/virtual/environment
    ```
    ```
    source /path/to/new/virtual/environment/bin/activate
    ```

2. Install packages

    from activated environment:
    ```
    pip install -r requirements.txt
    ```

3.  Split dataset on train/val/test samples:
    ```
    ROOT_PATH=./ python train_test_split.py -i path/to/dataframe -o path/to/save/splited/dataframes
    ```

4. ClearML setting
    - [in your ClearML profile](https://app.community.clear.ml/profile) click "Create new credentials"
    - write down `clearml-init` and continue by instruction steps

### Training on generated data

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/ocr python train_on_generator.py configs/simple_config.py
```

### Training on real data

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/ocr python train.py configs/simple_config.py
```

### Export to script

```
notebooks/to-script.ipynb
```

### Download weights

```
dvc pull -R weights -r storage (for ssh)
dvc pull -R weights -r gstorage (for gdrive)
```

### Predict

```
python predict.py -w <model weghts> -i <path to image>
```

### Useful links
* https://pyimagesearch.com/2014/12/15/real-time-barcode-detection-video-python-opencv/
* https://www.labeljoy.com/ru/podderzhivat/tipy-shtrikh-kodov/