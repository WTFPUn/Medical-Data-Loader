# Medical Data Loader
This project is a simple data loader for medical data. Detail will be added soon.

## Installation
```bash
cd medical-data-loader
pip install requirements/requirements.txt
```

## Usage
first, You need to download your data and put it in your data directory. Format of the data should be like this:
```
path/to/data
    /data
        img0001.nii.gz
        img0002.nii.gz
        ...
    /label
        label0001.nii.gz
        label0002.nii.gz
        ...

```

Then run python command to create metadata to split the data into train, validation, and test set.
```python
from DataEngine import get_split_meta_data

data_dir = 'path/to/data'

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

get_split_meta_data(data_dir, "name_of_metadata_file.json", [train_ratio, val_ratio, test_ratio])
```

Then you can use Dataengine by following [this file](data_template.ipynb)