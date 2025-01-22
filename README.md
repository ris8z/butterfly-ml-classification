# Butterfly classification project

## setup

### virtual env

#### Prerequisites
You need to have installed on your machine:
- python3
- pip

#### How to create it 

create a python virtual env with:
```bash
#linux
python3 -m venv env

#windows
python -m venv env
```

active it with:
```bash
#linux
source env/bin/active

#windows
env\Scripts\activate
```

then load all the external modules and dependencies with:
```bash
pip install -r requirements.txt
```

### create all the 3 different data_sets
You just need to run all the script in the data_processing folder (use only python for windows)

```bash
python3 01_checking_data.py
python3 02_splitting_dataset.py
python3 03_data_augmentation.py
python3 04_checking_augmented_data.py
python3 05_balancing_data.py
python3 06_checking_balanced_data.py
```

## How to play with the code
In both scirpts:
- ai_models/mobilenet_logisticregression.py
- ai_models/space_vector_machine.py

there is a section for choosing which dataset denoted by 3 different codes:
- *00* (simple split 70% of each calss in the train set and the rest 30% in the train set) 
- *01* (augmented data for each imgae generate 3 new images)
- *02* (augmented and balanced data each call has 100 entris)

To select one type *uncomment the 3 lines after the line with the code*
and **make sure that all the other are commented**

### Examples
this one is good because just one is uncommented
```python
#######################################################################
# chose which type of dataset to use (check docs to create the data)  #
#######################################################################

# base split case (code 00)
#TRAIN_CSV = os.path.join(DATASET_FOLDER, "train_set.csv")
#TRAIN_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "images")
#MODEL_PATH = os.path.join(os.path.dirname(__file__), "./../trained/mn_lr_00.joblib")

# tipled the data set with data augmentation case (code 01)
#TRAIN_CSV = os.path.join(DATASET_FOLDER, "train_set_augmented.csv")
#TRAIN_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "augmented_images")
#MODEL_PATH = os.path.join(os.path.dirname(__file__), "./../trained/mn_lr_01.joblib")

# data augmented and balanced (code 02)
TRAIN_CSV = os.path.join(DATASET_FOLDER, "train_set_augmented_balance.csv")
TRAIN_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "balance_augmented_images")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "./../trained/mn_lr_02.joblib")

#######################################################################
```
this one is not good because just more then one option uncommented
```python
#######################################################################
# chose which type of dataset to use (check docs to create the data)  #
#######################################################################

# base split case (code 00)
#TRAIN_CSV = os.path.join(DATASET_FOLDER, "train_set.csv")
#TRAIN_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "images")
#MODEL_PATH = os.path.join(os.path.dirname(__file__), "./../trained/mn_lr_00.joblib")

# tipled the data set with data augmentation case (code 01)
TRAIN_CSV = os.path.join(DATASET_FOLDER, "train_set_augmented.csv")
TRAIN_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "augmented_images")
#MODEL_PATH = os.path.join(os.path.dirname(__file__), "./../trained/mn_lr_01.joblib")

# data augmented and balanced (code 02)
TRAIN_CSV = os.path.join(DATASET_FOLDER, "train_set_augmented_balance.csv")
TRAIN_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "balance_augmented_images")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "./../trained/mn_lr_02.joblib")

#######################################################################
```

## How it works
Once you run a model with the "python3" and the name of the module, is going to train and than evaluate on you machine
and save the trained module in the trained folder.

All the result is going to be printed to the screen, the metrics that we used are:
- Total accuracy of the module 
- precision 
- recall    
- f1-score  
- support (n of sample in the test set)
- accuracy (of each class) 

A copy of the results found are in the results folder.
Each name is the short for the alogithm used and the code of the dataset used (00, 01, 02)
