# Tensorflow Data Agumentation
Classify 17 classes of flowers. [Link](https://www.kaggle.com/allenjie/flowers17).

Each class only contains 60 images and has to be upsampled using Data Augmentation.

## Setup
Generate Virtual environment
```bash
python3 -m venv ./venv
```
Enter environment
```bash
source venv/bin/activate
```
Install required libraries
```bash
pip install -r requirements.txt
```
Train model
```bash
python3 train.py -d <path to dataset>
```
