#!/bin/bash
curl -L -o ./archive.zip https://www.kaggle.com/api/v1/datasets/download/rm1000/brain-tumor-mri-scans

# unzip the downloaded file to ./data
unzip -o ./archive.zip -d ./data

# remove the downloaded zip file
rm ./archive.zip
