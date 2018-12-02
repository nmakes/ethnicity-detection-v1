# Ethnicity Detection Using Convolutional Neural Networks

We train 4 CNN architectures for 30 epochs and comapre the classification scores. The data used is taken from [UTKFace dataset](https://susanqq.github.io/UTKFace/). Each network achieves 75%+ peak accuracy in classification.

Venkat, N., Srivastava, S. (2018). Ethnicity Detection using Deep Convolutional Neural Networks. 
[Download the report here](doi.org/10.13140/RG.2.2.34591.20642).

## STEP 1: Getting the models and train-test split
You can [download models and train-test split here](https://drive.google.com/drive/folders/18CBSmBZo0gjyGNdMBBRrrxD0YvjelOmI?usp=sharing)

## STEP 2: Installing Requirements
	
	python3 -m pip install -r requirements.txt

## STEP 3: Running
Executing the code for training and testing

	python3 main.py

Executing the code to view results on the trained models

	python3 accuracies.py