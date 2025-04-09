# [archived] Ethnicity Detection Using Convolutional Neural Networks

Thank you for visiting this page. This project is no longer being maintained.

Quick notes on what this project was about:

* We trained 4 CNN architectures for 30 epochs and comapred the classification scores on the [UTKFace dataset](https://susanqq.github.io/UTKFace/).
* To run:
  * Install requirements `python3 -m pip install -r requirements.txt`
  * Obtain the [UTKFace dataset](https://susanqq.github.io/UTKFace/), and use [these functions](https://github.com/nmakes/ethnicity-detection-v1/blob/031944e308ef4274129074c8c62f58d18a1eb48e/main.py#L395-L402) to generate the train/test split.
  * Run `python3 main.py` for training
  * Run `python3 accuracies.py` for computing metrics
