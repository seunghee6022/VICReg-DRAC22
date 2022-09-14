# VICReg-DRAC22: VICReg Application to Diabetic Retinopathy Analysis Challenge 2022
VICReg(Variance-Invariance-Covariance Regularization for Self-Supervised Learning) Implementation of the Diabetic Retinopathy Image classification task(task3) hosted by Diabetic Retinopathy Analysis Challenge (DRAC22)
___

### Original Information Links

* VIGReg github link:
https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

* Link for the Retinopathy Analysis Challenge 2022:
https://drac22.grand-challenge.org/

___
### Main Changes
2.  Created `dataset.py`
  * Split the one image dataset folder(`1. Original Images`) to train, val, test folders(all are splited in `train_data` folder)
  * created `ChallengeDataset` for challenge dataset
2.  Modified `moco.py` to `model.py`
4.  Lots of modification in `utils.py`
  * 
5.  Parameter changes in `main.py`
  * Add new EarlyStopping parameter `patience` to trainer 
  * `VICRegParams`
