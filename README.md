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
1.  Created `dataset.py`
  * Split the one image dataset folder(`1. Original Images`) to train, val, test folders(all are splited in `train_data` folder) beforehand.
  * created `ChallengeDataset` for challenge dataset. It is imported in `utils.py` and will be returned Dataset for train/validation by calling `configure_train` and `configure_vallidation` funtions. `get_train` and `get_validation` will return each training dataset and validation dataset in DataBase class named `DiabeticRetinopathyGradingDataset` 
 
2.  Modified `moco.py` to `model.py`
3.  Lots of modification in `utils.py`
  * Import `ChallengeDataset` in `utils.py` 
  * Created class named `DiabeticRetinopathyGradingDataset` with defined class `DatasetBase`.
  * `ChallengeDataset` is returned for train/validation by calling `configure_train` and `configure_vallidation` funtions. `get_train` and `get_validation` will return each training dataset and validation dataset in DataBase class named `DiabeticRetinopathyGradingDataset` 
  ``` 
  from dataset import ChallengeDataset 
  ...
  challenge_default_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

@attr.s(auto_attribs=True, slots=True)
class DiabeticRetinopathyGradingDataset(DatasetBase):

    transform_train: Callable[[Any], torch.Tensor] = challenge_default_transform
    transform_test: Callable[[Any], torch.Tensor] = challenge_default_transform
    _train_ds: Optional[torch.utils.data.Dataset] = None
    _validation_ds: Optional[torch.utils.data.Dataset] = None
    _test_ds: Optional[torch.utils.data.Dataset] = None


    def configure_train(self):
        ...
        return ChallengeDataset(img_path, csv_file_path, transform=self.transform_train)

    def configure_validation(self):
        ...
        return ChallengeDataset(img_path, csv_file_path, transform=self.transform_test)

    def get_train(self) -> torch.utils.data.Dataset:
        ...
        return self._train_ds

    def get_validation(self) -> torch.utils.data.Dataset:
        ...
        return self._validation_ds
  
  ```
  * In DataBase class named `DiabeticRetinopathyGradingDataset`, each transform modules for training and testing are differently applied and by calling `get_class_dataset` function and defined by `get_class_transforms` in class `MoCoTransforms`.
   * The reason why defining two different transform for train and test is that transform for train needs to be joint embedding(image data is saved in list data structure) so two transform are needed to be applied for two different images in the list. However, for test one, it just needs one transform for a single image. 
 ``` 
 
def get_class_dataset(name: str) -> DatasetBase:
    if name == "stl10":
        transform_train, transform_test = get_class_transforms(96, 128)
        return STL10LabeledDataset(transform_train=transform_train, transform_test=transform_test)
    elif name == "imagenet":
        transform_train, transform_test = get_class_transforms(224, 256)
        return ImagenetDataset(transform_train=transform_train, transform_test=transform_test)
    elif name == "cifar10":
        transform_train, transform_test = get_class_transforms(32, 36)
        return CIFAR10Dataset(transform_train=transform_train, transform_test=transform_test)
    elif name == "challenge":
        transform_train, transform_test = get_class_transforms(224, 256)
        return DiabeticRetinopathyGradingDataset(transform_train=transform_train, transform_test=transform_test)
    raise NotImplementedError(f"Dataset {name} not defined")
 ```
4.  Parameter changes in `main.py`
  * Add new EarlyStopping parameter `patience` to trainer 
  * `VICRegParams`
