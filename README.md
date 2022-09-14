# VICReg-DRAC22: VICReg Application to Diabetic Retinopathy Analysis Challenge 2022
VICReg(Variance-Invariance-Covariance Regularization for Self-Supervised Learning) Implementation of the Diabetic Retinopathy Image classification task(task3) hosted by Diabetic Retinopathy Analysis Challenge (DRAC22)
___

### Original Code Github Links

* VIGReg github link:
https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

### Information about the Retinopathy Analysis Challenge 2022
* Link for the Retinopathy Analysis Challenge 2022:
https://drac22.grand-challenge.org/

___

### Setup

The setup below works on silicon Mac. Windows should work in a similar fashion. Just give it a quick google.
```bash
python3 -m venv <directory name>
source <directory name>/bin/activate
pip install -r requirements.txt
```
The dataset should be stored in a folder called `data` in the same location as `main.py` and all images should reside is a child folder called `C. Diabetic Retinopathy Grading/train_data`. These names can also be adjusted in the config file. You can read more about the dataset in the corresponding section below.
___
### Run

```python
python main.py
```
___
### Main Changes
1.  Created `dataset.py`
  * Split the one image dataset folder(`1. Original Images`) to train, val, test folders(all are splited in `train_data` folder) beforehand.
  * created `ChallengeDataset` for challenge dataset. It is imported in `utils.py` and will be returned Dataset for train/validation by calling `configure_train` and `configure_vallidation` funtions. `get_train` and `get_validation` will return each training dataset and validation dataset in DataBase class named `DiabeticRetinopathyGradingDataset` 
 
2.  Parameter changes in `main.py`
  * Add new EarlyStopping parameter `patience` to trainer 
  * `VICRegParams`
3.  Modified `moco.py` to `model.py`
 * Model class name is `SelfSupervisedMethod` and here specify the custom challenge dataset imported from `utils.DatasetBase`
 ```python
 class SelfSupervisedMethod(pl.LightningModule):
    model: torch.nn.Module
    dataset: utils.DatasetBase
    hparams: AttributeDict
    embedding_dim: Optional[int]

    def __init__(
        self, 
        ...
 ```
 * Add `Adam optimizer` in `configure_optimizers`
 ```python
  def configure_optimizers(self):
        ...
        if self.hparams.optimizer_name == "sgd":
            optimizer = torch.optim.SGD
        elif self.hparams.optimizer_name == "adam":
            optimizer = torch.optim.Adam
          
        elif self.hparams.optimizer_name == "lars":
            optimizer = partial(LARS, warmup_epochs=self.hparams.lars_warmup_epochs, eta=self.hparams.lars_eta)
        else:
            raise NotImplementedError(f"No such optimizer {self.hparams.optimizer_name}")

        encoding_optimizer = optimizer(
            param_groups,
            lr=self.hparams.lr,
            # momentum=self.hparams.momentum, # Momentum is not needed for Adam Optimizer
            weight_decay=self.hparams.weight_decay,
        )

        ...
        return [encoding_optimizer], [self.lr_scheduler]
 ```
 * `_get_vicreg_loss` function calculates the VICReg loss
  ```python

       
 
    def _get_vicreg_loss(self, z_a, z_b, batch_idx):
        assert z_a.shape == z_b.shape and len(z_a.shape) == 2
        print("Start calculating the VICReg loss")
        
        # invariance loss
        loss_inv = F.mse_loss(z_a, z_b)

        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.hparams.variance_loss_epsilon)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.hparams.variance_loss_epsilon)
        loss_v_a = torch.mean(F.relu(1 - std_z_a))
        loss_v_b = torch.mean(F.relu(1 - std_z_b))
        loss_var = loss_v_a + loss_v_b

        # covariance loss
        N, D = z_a.shape
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = ((z_a.T @ z_a) / (N - 1)).square()  # DxD
        cov_z_b = ((z_b.T @ z_b) / (N - 1)).square()  # DxD
        loss_c_a = (cov_z_a.sum() - cov_z_a.diagonal().sum()) / D
        loss_c_b = (cov_z_b.sum() - cov_z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b

        weighted_inv = loss_inv * self.hparams.invariance_loss_weight
        weighted_var = loss_var * self.hparams.variance_loss_weight
        weighted_cov = loss_cov * self.hparams.covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov
        
        return {
            "loss": loss,
            "loss_invariance": weighted_inv,
            "loss_variance": weighted_var,
            "loss_covariance": weighted_cov,
        }
 ...
    
  ```
  * `train_dataloader` and `val_dataloader` are defined by `DataLoader` in the model class. It get the hyperparameters from `VICRegParams` in `main.py`
  ```python
  def prepare_data(self) -> None:
        self.dataset.get_train()
        self.dataset.get_validation()

    def train_dataloader(self):
        return DataLoader(
            self.dataset.get_train(),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset.get_validation(),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
        ) 

  ```
  * 
4.  Lots of modification in `utils.py`
  * Import `ChallengeDataset` in `utils.py` 
  * Created class named `DiabeticRetinopathyGradingDataset` with defined class `DatasetBase`.
  * `ChallengeDataset` is returned for train/validation by calling `configure_train` and `configure_vallidation` funtions. `get_train` and `get_validation` will return each training dataset and validation dataset in DataBase class named `DiabeticRetinopathyGradingDataset` 
  ```python  
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
 ```python  
 
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
___
### Results and Conclusion
You can find in `Report_of_VICReg_Application_to_the_DRAC_22.pdf` in the current repository.
