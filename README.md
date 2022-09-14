# VICReg-DRAC22: VICReg Application to Diabetic Retinopathy Analysis Challenge 2022
VICReg(Variance-Invariance-Covariance Regularization for Self-Supervised Learning) Implementation of the Diabetic Retinopathy Image classification task(task3) hosted by Diabetic Retinopathy Analysis Challenge (DRAC22)
___

### Original Information Links

* VIGReg github link:
https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

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
 
2.  Modified `moco.py` to `model.py`
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
 * 
  ```python

       

    def _get_embeddings(self, x):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        bsz, nd, nc, nh, nw = x.shape
        assert nd == 2, "second dimension should be the split image -- dims should be N2CHW"
        im_q = x[:, 0].contiguous()
        im_k = x[:, 1].contiguous()

        # compute query features
        emb_q = self.model(im_q)
        q_projection = self.projection_model(emb_q)
        q = self.prediction_model(q_projection)  # queries: NxC
        if self.hparams.use_lagging_model:
            # compute key features
            with torch.no_grad():  # no gradient to keys
                if self.hparams.shuffle_batch_norm:
                    im_k, idx_unshuffle = utils.BatchShuffleDDP.shuffle(im_k)
                k = self.lagging_projection_model(self.lagging_model(im_k))  # keys: NxC
                if self.hparams.shuffle_batch_norm:
                    k = utils.BatchShuffleDDP.unshuffle(k, idx_unshuffle)
        else:
            emb_k = self.model(im_k)
            k_projection = self.projection_model(emb_k)
            k = self.prediction_model(k_projection)  # queries: NxC

        if self.hparams.use_unit_sphere_projection:
            q = torch.nn.functional.normalize(q, dim=1)
            k = torch.nn.functional.normalize(k, dim=1)

        return emb_q, q, k

 
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        all_params = list(self.model.parameters())
        x, class_labels = batch  # batch is a tuple, we just want the image

        emb_q, q, k = self._get_embeddings(x)
        pos_ip, neg_ip = self._get_pos_neg_ip(emb_q, k)

        logits, labels = self._get_contrastive_predictions(q, k)
        if self.hparams.use_vicreg_loss:
            losses = self._get_vicreg_loss(q, k, batch_idx)
            contrastive_loss = losses["loss"]
        else:
            losses = {}
            contrastive_loss = self._get_contrastive_loss(logits, labels)

            if self.hparams.use_both_augmentations_as_queries:
                x_flip = torch.flip(x, dims=[1])
                emb_q2, q2, k2 = self._get_embeddings(x_flip)
                logits2, labels2 = self._get_contrastive_predictions(q2, k2)

                pos_ip2, neg_ip2 = self._get_pos_neg_ip(emb_q2, k2)
                pos_ip = (pos_ip + pos_ip2) / 2
                neg_ip = (neg_ip + neg_ip2) / 2
                contrastive_loss += self._get_contrastive_loss(logits2, labels2)

        contrastive_loss = contrastive_loss.mean() * self.hparams.loss_constant_factor

        log_data = {
            "step_train_loss": contrastive_loss,
            "step_pos_cos": pos_ip,
            "step_neg_cos": neg_ip,
            **losses,
        }

        with torch.no_grad():
            self._momentum_update_key_encoder()

        some_negative_examples = (
            self.hparams.use_negative_examples_from_batch or self.hparams.use_negative_examples_from_queue
        )
        if some_negative_examples:
            acc1, acc5 = utils.calculate_accuracy(logits, labels, topk=(1, 5))
            log_data.update({"step_train_acc1": acc1, "step_train_acc5": acc5})

        # dequeue and enqueue
        if self.hparams.use_negative_examples_from_queue:
            self._dequeue_and_enqueue(k)

        self.log_dict(log_data)
        return {"loss": contrastive_loss}

    def validation_step(self, batch, batch_idx):
        x, class_labels = batch
        with torch.no_grad():
            emb = self.model(x)

        return {"emb": emb, "labels": class_labels}

    def validation_epoch_end(self, outputs):
        ...

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

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            encoding_optimizer,
            self.hparams.max_epochs,
            eta_min=self.hparams.final_lr_schedule_value,
        )
        return [encoding_optimizer], [self.lr_scheduler]

    ...
    


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

    @classmethod
    def params(cls, **kwargs) -> ModelParams:
        return ModelParams(**kwargs)


  ```
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
4.  Parameter changes in `main.py`
  * Add new EarlyStopping parameter `patience` to trainer 
  * `VICRegParams`
