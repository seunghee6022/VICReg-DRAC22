import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl


import pandas as pd
import matplotlib.pyplot as plt 

# make the plots pretty
import matplotlib as mpl
from matplotlib import rc

from model import SelfSupervisedMethod
from model_params import VICRegParams

# =============================================================================

hparams = VICRegParams(
   
   invariance_loss_weight = 25.0,
   variance_loss_weight = 25.0,
   covariance_loss_weight= 1.0,
   use_vicreg_loss= True,
   dataset_name="challenge",
   num_data_workers=8,
   transform_apply_blur=False,
   mlp_hidden_dim=64,
   dim=64,
   batch_size=32,
   lr=0.015,
   optimizer_name= "adam",
   final_lr_schedule_value=0,
   weight_decay=1e-4,
)

logger = CSVLogger('logger_history', name="logger")

#epoch 200~300
model = SelfSupervisedMethod(hparams)
trainer = pl.Trainer(
    devices='auto',
    accelerator="auto",
    logger=logger,
    max_epochs=100,
    log_every_n_steps=1,
    callbacks=[EarlyStopping(monitor="step_train_loss", patience=5)]
    
    )



def run_():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run_()

    trainer.fit(model)

    run_df = pd.read_csv('logger_history/logger/version_114/metrics copy.csv', sep=',')

    run = run_df[run_df['loss'].notna()]
    # print(run)

    mpl.rcParams['mathtext.default']='regular'
    plt.rc('font', family='serif', size=16)
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12) 

    plt.plot(run['step'], run['loss'],  color='red')
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.title('VICReg')
    plt.xlim([0,50])
    plt.show()

    plt.plot(run['step'], run['loss_variance'],  color='green', label='variance loss')
    plt.plot(run['step'], run['loss_invariance'],  color='blue', label='invariance loss')
    plt.plot(run['step'], run['loss_covariance'],  color='cyan', label='covariance loss')
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.legend(fontsize=12, frameon=False)
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.title('VICReg')
    plt.xlim([0,50])
    plt.show()

    run = run_df[run_df['train_class_acc'].notna()]
    mpl.rcParams['mathtext.default']='regular'
    plt.rc('font', family='serif', size=16)
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12) 

    plt.plot(run['epoch'], run['train_class_acc'],  color='red')
    plt.plot(run['epoch'], run['valid_class_acc'],  color='blue')
    plt.ylabel('Accuracy(%)')
    plt.xlabel('epoch')
    plt.title('VICReg')
    plt.xlim([0,3])
    plt.show()

    run_df = pd.read_csv('logger_history/logger/version_79/metrics.csv', sep=',')
    run = run_df[run_df['loss'].notna()]
    # print(run)

    mpl.rcParams['mathtext.default']='regular'
    plt.rc('font', family='serif', size=16)
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12) 

    plt.plot(run['step'], run['loss'],  color='red')
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.title('VICReg')
    plt.xlim([0,500])
    plt.show()

    plt.plot(run['step'], run['loss_variance'],  color='green', label='variance loss')
    plt.plot(run['step'], run['loss_invariance'],  color='blue', label='invariance loss')
    plt.plot(run['step'], run['loss_covariance'],  color='cyan', label='covariance loss')
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.legend(fontsize=12, frameon=False)
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.title('VICReg')
    plt.xlim([0,500])
    plt.show()

    run = run_df[run_df['train_class_acc'].notna()]
    mpl.rcParams['mathtext.default']='regular'
    plt.rc('font', family='serif', size=16)
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12) 

    plt.plot(run['epoch'], run['train_class_acc'],  color='red')
    plt.plot(run['epoch'], run['valid_class_acc'],  color='blue')
    plt.ylabel('Accuracy(%)')
    plt.xlabel('epoch')
    plt.title('VICReg')
    plt.xlim([0,80])
    plt.show()
