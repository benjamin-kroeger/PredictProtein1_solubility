import argparse
import os
from datetime import datetime

import pandas as pd
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, GradientAccumulationScheduler
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold

from Model_training.models.base_model import BaseModel
from Model_training.models.lin_models import *
from Model_training.models.fine_tune import fine_tune_t5,fine_tune_lora
from NetSolp_Dataset import NetsolpDataset
from NESG_Dataset import NESGDataset
from utils.constants import seq_encoding_enum, mode_enum
import pytorch_lightning as pl
import wandb
import seaborn as sns
import matplotlib.pyplot as plt

def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Protein Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data related arguments
    parser.add_argument('--solubility_data', type=str, required=True,
                        help='Path to the solubility file')
    parser.add_argument('--protein_embedds', type=str, required=False,
                        help='Path to the protein embeddings')
    # Training related arguments
    parser.add_argument('--batch_size', type=int, required=False, default=8,
                        help='Batch size!')
    parser.add_argument('--lr', type=float, required=False, default=1e-3,
                        help='Learning Rate!')
    parser.add_argument('--drop', type=float, required=False, default=0.1,
                        help='Dropout of model')
    parser.add_argument('--reg', type=float, required=False, default=0,
                        help='Weight decay')
    parser.add_argument('--gamma', type=float, required=False, default=1.00,
                        help='Gamma value for Exponential LR-Decay')
    parser.add_argument('--acc_grad', action='store_true', required=False, default=False,
                        help='Set if gradient accumulation is desired')
    parser.add_argument('--epochs', type=int, required=False, default=100,
                        help='Set maxs number of epochs.')
    parser.add_argument('--model', type=str, required=True, default="Regressor_Simple",
                        help='Model architecture')

    # arguments required for testing
    parser.add_argument('--test_model', action='store_true', default=False,required=False,
                        help='Once the final model Architecture is decided this can be used to train it and test it')
    parser.add_argument('--test_solubility_data', type=str, required=False,
                        help='Path to the solubility file')
    parser.add_argument('--test_protein_embedds', type=str, required=False,
                        help='Path to the protein embeddings')


    # Performance related arguments
    parser.add_argument('--num_workers', type=int, required=False, default=8,
                        help='CPU Cores')
    parser.add_argument('--half_precision', action='store_true', default=False, help='Train the model with torch.float16 instead of torch.float32')
    parser.add_argument('--cpu_only', action='store_true', default=False, help='If the GPU is to WEAK use cpu only')

    args = parser.parse_args()

    return args



def main(args):
    # set the default dtype to float32
    dtype = torch.float32
    if args.half_precision:
        dtype = torch.float16

    # get the type of sequence encoding in oder to create the datasets
    seq_encoding = globals()[args.model].seq_encoding
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cpu_only:
        device = torch.device('cpu')

    # create an experiment name
    experiment_name = f'{args.model}-{datetime.now().strftime("%d/%m/%Y|%H:%M")}-{os.environ.get("USERNAME")}'

    # create a var that stores the result of the best checkpoint
    best_checkpoint_path = None
    best_val_loss = 5

    # initialize 5 fold cross validation
    for fold in range(5):
        train_data_set = NetsolpDataset(seq_encoding=seq_encoding, set_mode=mode_enum.train, val_partion=fold, dtype=dtype,
                                        path_to_seq_data=args.solubility_data, path_to_embedds=args.protein_embedds)
        val_data_set = NetsolpDataset(seq_encoding=seq_encoding, set_mode=mode_enum.val, val_partion=fold, dtype=dtype,
                                      path_to_seq_data=args.solubility_data, path_to_embedds=args.protein_embedds)

        # init the model and send it to the device
        model = globals()[args.model](args=args, train_set=train_data_set, val_set=val_data_set)
        model.to(device)

        callbacks = []
        # set up early stopping and storage of the best model
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
        callbacks.append(early_stop_callback)
        best_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode="min", dirpath="Data/chpts",
                                          filename=f'{type(model).__name__}' + "-{epoch:02d}-{val_loss:.2f}", auto_insert_metric_name=True)
        callbacks.append(best_checkpoint)
        if args.acc_grad:
            accumulator = GradientAccumulationScheduler(scheduling={0:15,1: 20, 4: 20, 8: 15})
            callbacks.append(accumulator)

        # set up a logger
        wandb_logger = WandbLogger(name=f'{experiment_name}_{fold}',entity='pp1-solubility', project='solubility-prediction')
        wandb_logger.watch(model)
        # add experiment name so that we can group runs in wandb
        wandb_logger.experiment.config['experiment'] = experiment_name

        trainer = pl.Trainer(
            precision='16-mixed' if args.half_precision else 32,
            max_epochs=args.epochs,
            accelerator='gpu' if device == torch.device('cuda') else 'cpu',
            devices=1,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            logger=wandb_logger,
            log_every_n_steps=3
        )
        # train the model
        trainer.fit(model)
        # load the best model and run one final validation
        best_model_path = trainer.checkpoint_callback.best_model_path
        # best_model_path = "Data/chpts/testing_chpts"
        result = trainer.validate(ckpt_path=best_model_path)
        # if the last model was the best model yet store the path to the checkoint
        if result[0]['val_loss'] < best_val_loss:
            best_checkpoint_path = best_model_path

        wandb_logger.finalize('success')
        wandb.finish()


def test_best_model():
    pass

    if args.test_model:

        # set up a logger
        wandb_logger = WandbLogger(name=f'{experiment_name}_TEST', entity='pp1-solubility', project='solubility-prediction')
        # add experiment name so that we can group runs in wandb
        wandb_logger.experiment.config['experiment'] = experiment_name

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        trainer = pl.Trainer(
            precision='16-mixed' if args.half_precision else 32,
            accelerator='gpu' if device == torch.device('cuda') else 'cpu',
            devices=1,
            logger=wandb_logger,
            log_every_n_steps=3
        )

        test_metrics = []

        for ids, _ in kf.split(pd.read_csv(args.test_solubility_data)):
            test_Dataset = NESGDataset(seq_encoding=seq_encoding,dtype=dtype,path_to_seq_data=args.test_solubility_data,path_to_embedds=args.test_protein_embedds)
            sampler = torch.utils.data.SubsetRandomSampler(ids)
            model = globals()[args.model](args=args, test_set=test_Dataset,sampler=sampler)

            test_results = trainer.test(model=model,ckpt_path=best_checkpoint_path)
            test_metrics.extend(list(test_results[0].items()))

        test_metrics = pd.DataFrame(data=test_metrics,columns=['metric','value'])
        test_metrics.to_csv(f'{args.model}_test_metrics.csv')
        sns.barplot(data=test_metrics,x='metric',y='value',errorbar='se')
        plt.show()



def seed_all(seed):
    if not seed:
        seed = 10
    # torch.use_deterministic_algorithms(True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed_all(42)
    args = init_parser()
    if args.test_model:
        test_best_model()
    else:
        main(args)
