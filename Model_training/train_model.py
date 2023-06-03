import argparse
import os
from datetime import datetime

import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from Model_training.models.base_model import BaseModel
from NetSolp_Dataset import NetsolpDataset
from utils.constants import seq_encoding_enum, mode_enum
import pytorch_lightning as pl
import wandb

def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Protein Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data related arguments
    parser.add_argument('--solubility_data', type=str, required=True,
                        help='Path to the solubility file')
    parser.add_argument('--protein_embedds', type=str, required=True,
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
    parser.add_argument('--model', type=str, required=False, default="Regressor_Simple",
                        help='Model architecture')

    # Performance related arguments
    parser.add_argument('--num_workers', type=int, required=False, default=8,
                        help='CPU Cores')
    parser.add_argument('--half_precision', action='store_true', default=False, help='Train the model with torch.float16 instead of torch.float32')
    parser.add_argument('--cpu_only', action='store_true', default=False, help='If the GPU is to WEAK use cpu only')

    args = parser.parse_args()

    return args


# TODO: Implement main training Loop with 5 folc cross val and wandb logging automatic dataset detection and dataloader init
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
        device = torch.dtype('cpu')

    # create an experiment name
    experiment_name = f'{args.model}-{datetime.now().strftime("%d/%m/%Y|%H:%M")}-{os.environ.get("USERNAME")}'

    # initialize 5 fold cross validation
    for fold in range(5):
        train_data_set = NetsolpDataset(seq_encoding=seq_encoding, set_mode=mode_enum.train, val_partion=fold, dtype=dtype,
                                        path_to_seq_data=args.solubility_data, path_to_embedds=args.path_to_embedds)
        val_data_set = NetsolpDataset(seq_encoding=seq_encoding, set_mode=mode_enum.val, val_partion=fold, dtype=dtype,
                                      path_to_seq_data=args.solubility_data, path_to_embedds=args.path_to_embedds)
        # init the model and send it to the device
        model = globals()[args.model](args=args, train_set=None, val_set=None)
        model.to(device)
        # set up early stopping and storage of the best model
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
        best_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode="min", dirpath="Data/chpts",
                                          filename=f'{type(model).__name__}' + "-{epoch:02d}-{val_loss:.2f}", auto_insert_metric_name=True)
        # set up a logger
        wandb_logger = WandbLogger(name=f'{experiment_name}_{fold}', project='pp1_test')
        wandb_logger.watch(model)
        # add experiment name so that we can group runs in wandb
        wandb_logger.experiment.config['experiment'] = experiment_name

        trainer = pl.Trainer(
            precision=16 if args.half_precision else 32,
            max_epochs=args.epochs,
            accelerator='gpu' if device == torch.device('cuda') else None,
            devices=1 if device == torch.device('cuda') else None,
            callbacks=[early_stop_callback, best_checkpoint],
            num_sanity_val_steps=0,
            logger=wandb_logger,
            accumulate_grad_batches={0: 10,
                                     4: 5,
                                     8: 1} if args.acc_grad else 1,
            log_every_n_steps=3
        )
        # train the model
        trainer.fit(model)
        # load the best model and run one final validation
        best_model_path = trainer.checkpoint_callback.best_model_path
        result = trainer.validate(ckpt_path=best_model_path)

        wandb_logger.finalize('success')
        wandb.finish()

if __name__ == '__main__':
    args = init_parser()
    main(args)
