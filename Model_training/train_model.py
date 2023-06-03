import argparse

import torch

from Model_training.models.base_model import BaseModel
from NetSolp_Dataset import NetsolpDataset
from utils.constants import seq_encoding_enum, mode_enum


def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Protein Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--solubility_data', type=str, required=True,
                        help='Path to the solubility file')
    parser.add_argument('--protein_embedds', type=str, required=True,
                        help='Path to the protein embeddings')

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

    parser.add_argument('--prot_embedd', type=str, required=False, default="prott5",
                        help='Set the embedding type for proteins: Options: prtott5, esm2')

    # TODO: add new args
    parser.add_argument('--model', type=str, required=False, default="Regressor_Simple",
                        help='Model architecture')

    parser.add_argument('--num_workers', type=int, required=False, default=8,
                        help='CPU Cores')

    parser.add_argument('--half_precision', action='store_true', default=False)

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

    for fold in range(5):
        train_data_set = NetsolpDataset(seq_encoding=seq_encoding, set_mode=mode_enum.train, val_partion=fold, dtype=dtype,
                                        path_to_seq_data=args.solubility_data, path_to_embedds=args.path_to_embedds)
        val_data_set = NetsolpDataset(seq_encoding=seq_encoding, set_mode=mode_enum.val, val_partion=fold, dtype=dtype,
                                      path_to_seq_data=args.solubility_data, path_to_embedds=args.path_to_embedds)

        model = globals()[args.model](args=args, train_set=None, val_set=None)


if __name__ == '__main__':
    args = init_parser()
    main(args)
