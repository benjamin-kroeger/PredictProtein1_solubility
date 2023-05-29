import argparse


def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Protein Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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

    opt = parser.parse_args()

    return opt

def test():
    """
    lsadkjfösldfjasölkfjölks
    :return:
    """

test