from utils import generate_pngs
import argparse
parser = argparse.ArgumentParser(description='Generate PNG directory for the numpy files.')
parser = prepare_parser(parser)

parser.add_argument('--fold', type=in, default=0,
                     help='Fold to create the pngs')

args = parser.parse_args()

generate_pngs(f'./data/MIP/fold_{args.fold}/train_data/')
generate_pngs(f'./data/MIP/fold_{args.fold}/val_data/')
