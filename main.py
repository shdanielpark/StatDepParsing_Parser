import argparse
from utils import train_model, test_model

''' argparse functions and logic learned from documentation '''
''' https://docs.python.org/3/library/argparse.html '''

parser = argparse.ArgumentParser(description='Train and make predictions with a dependency parser.')
parser.add_argument('--task', choices=['train', 'test'], help='Train or test the parser.')
parser.add_argument('--train_filepath', type=str, help='The filepath to whichever file you wish to train on.')
parser.add_argument('--test_filepath', type=str, help='The filepath to whichever file you wish to test on.')
parser.add_argument('--model_filepath', type=str, help='The filepath to and from your dependency parser.')
parser.add_argument('--num_epochs', help='How many epochs if training?')

args = parser.parse_args()

if args.task == 'train':
		train_model(int(args.num_epochs), args.train_filepath, args.model_filepath)
elif args.task == 'test':
		test_model(args.model_filepath, args.test_filepath)
