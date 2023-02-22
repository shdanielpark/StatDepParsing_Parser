import argparse
import cPickle, gzip

from reader import Reader, Data, Sentence, Token
from feature import FeatureMapping
from eisner import Eisner
from model import Model

def train_model(arguments):
		reader = Reader(train_filepath)
		training_data = Data(reader.read_file())

		feature_map = FeatureMapping(training_data.sentences)
		feature_map.create_map()
		feature_map.frozen=True

		dep_parser = Model(training_data, feature_map)
		dep_parser.train(arguments.num_epochs)

		stream = gzip.open(outfile, 'wb')
		cPickle.dump(dep_parser,stream,-1)
		stream.close()

def test_model(arguments):
		testing_data = Reader(test_filepath)
		testing_data.read_file()

		stream = gzip.open(infile, 'rb')
		dep_parser = cPickle.load(stream)
		stream.close()
		dep_parser.make_predictions(testing_data.sentences)


''' argparse code learned from documentation '''
''' https://docs.python.org/3/library/argparse.html '''

parser = argparse.ArgumentParser(description='Train and make predictions with a dependency parser.')
parser.add_argument('action', choices=['train', 'test'], help='Train or test the parser.')
parser.add_argument('language', choices=['en', 'de'], help='English (-en) or German (-de)?')
parser.add_argument('')