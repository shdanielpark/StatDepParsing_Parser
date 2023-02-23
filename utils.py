import pickle, gzip

from IO import Reader, Writer, Data, Sentence, Token
from feature import FeatureMapping
from eisner import Eisner
from model import Model

'''
		Utils script for running train/test functions.
		Arguments configured via argparse.
'''

def train_model(num_epochs, train_filepath, model_filepath):

		'''
				Training process:
					1. Read in training_data with a Reader.
					2. Create corresponding feature_map dictionary.
							a. Once done populating, set feature_map.frozen to True.
					3. Create the dependency parser Model with training_data and feature_map.
					4. Call train() and run the training process.
					5. Once training is over, save the model via cPickle and gzip.
		'''
		reader = Reader(train_filepath)
		training_data = Data(reader.read_file())

		feature_map = FeatureMapping(training_data.sentences)
		feature_map.create_map()
		feature_map.frozen=True

		dep_parser = Model(training_data, feature_map)
		dep_parser.train(num_epochs)

		stream = gzip.open(model_filepath, 'wb')
		pickle.dump(dep_parser,stream,-1)
		stream.close()

def test_model(model_filepath, test_filepath):

		'''
				Testing process:
					1. Read in testing data with a Reader.
					2. Load in trained dependency parser model via cPickle and gzip.
					3. Call Model.make_predictions on the testing data's sentences.
					4. Populate the Token.heads with predictions.
					5. Write out the list of sentences with a Writer.
		'''
		reader = Reader(test_filepath)
		testing_data = Data(reader.read_file())

		stream = gzip.open(model_filepath, 'rb')
		dep_parser = pickle.load(stream)
		stream.close()
		dep_parser.make_predictions(testing_data)
		for sentence in testing_data.sentences:
				for token in sentence.tokens:
						token.head = token.x
						token.x = "_"
		writer = Writer(test_filepath, testing_data.sentences)
		writer.write_file()
