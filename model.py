import numpy as np
import timeit
import random

from IO import Reader, Writer, Data, Sentence, Token
from feature import FeatureMapping
from eisner import Eisner

class Model:

	'''
			Use the Model object to create the dependency parser.
			Uses every function and object imported above.
	'''

	def __init__(self, data, feature_mapping):		# Initialize with data (list of sentences) and a feature map
		self.data = data
		self.feature_mapping = feature_mapping
		# Create weight_vector of 0's on instantiation
		self.weight_vector = np.zeros(len(self.feature_mapping.map), dtype=np.float32)

	def edge_scores(self, sentence):
			'''
					Calculate edge_scores for a given sentence
			'''
			n = len(sentence.tokens)
			arcs = sentence.potential_arcs()		# Get all arcs
			scores = np.zeros([n, n], dtype=np.float32)		# Init scores to 0's
			for arc in arcs:
					# Get feature vector ids corresponding to current arc
					feature_vectors = self.feature_mapping.extract_features(sentence, arc)
					arc_score = 0		# Init arc's score to 0
					for vector in feature_vectors:
							arc_score += self.weight_vector[vector]		# Add corresponding weight to arc score
					head, dep = int(arc[0]), int(arc[1])
					scores[head][dep] = arc_score		# Store arc score in scores matrix
			return scores

	def train(self, epochs=5):
			'''
					Trains the parser for given epochs.
					Epochs defaulted to 5 when not given an argument.
					No output; trains and updates the parser over epochs.
			'''
			decoder = Eisner()		# Create Eisner decoder
			for i in np.arange(epochs):		# Iterate through epochs
					random.shuffle(self.data.sentences)		# Shuffle data during each epoch
					# Timer
					starttime = timeit.default_timer()
					print("Start time: " + str(starttime))
					print("Epoch: " + str(i+1))
					sentence_count = 1
					for sentence in self.data.sentences:
							arc_scores = self.edge_scores(sentence)		# Calculate edge_scores for current sentence
							predicted = decoder.decode(arc_scores)		# Get best tree according to arc_scores
							gold = sentence.gold_arcs()
							# Compare predicted tree with gold tree looping over tokens
							for token in sentence.tokens[1:]:
									dep = token.id			# Current token is dep
									predicted_head = predicted[dep]		# Get predicted head from arc dictionary
									gold_head = gold[dep]		# Get actual (gold) head
									predicted_arc = (predicted_head, dep)		# Create arcs for extracting features
									gold_arc = (gold_head, dep)
									token.x = predicted_head		# Store predicted head in Token

									# Weights update (ie. training) time
									if predicted_arc[0] != gold_arc[0]:		# If predicted arc/head is incorrect:
											# Get feature vector_ids for both predicted and gold arcs
											predicted_arc_vector_indices = self.feature_mapping.extract_features(sentence, predicted_arc)
											gold_arc_vector_indices = self.feature_mapping.extract_features(sentence, gold_arc)
											# Update weight vector by raising gold vector_id weights and lowering (incorrect) predicted vector_id weights
											self.weight_vector[gold_arc_vector_indices] += 1
											self.weight_vector[predicted_arc_vector_indices] -= 1
							sentence_count += 1
							if sentence_count % 100 == 0:
									print("Time taken for past", sentence_count, "sentences:", (timeit.default_timer() - starttime))
											
					self.data.evaluate()		# Calculate current UAS
					print("Time taken for epoch:", (timeit.default_timer() - starttime))
					print("")

	def make_predictions(self, data):
			'''
					Function that makes predictions with the current parser on given data.
					Used during validation and testing.
					Similar process to training, except no feature_map population nor updating weights.
					Populates one of the empty columns with the predicted_head.
			'''
			decoder = Eisner()
			for sentence in data.sentences:
					arc_scores = self.edge_scores(sentence)		# Calculate arc scores
					predicted = decoder.decode(arc_scores)		# Predict best tree
					for token in sentence.tokens[1:]:
							dep = token.id
							predicted_head = predicted[dep]		# Get predicted head from dictionary.
							sentence.tokens[int(dep)].x = predicted_head		# Populate Token.x with predicted head
