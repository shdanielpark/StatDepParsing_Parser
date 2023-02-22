import numpy as np
import timeit
import random

from reader import Reader, Sentence, Token
from feature import FeatureMapping
from eisner import Eisner

class Model:

	def __init__(self, sentences, feature_mapping):
		self.sentences = sentences
		self.feature_mapping = feature_mapping
		self.weight_vector = np.zeros(len(self.feature_mapping.map), dtype=np.float32)

	def edge_scores(self, sentence):
			arcs = sentence.potential_arcs()
			scores = np.zeros([len(sentence.tokens), len(sentence.tokens)], dtype=np.float32)
			for arc in arcs:
					feature_vectors = self.feature_mapping.extract_features(sentence, arc)
					head, dep = int(arc[0]), int(arc[1])
					scores[head][dep] += sum(self.weight_vector[feature_vectors])
			return scores

#	def evaluate(self, predicted):
#			correct_heads, total_tokens = 0, 0
#			for sentence in self.sentences:
#					for token in sentence.tokens:


	def train(self, epochs):
			decoder = Eisner()
			for i in np.arange(epochs):
					random.shuffle(self.sentences)

					starttime = timeit.default_timer()
					print("Start time: " + str(starttime))
					print("Epoch: " + str(i+1))

					correct_arcs = 0; total_arcs = 0

					sentence_count = 1
					for sentence in self.sentences:
							arc_scores = self.edge_scores(sentence)
							predicted = decoder.parse(arc_scores); gold = sentence.gold_arcs()
							#print(predicted)
							#print("")
							#print(gold)
							# Compare predicted tree with gold tree looping over tokens
							for token in sentence.tokens[1:]:
									total_arcs += 1
									dep = token.id
									#print(dep)
									predicted_head = predicted[dep]
									#print("Predicted head: " + predicted_head)
									gold_head = gold[dep]
									#print("Gold head: " + gold_head)
									predicted_arc = (predicted_head, dep)
									gold_arc = (gold_head, dep)
									#print("")
									#print("Predicted: " + (str(predicted_arc)))
									#print("Gold: " + str(gold_arc))
									if predicted_arc[0] != gold_arc[0]:
											predicted_arc_vector_indices = self.feature_mapping.extract_features(sentence, predicted_arc)
											gold_arc_vector_indices = self.feature_mapping.extract_features(sentence, gold_arc)
											#print(predicted_arc_vector, gold_arc_vector)
											self.weight_vector[gold_arc_vector_indices] += 1
											self.weight_vector[predicted_arc_vector_indices] += 1
									else:
											correct_arcs += 1
							sentence_count += 1
							if sentence_count % 100 == 0:
									#print("Sentence:", sentence_count)
									print("Time taken for past", sentence_count, "sentences:", (timeit.default_timer() - starttime))
											
					UAS = correct_arcs / total_arcs
					print("UAS: " + str(UAS))
					print("Time taken for epoch:", (timeit.default_timer() - starttime))

	def make_predictions(self):
			decoder = Eisner()
			for sentence in self.sentences:
					arc_scores = self.edge_scores(sentence)
					predicted = decoder.parse(arc_scores)
					for arc in predicted:
							dep = arc[1]; predicted_head = arc[0]
							sentence.tokens[dep].head = predicted_head