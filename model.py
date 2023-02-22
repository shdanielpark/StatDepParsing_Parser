import numpy as np
import timeit
import random

from IO import Reader, Writer, Data, Sentence, Token
from feature import FeatureMapping
from eisner import Eisner

class Model:

	def __init__(self, data, feature_mapping):
		self.data = data
		self.feature_mapping = feature_mapping
		self.weight_vector = np.zeros(len(self.feature_mapping.map), dtype=np.float32)

	def edge_scores(self, sentence):
			n = len(sentence.tokens)
			arcs = sentence.potential_arcs()
			scores = np.zeros([n, n], dtype=np.float32)
			for arc in arcs:
					feature_vectors = self.feature_mapping.extract_features(sentence, arc)
					#print(feature_vectors)
					arc_score = 0.0
					#print(sum(self.weight_vector[feature_vectors]))
					for vector in feature_vectors:
							arc_score += self.weight_vector[vector]
					head, dep = int(arc[0]), int(arc[1])
					#print(arc_score)
					scores[head][dep] = arc_score
			return scores

	def train(self, epochs):
			decoder = Eisner()
			for i in np.arange(epochs):
					random.shuffle(self.data.sentences)

					starttime = timeit.default_timer()
					print("Start time: " + str(starttime))
					print("Epoch: " + str(i+1))
					sentence_count = 1
					for sentence in self.data.sentences:
							arc_scores = self.edge_scores(sentence)
							predicted = decoder.decode(arc_scores); gold = sentence.gold_arcs()
							#print(predicted)
							#print("")
							#print(gold)
							# Compare predicted tree with gold tree looping over tokens
							for token in sentence.tokens[1:]:
									dep = token.id
									predicted_head = predicted[dep]
									#print("Dep:", dep)
									#print("Predicted head: " + predicted_head)
									gold_head = gold[dep]
									#print("Gold head: " + gold_head)
									predicted_arc = (predicted_head, dep)
									gold_arc = (gold_head, dep)
									token.x = predicted_head
									#print("")
									#print("Predicted: " + (str(predicted_arc)))
									#print("Gold: " + str(gold_arc))

									if predicted_arc[0] != gold_arc[0]:
											predicted_arc_vector_indices = self.feature_mapping.extract_features(sentence, predicted_arc)
											gold_arc_vector_indices = self.feature_mapping.extract_features(sentence, gold_arc)
											#print(predicted_arc_vector, gold_arc_vector)
											self.weight_vector[gold_arc_vector_indices] += 1
											self.weight_vector[predicted_arc_vector_indices] -= 1
							sentence_count += 1
							if sentence_count % 100 == 0:
									#print("Sentence:", sentence_count)
									print("Time taken for past", sentence_count, "sentences:", (timeit.default_timer() - starttime))
											
					self.data.evaluate()
					print("Time taken for epoch:", (timeit.default_timer() - starttime))
					print(sum(self.weight_vector))
					print("")


	def make_predictions(self, data):
			decoder = Eisner()
			for sentence in data.sentences:
					arc_scores = self.edge_scores(sentence)
					predicted = decoder.decode(arc_scores)
					for token in sentence.tokens[1:]:
							dep = token.id
							predicted_head = predicted[dep]
							sentence.tokens[int(dep)].x = predicted_head