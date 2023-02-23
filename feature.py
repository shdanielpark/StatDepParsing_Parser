import timeit
import numpy as np

class FeatureMapping:
	
	'''
		Class object for feature map dictionary.
		IMPORTANT: m features for each n token
		AKA n*m feature vectors
		.map -> dictionary
			keys = feature_name, values = feature_vector_id
				ex. feature_name = 'hform=likes'
					feature_vector_id = 1
	'''
	
	def __init__(self, sentences):
		'''
			Initialize with a list of sentences before training.
		'''
		self.sentences = sentences
		self.map = {}		# Feature map dictionary
		self.vector_id = 0	# Current vector_id; updated during population
		self.frozen = False       # False when populating; set to True after fully populated with training data
	
	def create_features(self, sentence, arc):
		'''
			Helper function to create full feature names given a specific arc in a sentence.

			Inputs: - Sentence object of Tokens
					- arc tuple (head, dependant)

			Feature templates used are specified in the feature_templates list
			Vector_ids corresponding to relevant features of the arc specified in the features list
		'''

		# Get int and str versions of head/dependant ids for ease of use
		head_id = int(arc[0]); dep_id = int(arc[1])
		head = sentence.tokens[head_id]; dep = sentence.tokens[dep_id]
		final = len(sentence.tokens) - 1		# To stop from going out of index bounds
		
		# Get arc's direction
		if head_id > dep_id:
			direction = "L"
		else:
			direction = "R"

		# Get arc's distance
		distance = str(abs(head_id - dep_id))

		# Create basic constituent feature template values
		hform = "_NULL_" if head.form == "_" else head.form
		hpos = "_NULL_" if head.pos == "_" else head.pos
		dform = "_NULL_" if dep.form == "_" else dep.form
		dpos = "_NULL_" if dep.pos == "_" else dep.pos
		bpos = ""

		# Right arc; get hpos_plus1 and dpos_minus1 easily
		if direction == "R":
			hpos_minus1 = "_NULL_" if head_id == 0 else sentence.tokens[head_id-1].pos
			hpos_plus1 = sentence.tokens[head_id+1].pos
			dpos_minus1 = sentence.tokens[dep_id-1].pos
			dpos_plus1 = "_NULL_" if dep_id == final else sentence.tokens[dep_id+1].pos
		# Left arc; get hpos_minus1 and dpos_plus1 easily
		else:
			hpos_minus1 = sentence.tokens[head_id-1].pos
			hpos_plus1 = "_NULL_" if head_id == final else sentence.tokens[head_id+1].pos
			dpos_minus1 = "_NULL_" if dep_id == 0 else sentence.tokens[dep_id-1].pos
			dpos_plus1 = sentence.tokens[dep_id+1].pos

		# If head and dep are not next to each other
		if int(distance) > 0:
			if direction == "R":
				for i in np.arange(1, int(distance)):
					bpos = sentence.tokens[head_id+i].pos
			else:
				for i in np.arange(1, int(distance)):
					bpos = sentence.tokens[dep_id+i].pos
		# bpos limited to the last detected 'between' token

		# Dictionary to refer to above values
		feature_dict = {"hform": hform, "hpos": hpos, "dform": dform, "dpos": dpos, 
						"hpos-1": hpos_minus1, "hpos+1": hpos_plus1, "dpos-1": dpos_minus1, 
						"dpos+1": dpos_plus1, "bpos": bpos}

		# Feature templates to be used
		feature_templates = ["hform", "hpos", "dform", "dpos", "hform, hpos", "dform, dpos",   # Unigram features
							 "hform, hpos, dform, dpos",                                       # Bigram features
							 "hpos, dform, dpos", "hform, dform, dpos", 
							 "hform, hpos, dform", "hform, hpos, dpos", 
							 "hform, dform", "hpos, dpos", 
							 "hpos, dpos, hpos+1, dpos-1", "hpos, dpos, hpos-1, dpos-1", 		# 'Other' features
							 "hpos, dpos, hpos-1, dpos+1", "hpos, bpos, dpos"]
		
		# Initialize feature list
		full_features = []
		
		# Creating full feature values
		for feature_template in feature_templates:
			full_feature = ""    # init as empty string
			full_feature_components = [feature_template, "="]    # populate with template and =
			sub_features = feature_template.split(", ")    # For features with multiple constituents
			last_feature = sub_features[-1]    # Check if at last feature for direction and distance
			# Iterate through each full_feature constituent
			# Use feature_dict[sub_feature] to refer to values
			for sub_feature in sub_features:
				feature_val = feature_dict[sub_feature]
				if sub_feature != last_feature:
					#if sub_feature != "bpos":
					full_feature_components.append(feature_val)
					full_feature_components.append("+")
							# Remnants of code from attempting to properly implement bpos
					#else:
						#for b_pos in feature_dict[sub_feature]:		# bpos
							#full_feature_components.append(feature_val)
							#full_feature_components.append("+")
				else:		# Last feature; add direction and distance to the end of the feature
					full_feature_components.append(feature_val)
					full_feature_components.append("+")
					full_feature_components.append(direction)
					full_feature_components.append("+")
					full_feature_components.append(distance)

			full_feature = full_feature.join(full_feature_components)    # str.join is faster than + concat
			full_features.append(full_feature)		# Add full_feature to resulting list
			
		return full_features
	
	def create_map(self):
		'''
			Creates feature map for all sentences and arcs of self.sentences
		'''
		sentence_count = 0	# Keeping track of progress
		starttime = timeit.default_timer()		# Timer
		print("Start time: " + str(starttime))
		for sentence in self.sentences:
			sentence_count += 1
			arcs = sentence.potential_arcs()		# Get all arcs of sentence
			for arc in arcs:
				# Iterate through full feature values for current sentence-arc
				for feature in self.create_features(sentence, arc):
					# If full_feature not in map and still populating (ie. self.frozen == False)
					if feature not in self.map.keys() and self.frozen == False:
						self.map[feature] = self.vector_id
						self.vector_id += 1
			if sentence_count % 1000 == 0:
				print(sentence_count, "sentences took", (timeit.default_timer() - starttime), "to map")
		
		
	def extract_features(self, sentence, arc):
		'''
			Feature extraction for a particular arc; used during training and validation/testing
		'''

		# Init vectors for current arc
		feature_vectors = []
		
		for feature in self.create_features(sentence, arc):
			if feature in self.map.keys():		# Check if feature exists ie. for unknown dev/test features
				feature_vectors.append(self.map[feature])
			else:
				continue		# Skip for unknown features

		return feature_vectors
