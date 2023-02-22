class FeatureMapping:
	
	'''
		IMPORTANT: m features for each n token
		AKA n*m feature vectors
		.map -> dictionary
			keys = feature_name, values = feature_vector_id
				ex. feature_name = 'hform=likes'
					feature_vector_id = 1
	'''
	
	def __init__(self, sentences):
		self.sentences = sentences
		self.map = {}
		self.vector_id = 0
		self.frozen = False       
	
	def create_features(self, sentence, arc):
		'''
			Simultaneously populate feature dictionary and extract feature for current arc in sentence
			Feature templates used are specified in the feature_templates list
			Each arc will have vector_ids corresponding to relevant features specified in the features list
		'''
		head_id = int(arc[0]); dep_id = int(arc[1])
		head = sentence.tokens[head_id]; dep = sentence.tokens[dep_id]
		final = len(sentence.tokens) - 1
		
		# Get direction of arc
		if head_id > dep_id:
			direction = "L"
		else:
			direction = "R"

		# Get distance of arc
		distance = str(abs(head_id - dep_id))

		# Create basic constituent feature template values
		hform = "_NULL_" if head.form == "_" else head.form
		hpos = "_NULL_" if head.pos == "_" else head.pos
		dform = "_NULL_" if dep.form == "_" else dep.form
		dpos = "_NULL_" if dep.pos == "_" else dep.pos
		bpos = []

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
		#if distance > 0:
		#	if direction == "R":
		#		for i in np.arange(1, distance):
		#			bpos.append(sentence.tokens[head_id+i].pos)
		#	else:
		#		for i in np.arange(1, distance):
		#			bpos.append(sentence.tokens[dep_id+i].pos)

		# Dictionary to refer to above values
		feature_dict = {"hform": hform, "hpos": hpos, "dform": dform, "dpos": dpos, 
						"hpos-1": hpos_minus1, "hpos+1": hpos_plus1, "dpos-1": dpos_minus1, 
						"dpos+1": dpos_plus1}#, "bpos": bpos}

		# Feature templates to be used
		feature_templates = ["hform", "hpos", "dform", "dpos", "hform, hpos", "dform, dpos",   # Unigram features
							 "hform, hpos, dform, dpos",                                       # Bigram features
							 "hpos, dform, dpos", "hform, dform, dpos", 
							 "hform, hpos, dform", "hform, hpos, dpos", 
							 "hform, dform", "hpos, dpos", 
							 "hpos, dpos, hpos+1, dpos-1", "hpos, dpos, hpos-1, dpos-1", 
							 "hpos, dpos, hpos-1, dpos+1"]#, "hpos, bpos, dpos"]
		
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
					#else:
					#	for b_pos in feature_dict[sub_feature]:		# bpos
					#		full_feature_components.append()
				else:
					full_feature_components.append(feature_val)
					full_feature_components.append("+")
					full_feature_components.append(direction)
					full_feature_components.append("+")
					full_feature_components.append(distance)

			full_feature = full_feature.join(full_feature_components)    # .join is faster than + concat
			full_features.append(full_feature)
			
		return full_features
	
	def create_map(self):
		'''
			Creates feature mapping for all sentences and arcs of self.sentences
		'''
		sentence_count = 0
		for sentence in self.sentences:
			sentence_count += 1
			arcs = sentence.potential_arcs()
			for arc in arcs:
				# Iterate through full feature values for current sentence-arc
				for feature in self.create_features(sentence, arc):
					if feature not in self.map.keys() and self.frozen == False:    # If full_feature not in map
						self.map[feature] = self.vector_id
						self.vector_id += 1
			if sentence_count % 1000 == 0:
				print(sentence_count, "sentences mapped")
		
		
	def extract_features(self, sentence, arc):
		# Init vectors for current arc
		feature_vectors = []
		
		for feature in self.create_features(sentence, arc):
			feature_vectors.append(self.map[feature])

		return feature_vectors