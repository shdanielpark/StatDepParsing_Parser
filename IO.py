from itertools import permutations

'''
	Script with functions and classes for reading and writing treebank files
'''

class Token:
	
	def __init__(self):		# Read all elements of tokens in data
		self.id = None
		self.form = None
		self.lemma = None
		self.pos = None
		self.xpos = None
		self.morph = None
		self.head = None
		self.deprel = None
		self.x = None
		self.y = None

class Sentence:		# Add all tokens as a list
	
	def __init__(self, token_items):        
		# create ROOT token
		root = Token()
		root.id = '0'
		root.form = 'ROOT'
		root.lemma = '_'
		root.pos = 'ROOT'
		root.xpos = '_'
		root.morph = '_'
		root.head = '_'
		root.deprel = '_'
		root.x = '_'
		root.y = '_'
		
		# initialize tokens with ROOT
		self.tokens = [root]
		
		# add each token in token_items to sentence
		for token in token_items:
			self.tokens.append(token)
				
	def potential_arcs(self):
		'''
			Helper function to get all potential arcs of the sentence.
		'''
		# All possible permutations of tokens except for ROOT
		arc_list = [(arc[0].id, arc[1].id) for arc in list(permutations(self.tokens[1:], 2))]
		for token in self.tokens[1:]:
			arc_list.append((self.tokens[0].id, token.id))		# Add ROOT arcs separately; only right-arc possible
		return set(arc_list)		# May be unnecessary, but no duplicate arcs
		
	def gold_arcs(self):
		'''
			Helper function to get the gold arcs of the sentence directly from Token items.
		'''
		arc_list = {}		# Dictionary of arcs; dependants are unique and therefore used as keys
		for token in self.tokens[1:]:
			head = token.head; dep = token.id
			arc_list[dep] = head
		return arc_list

class Data:		# Helpful class to gather all sentences of a file

	def __init__(self, sentences):
		self.sentences = sentences

	def evaluate(self):		# UAS function; named evaluate for possible expansion with LAS
		correct, total, sentence_count = 0, 0, 0		# Init counts to 0
		for sentence in self.sentences:
			for token in sentence.tokens:
				total += 1
				if token.id == '0':		# Skip root as a dependant
					continue
				else:
					predicted_head = token.x		# Store predicted head in one of the empty columns
					gold_head = token.head		# Get gold head of current token
					if predicted_head == gold_head:		# If correct, update count
						correct += 1
			sentence_count += 1
		uas = correct/total		# Simple UAS with print statement
		print("UAS score on", total, "tokens over", sentence_count, "sentences:", uas)
		return uas

class Reader:

	'''
		Use the Reader object to read files as lists of Sentence objects.
	'''
	
	def __init__(self, filepath):
		self.filepath = filepath	# Given the source filepath of the file
	
	def read_file(self):		# Read file with given filepath
		f = open(self.filepath)
		sentences = []		# Init list of sentences
	
		# Init token_items; information for each token
		token_items = []
	
		for line in f:
			# Init current sentence and token
			token = Token()
		
			# If not at end of sentence
			if line != '\n':
				items = line.split('\t')
			
				# Add token data
				token.id = items[0]
				token.form = items[1]
				token.lemma = items[2]
				token.pos = items[3]
				token.xpos = items[4]
				token.morph = items[5]
				token.head = items[6]
				token.deprel = items[7]
				token.x = items[8]
				token.y = items[9]
			
				# Add Token to Sentence
				token_items.append(token)
		
			# Add sentence and reset token_items
			else:
				sentences.append(Sentence(token_items))
				token_items = []
				
		f.close()		# Just to be safe
		return sentences

class Writer:

	'''
		Use the Writer object to write a list of sentences back into .CONLL06 format
	'''
	
	def __init__(self, filepath, sentences):		# filepath here is the target for writing the file
		self.filepath = filepath
		self.sentences = sentences
		
	def write_file(self):
		target_filename = self.filepath.split('/')[-1]		# Remove potential tag ie. [.blind, .gold]
		with open(target_filename+'.pred', 'w') as f:		# Add .pred tag
			for sentence in self.sentences:
				for token in sentence.tokens:
					if token.form == 'ROOT':		# Don't write the ROOT token
						continue
					else:		# Include every other token with all information
						line = ""
						line+=token.id+'\t'
						line+=token.form+'\t'
						line+=token.lemma+'\t'
						line+=token.pos+'\t'
						line+=token.xpos+'\t'
						line+=token.morph+'\t'
						line+=token.head+'\t'
						line+=token.deprel+'\t'
						line+=token.x+'\t'
						line+=token.y
					
						f.write(str(line))		# Write token line
				f.write('\n')		# Newline
			f.close()		# Insurance
