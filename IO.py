from itertools import permutations

class Token:
	
	def __init__(self):
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

class Sentence:
	
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
			arc_list = list of (head_int, token(dependency)_int) tuples
		'''
		#arc_list = list(permutations(self.tokens[1:], 2))
		arc_list = [(arc[0].id, arc[1].id) for arc in list(permutations(self.tokens[1:], 2))]
		for token in self.tokens[1:]:
			arc_list.append((self.tokens[0].id, token.id))
		return set(arc_list)
		
	def gold_arcs(self):
		'''
			arc_list = dictionary (key=dependent, val=head)
		'''
		arc_list = {}
		for token in self.tokens[1:]:
			#if token.form == "ROOT":
				#continue
			#else:
			head = token.head; dep = token.id
			arc_list[dep] = head
		return arc_list

class Data:

	def __init__(self, sentences):
		self.sentences = sentences

	def evaluate(self):
		correct, total, sentence_count = 0, 0, 0
		for sentence in self.sentences:
			for token in sentence.tokens:
				if token.id == '0':
					continue
				else:
					predicted_head = token.head[1]
					gold_head = token.head[0]
					total += 1
					if predicted_head == gold_head:
						correct += 1
		uas = correct/total
		print("UAS score on", total, "tokens over", sentence_count, "sentences:", uas)
		return uas

class Reader:
	
	def __init__(self, filepath):
		self.filepath = filepath
	
	def read_file(self):
		f = open(self.filepath)
		sentences = []
	
		# init token_items
		token_items = []
	
		for line in f:
			# init current sentence and token
			token = Token()
		
			# if not at end of sentence
			if line != '\n':
				items = line.split('\t')
				#print(items)
			
				# add token data
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
			
				# add Token to Sentence
				token_items.append(token)
		
			# add sentence and reset token_items
			else:
				sentences.append(Sentence(token_items))
				token_items = []
				
		f.close()
		return sentences

class Writer:
	
	def __init__(self, filepath, sentences):
		self.filepath = filepath
		self.sentences = sentences
		
	def write_file(self):
		target_filename = self.filepath.split('/')[-1]
		with open(target_filename+'.pred', 'w') as f:
			for sentence in self.sentences:
				for token in sentence.tokens:
					if token.form == 'ROOT':
						continue
					else:
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
					
						f.write(str(line))
				f.write('\n')
			f.close()