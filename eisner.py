import numpy as np

'''
		Backtracking matrices and logic based from https://github.com/daandouwe/perceptron-dependency-parser
'''

class Eisner:

	'''
			Class object for the decoder using Eisner's algorithm.
	'''

	def decode(self, edge_scores):
		'''
				Calculates the best tree given an edge_scores matrix.
				Returns a dictionary of arc tuples (head, dependant).
		'''
		n=edge_scores.shape[0]		# Get shape of current sentence
		best_tree = -np.ones(n, dtype=int)		# Init tree with negative values
		
		# Initialize matrices with zeros
		# Left and right matrices combined with a third dimension -> direction
			# direction=1 -> right; direction=0 -> left
		O = np.zeros([n, n, 2])
		C = np.zeros([n, n, 2])

		# Initialize backtracking matrices with negative integers
		b_O = -np.ones([n, n, 2], dtype=int)
		b_C = -np.ones([n, n, 2], dtype=int)
		
		# Iterating through m and s from pseudocode in lecture
		for m in np.arange(1, n):
			for s in np.arange(0, n-m):
				t = s+m

				# max_vals from pseudocode matrices compiled into O and C 3-dim matrices
				# O_l
				O_l_list = C[s, s:t, 1] + C[s+1:t+1, t, 0] + edge_scores[t, s]
				O[s, t, 0] = np.max(O_l_list)
				b_O[s, t, 0] = s + np.argmax(O_l_list)

				# O_r
				O_r_list = C[s, s:t, 1] + C[s+1:t+1, t, 0] + edge_scores[s, t]
				O[s, t, 1] = np.max(O_r_list)
				b_O[s, t, 1] = s + np.argmax(O_r_list)
					
				# C_l
				C_l_list = C[s, s:t, 0] + O[s:t, t, 0]
				C[s, t, 0] = np.max(C_l_list)
				b_C[s, t, 0] = s + np.argmax(C_l_list)
								
				# C_r
				C_r_list = O[s, s+1:t+1, 1] + C[s+1:t+1, t, 1]
				C[s, t, 1] = np.max(C_r_list)
				b_C[s, t, 1] = s + 1 + np.argmax(C_r_list)

		# Initiate backtracking with closed-right matrix
		self.backtrack(b_O, b_C, 0, n-1, 1, 1, best_tree)

		# Init calculated arcs
		arcs = {}
		
		# Iterate through length of the tree
		for dep in np.arange(len(best_tree)):		# Dependants are unique
			if dep == 0:    # Skip root
				continue
			else:
				head = best_tree[dep]		# Add head to tree at dep_index
				arcs[str(dep)] = str(head)		# Add head to arcs[dep] for reference
		
		return arcs

	def backtrack(self, b_O, b_C, s, t, dir, structure, tree):
		'''
				Recursive backtracking function; calls return at points to stop recursion.

				Inputs: - backtracking matrices for open and closed structures
								- s and t for iteration through arcs
								- direction (L=0/R=1) and structure (O=0/C=1)
								- current tree
		'''
		if s == t:		# If at end of arcs
			return

		if structure == 1:		# Closed-structure backtracking
			backpointer = b_C[s, t, dir]		# Get current backpointer for (s, t, direction)
			if dir == 0:		# Left structure
				# Recursive calls with backpointer to backtrack through best arcs
				self.backtrack(b_O, b_C, s, backpointer, 0, 1, tree)
				self.backtrack(b_O, b_C, backpointer, t, 0, 0, tree)
				return
			else:		# Right structure
				self.backtrack(b_O, b_C, s, backpointer, 1, 0, tree)
				self.backtrack(b_O, b_C, backpointer, t, 1, 1, tree)
				return
		else:		# Repeat above procedure for open-structure backtracking; update trees here
			backpointer = b_O[s, t, dir]
			if dir == 0:
				tree[s] = t			# Update tree with arc (s, t)
				self.backtrack(b_O, b_C, s, backpointer, 1, 1, tree)
				self.backtrack(b_O, b_C, backpointer+1, t, 0, 1, tree)
				return
			else:
				tree[t] = s			# Update tree with arc (t, s)
				self.backtrack(b_O, b_C, s, backpointer, 1, 1, tree)
				self.backtrack(b_O, b_C, backpointer+1, t, 0, 1, tree)
				return
