import numpy as np

class Eisner:

	def decode(self, edge_scores):
		n=edge_scores.shape[0]
		best_tree = -np.ones(n, dtype=int)
		
		# Initialize matrices with zeros
		O = np.zeros([n, n, 2])
		C = np.zeros([n, n, 2])

		# Initialize backtracking matrices with negative integers
		b_O = -np.ones([n, n, 2], dtype=int)
		b_C = -np.ones([n, n, 2], dtype=int)
		
		for m in np.arange(1, n):
			for s in np.arange(0, n-m):
				t = s+m

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

		self.backtrack(b_O, b_C, 0, n-1, 1, 1, best_tree)

		arcs = {}
		
		for dep in np.arange(len(best_tree)):
			if dep == 0:    # Skip root
				continue
			else:
				head = best_tree[dep]
				arcs[str(dep)] = str(head)
		
		return arcs

	def backtrack(self, b_O, b_C, s, t, dir, structure, tree):
		if s == t:
			return

		if structure == 1:
			backpointer = b_C[s, t, dir]
			if dir == 0:		# Left structure
				self.backtrack(b_O, b_C, s, backpointer, 0, 1, tree)
				self.backtrack(b_O, b_C, backpointer, t, 0, 0, tree)
				return
			else:
				self.backtrack(b_O, b_C, s, backpointer, 1, 0, tree)
				self.backtrack(b_O, b_C, backpointer, t, 1, 1, tree)
				return
		else:
			backpointer = b_O[s, t, dir]
			if dir == 0:
				tree[s] = t
				self.backtrack(b_O, b_C, s, backpointer, 1, 1, tree)
				self.backtrack(b_O, b_C, backpointer+1, t, 0, 1, tree)
				return
			else:
				tree[t] = s
				self.backtrack(b_O, b_C, s, backpointer, 1, 1, tree)
				self.backtrack(b_O, b_C, backpointer+1, t, 0, 1, tree)
				return