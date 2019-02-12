from MDP import MDP

class BellmanDPSolver(object):
	def __init__(self):
		self.MDP = MDP()
		self.disc = 1

	def initVs(self):

		self.VS = dict.fromkeys(self.MDP.S,0)

	def BellmanUpdate(self):
		policy =dict.fromkeys(self.MDP.S,[])
		for s in self.MDP.S:
			max_va = -100000000
			for a in self.MDP.A:
				p = self.MDP.probNextStates(s, a)
				vs_a =0
				for s1 in list(p.keys()):
					vs_a += p[s1]*(self.MDP.getRewards(s,a,s1) + self.disc*self.VS[s1])
				if (vs_a >= max_va):
					if(vs_a==max_va):
						policy[s].append(a)
					else:
						policy[s] = [a]
					max_va = vs_a
			self.VS[s] = max_va

		return self.VS,policy




		
if __name__ == '__main__':
	solution = BellmanDPSolver()
	solution.initVs()
	for i in range(1):
		values, policy = solution.BellmanUpdate()
	#print("Values : ", values)
	# for k,v in policy.items():
	# 	print(k,v)
	#print("Policy : ", policy)

	for k, v in policy.items():
		print(k,v)




