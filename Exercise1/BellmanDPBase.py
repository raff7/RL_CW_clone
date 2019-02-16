from MDP import MDP
import matplotlib.pyplot as plt
import numpy as np

class BellmanDPSolver(object):
	def __init__(self):
		self.MDP = MDP()
		self.disc = 0.9

	def initVs(self):

		self.VS = dict.fromkeys(self.MDP.S,0)
		policy =dict.fromkeys(self.MDP.S,self.MDP.A)

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


def plot_value_and_policy(values, policy):
	data = np.zeros((5, 5))

	plt.figure(figsize=(12, 4))
	plt.subplot(1, 2, 1)
	plt.title('Value')
	for y in range(data.shape[0]):
		for x in range(data.shape[1]):
			data[y][x] = values[(x, y)]
			plt.text(x + 0.5, y + 0.5, '%.4f' % data[y, x], horizontalalignment='center', verticalalignment='center', )

	heatmap = plt.pcolor(data)
	plt.gca().invert_yaxis()
	plt.colorbar(heatmap)

	plt.subplot(1, 2, 2)
	plt.title('Policy')
	for y in range(5):
		for x in range(5):
			for action in policy[(x, y)]:
				if action == 'DRIBBLE_UP':
					plt.annotate('', (x + 0.5, y), (x + 0.5, y + 0.5), arrowprops={'width': 0.1})
				if action == 'DRIBBLE_DOWN':
					plt.annotate('', (x + 0.5, y + 1), (x + 0.5, y + 0.5), arrowprops={'width': 0.1})
				if action == 'DRIBBLE_RIGHT':
					plt.annotate('', (x + 1, y + 0.5), (x + 0.5, y + 0.5), arrowprops={'width': 0.1})
				if action == 'DRIBBLE_LEFT':
					plt.annotate('', (x, y + 0.5), (x + 0.5, y + 0.5), arrowprops={'width': 0.1})
				if action == 'SHOOT':
					plt.text(x + 0.5, y + 0.5, action, horizontalalignment='center', verticalalignment='center', )

	heatmap = plt.pcolor(data)
	plt.gca().invert_yaxis()
	plt.colorbar(heatmap)
	plt.show()
		
if __name__ == '__main__':
	solution = BellmanDPSolver()
	solution.initVs()
	for i in range(20000):
		values, policy = solution.BellmanUpdate()
		if(i==100 or i==1000 or i==10000):
			plot_value_and_policy(values, policy)
	#print("Values : ", values)
	# for k,v in policy.items():
	# 	print(k,v)
	#print("Policy : ", policy)

	# for k, v in (policy.items()):
	# 	print(k,v,values[k])
	plot_value_and_policy(values,policy)




