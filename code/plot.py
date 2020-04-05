import numpy as np
import matplotlib.pyplot as plt

datapath = "../data"

def learning_curve():
	score_list = []
	with open(datapath + '/NELL_mrr.txt') as f:
		lines = f.readlines()
		for line in lines:
			line = line.rstrip()
			score_list.append(line)

	score_list = np.array(score_list)

	# score_list_2 = []
	# with open(datapath + '/NELL_hits10_2.txt') as f:
	# 	lines = f.readlines()
	# 	for line in lines:
	# 		line = line.rstrip()
	# 		score_list_2.append(line)

	# score_list_2 = np.array(score_list_2)

	# score_list_3 = []
	# with open(datapath + '/NELL_hits10_3.txt') as f:
	# 	lines = f.readlines()
	# 	for line in lines:
	# 		line = line.rstrip()
	# 		score_list_3.append(line)

	# score_list_3 = np.array(score_list_3)

	fig, axes = plt.subplots()
	#plt.plot(score_list_3[:30],'-o', color='blue', linewidth=3, markerfacecolor = 'green', label="Ours_3")
	#plt.plot(score_list_2[:30],'-o', color='grey', linewidth=3, markerfacecolor = 'black', label="Ours_2")
	plt.plot(score_list[:],'-o', color='orange', linewidth=3, markerfacecolor = 'red', label="Ours_1")	
	# labels = ['0e3', '5e3', '10e3', '15e3', '20e3', '25e3'\
	# , '30e3', '35e3', '40e3']
	# axes.set_xticklabels(labels)
	plt.grid()
	fig.subplots_adjust(left=0.15)
	fig.subplots_adjust(bottom=0.15)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.xlabel('Training Iteration #',size=20)
	plt.ylabel('Hits@10',size=20)
	plt.legend(bbox_to_anchor=(0.55, 0.25), loc=2, prop={'size': 20})
	fig.suptitle('Learning Curves', fontsize=20)
	plt.show()
	#fig.savefig("lc_test.pdf", bbox_inches='tight')

	score_list.sort()
	ave_hits = 0.0
	for i in range(5):
		ave_hits += float(score_list[-i-1])
		#print score_list[-i]
	ave_hits = ave_hits / 5

	print ('ave_hits: ' + str(ave_hits))


learning_curve()

