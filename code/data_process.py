import numpy as np
from collections import defaultdict
import json
#import matplotlib.pyplot as plt

# distribution of relation triples #
# def rel_triples_dis(datapath): 
# 	known_rels = defaultdict(list)
# 	with open(datapath + '/path_graph') as f:
# 		lines = f.readlines()
# 		for line in lines:
# 			line = line.rstrip()
# 			e1,rel,e2 = line.split()
# 			known_rels[rel].append([e1,rel,e2])

# 	train_tasks = json.load(open(datapath + '/train_tasks.json'))

# 	for key, triples in train_tasks.items():
# 		known_rels[key] = triples

# 	# path_graph (background knowledge) + train_tasks =  known relations
# 	json.dump(known_rels, open(datapath + '/known_rels.json', 'w'))

# 	#print len(known_rels)

# 	# compute distribution
# 	triple_n_dis = [0] * 5000000
# 	for key, value in known_rels.iteritems():
# 		triple_n_dis[len(known_rels[key])] += 1

# 	fig, axes = plt.subplots()
# 	plt.plot(triple_n_dis[:100],'-', color='black', linewidth=3, markerfacecolor = 'b')
# 	fig.subplots_adjust(left=0.15)
# 	fig.subplots_adjust(bottom=0.2)
# 	plt.xticks(fontsize=25)
# 	plt.yticks(fontsize=25)
# 	plt.xlabel('Relation Frequency',size=25)
# 	plt.ylabel('Number',size=25)
# 	plt.grid()
# 	plt.show()

# 	# for key, triples in known_rels:
# 	# 	print len(known_rels[key])


# distribution of relation triples #
def build_vocab(datapath):
	rels = set()
	ents = set()

	with open(datapath + '/path_graph') as f:
		lines = f.readlines()
		for line in lines:
			line = line.rstrip()
			rel = line.split('\t')[1]
			e1 = line.split('\t')[0]
			e2 = line.split('\t')[2]
			rels.add(rel)
			rels.add(rel + '_inv')
			ents.add(e1)
			ents.add(e2)
	
	# relation/entity id map		
	relationid = {}
	for idx, item in enumerate(list(rels)):
		relationid[item] = idx

	entid = {}
	for idx, item in enumerate(list(ents)):
		entid[item] = idx

	#print len(entid)

	json.dump(relationid, open(datapath + '/relation2ids', 'w'))
	json.dump(entid, open(datapath + '/ent2ids', 'w'))  


def candidate_triples(datapath):
	ent2ids = json.load(open(datapath+'/ent2ids'))

	all_entities = ent2ids.keys()

	type2ents = defaultdict(set)
	for ent in all_entities:
		try:
			type_ = ent.split(':')[1]
			type2ents[type_].add(ent)
		except Exception as e:
			continue

	train_tasks = json.load(open(datapath + '/known_rels.json'))
	dev_tasks = json.load(open(datapath + '/dev_tasks.json'))
	test_tasks = json.load(open(datapath + '/test_tasks.json'))

	all_reason_relations = train_tasks.keys() + dev_tasks.keys() + test_tasks.keys()

	all_reason_relation_triples = train_tasks.values() + dev_tasks.values() + test_tasks.values()
	
	assert len(all_reason_relations) == len(all_reason_relation_triples) 

	rel2candidates = {}
	for rel, triples in zip(all_reason_relations, all_reason_relation_triples):
		possible_types = set()
		for example in triples:
			try:
				type_ = example[2].split(':')[1] # type of tail entity
				possible_types.add(type_)
			except Exception as e:
				print (example)

		candidates = []
		for type_ in possible_types:
			candidates += list(type2ents[type_])

		candidates = list(set(candidates))
		if len(candidates) > 1000:
			candidates = candidates[:1000]
		rel2candidates[rel] = candidates

		#rel2candidates[rel] = list(set(candidates))
		
	json.dump(rel2candidates, open(datapath + '/rel2candidates_all.json', 'w'))


def for_filtering(datapath, save=False):
	e1rel_e2 = defaultdict(list)
	train_tasks = json.load(open(datapath + '/train_tasks.json'))
	dev_tasks = json.load(open(datapath + '/dev_tasks.json'))
	test_tasks = json.load(open(datapath + '/test_tasks.json'))
	few_triples = []
	for _ in (train_tasks.values() + dev_tasks.values() + test_tasks.values()):
		few_triples += _
	for triple in few_triples:
		e1,rel,e2 = triple
		e1rel_e2[e1+rel].append(e2)
	if save:
		json.dump(e1rel_e2, open(datapath + '/e1rel_e2.json', 'w'))



		


	





