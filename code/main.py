import numpy as np
import random
import torch
from args import read_args
import data_process
import json
from data_generator import *
from matcher_0 import *
from matcher_lstmae import *
import torch.nn.functional as F
from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import os
torch.set_num_threads(1)
os.environ['CUDA_VISIBLE_DEVICES']='0'

class Model_Run(object):
	def __init__(self, arg):
		super(Model_Run, self).__init__()
		for k, v in vars(arg).items(): setattr(self, k, v)

		self.meta = not self.no_meta
		self.cuda = arg.cuda

		if self.random_embed:
			use_pretrain = False
		else:
			use_pretrain = True

		#print ("loading symbol id and pretrain embedding...")
		if self.test or self.random_embed:
			self.load_symbol2id()
			use_pretrain = False
		else:
			# load pretrained embedding
			self.load_embed()

		self.num_symbols = len(self.symbol2id.keys()) - 1 # one for 'PAD'
		self.pad_id = self.num_symbols
		self.use_pretrain = use_pretrain
		self.set_aggregator = args.set_aggregator
		self.embed_dim = args.embed_dim

		if self.set_aggregator == 'lstmae':
			self.matcher = EmbedMatcher_LSTMAE(self.embed_dim, self.num_symbols, use_pretrain=self.use_pretrain,
											   embed=self.symbol2vec, dropout=self.dropout, batch_size=self.batch_size,
											   process_steps=self.process_steps, finetune=self.fine_tune,
											   aggregate=self.aggregator)
		else:
			self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols, use_pretrain=self.use_pretrain,
										embed=self.symbol2vec, dropout=self.dropout, batch_size=self.batch_size,
										process_steps=self.process_steps, finetune=self.fine_tune,
										aggregate=self.aggregator)

		if self.cuda:
			self.matcher.cuda()

		self.batch_nums = 0
		self.parameters = filter(lambda p: p.requires_grad, self.matcher.parameters())
		self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)
		self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[10000], gamma=0.25)

		self.ent2id = json.load(open(self.datapath + '/ent2ids'))
		self.num_ents = len(self.ent2id.keys())

		degrees = self.build_graph(max_=self.max_neighbor)
		self.rel2candidates = json.load(open(self.datapath + '/rel2candidates_all.json')) 
		# load answer dict
		self.e1rel_e2 = defaultdict(list)
		self.e1rel_e2 = json.load(open(self.datapath + '/e1rel_e2.json'))


	def load_symbol2id(self):      
		symbol_id = {}
		rel2id = json.load(open(self.datapath + '/relation2ids'))
		ent2id = json.load(open(self.datapath + '/ent2ids'))
		i = 0
		for key in rel2id.keys():
			if key not in ['','OOV']:
				symbol_id[key] = i
				i += 1

		for key in ent2id.keys():
			if key not in ['', 'OOV']:
				symbol_id[key] = i
				i += 1

		symbol_id['PAD'] = i
		self.symbol2id = symbol_id
		self.symbol2vec = None


	def load_embed(self):
		symbol_id = {}
		rel2id = json.load(open(self.datapath + '/relation2ids'))
		ent2id = json.load(open(self.datapath + '/ent2ids'))

		print ("loading pre-trained embedding...")
		if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
			ent_embed = np.loadtxt(self.datapath + '/embed/entity2vec.' + self.embed_model)
			rel_embed = np.loadtxt(self.datapath + '/embed/relation2vec.' + self.embed_model)

			#print (ent_embed[0])

			if self.embed_model == 'ComplEx':
				# normalize the complex embeddings
				ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
				ent_std = np.std(ent_embed, axis=1, keepdims=True)
				rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
				rel_std = np.std(rel_embed, axis=1, keepdims=True)
				eps = 1e-3
				ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
				rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

			assert ent_embed.shape[0] == len(ent2id.keys())
			assert rel_embed.shape[0] == len(rel2id.keys())

			i = 0
			embeddings = []
			for key in rel2id.keys():
				if key not in ['','OOV']:
					symbol_id[key] = i
					i += 1
					embeddings.append(list(rel_embed[rel2id[key],:]))

			for key in ent2id.keys():
				if key not in ['', 'OOV']:
					symbol_id[key] = i
					i += 1
					embeddings.append(list(ent_embed[ent2id[key],:]))

			symbol_id['PAD'] = i
			embeddings.append(list(np.zeros((rel_embed.shape[1],))))
			embeddings = np.array(embeddings)
			assert embeddings.shape[0] == len(symbol_id.keys())

			self.symbol2id = symbol_id
			self.symbol2vec = embeddings


	def build_graph(self, max_=50):
		self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
		self.e1_rele2 = defaultdict(list)
		self.e1_degrees = defaultdict(int)

		with open(self.datapath + '/path_graph') as f:
			lines = f.readlines()
			for line in tqdm(lines):
				e1,rel,e2 = line.rstrip().split()
				self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
				self.e1_rele2[e2].append((self.symbol2id[rel+'_inv'], self.symbol2id[e1]))

		degrees = {}
		for ent, id_ in self.ent2id.items():
			neighbors = self.e1_rele2[ent]
			if len(neighbors) > max_:
				neighbors = neighbors[:max_]
			# degrees.append(len(neighbors)) 
			degrees[ent] = len(neighbors)
			self.e1_degrees[id_] = len(neighbors) # add one for self conn
			for idx, _ in enumerate(neighbors):
				self.connections[id_, idx, 0] = _[0]
				self.connections[id_, idx, 1] = _[1]

		# json.dump(degrees, open(self.dataset + '/degrees', 'w'))
		# assert 1==2

		return degrees


	def data_analysis(self):
		#data_process.rel_triples_dis(self.datapath)
		#data_process.build_vocab(self.datapath)
		data_process.candidate_triples(self.datapath)
		print("data analysis finish")


	def get_meta(self, left, right):
		if self.cuda:
			left_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left], axis=0))).cuda()
			left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
			right_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right], axis=0))).cuda()
			right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
		else:
			left_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left], axis=0)))
			left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left]))
			right_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right], axis=0)))
			right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right]))

		return (left_connections, left_degrees, right_connections, right_degrees)


	def save(self, path=None):
		if not path:
			path = self.save_path
		torch.save(self.matcher.state_dict(), path)


	def load(self):
		self.matcher.load_state_dict(torch.load(self.save_path))


	def train(self):
		print ('start training...')
		best_hits10 = 0.0
		hits10_file = open(self.datapath + "_hits10.txt", "w")
		hits5_file = open(self.datapath + "_hits5.txt", "w")
		hits1_file = open(self.datapath + "_hits1.txt", "w")
		mrr_file = open(self.datapath + "_mrr.txt", "w")
		for data in train_generate(self.datapath, self.batch_size, self.few, self.symbol2id, self.ent2id, self.e1rel_e2):
			support, query, false, support_left, support_right, query_left, query_right, false_left, false_right = data	

			support_meta = self.get_meta(support_left, support_right)
			query_meta = self.get_meta(query_left, query_right)
			false_meta = self.get_meta(false_left, false_right)

			if self.cuda:
				support = Variable(torch.LongTensor(support)).cuda()
				query = Variable(torch.LongTensor(query)).cuda()
				false = Variable(torch.LongTensor(false)).cuda()
			else:
				support = Variable(torch.LongTensor(support))
				query = Variable(torch.LongTensor(query))
				false = Variable(torch.LongTensor(false))

			if self.no_meta:
				if self.set_aggregator == 'lstmae':
					query_scores, ae_loss = self.matcher(query, support)
					false_scores, ae_loss = self.matcher(false, support)
				else:
					query_scores = self.matcher(query, support)
					false_scores = self.matcher(false, support)
			else:
				if self.set_aggregator == 'lstmae':
					query_scores, ae_loss, support_g_embed, query_g_embed = self.matcher(query, support, query_meta, support_meta)
					false_scores, ae_loss, support_g_embed, query_g_embed = self.matcher(false, support, false_meta, support_meta)
				else:
					query_scores, query_g_embed = self.matcher(query, support, query_meta, support_meta)
					false_scores, query_g_embed = self.matcher(false, support, false_meta, support_meta)

			margin_ = query_scores - false_scores
			loss = F.relu(self.margin - margin_).mean()

			# loss = - F.logsigmoid(query_scores - false_scores)
			# loss = loss.sum() / len(query_scores)

			#return loss

			if self.set_aggregator == 'lstmae':
				loss += args.ae_weight * ae_loss

			self.optim.zero_grad()
			loss.backward()

			self.optim.step()    

			if self.batch_nums % self.eval_every == 0:
				print ('batch num: '+str(self.batch_nums))
				print ('loss: '+str(loss))
				hits10, hits5, hits1, mrr = self.eval(meta=self.meta)
				#print (hits10)
				hits10_file.write(str(("%.3f" % hits10)) + "\n")
				hits5_file.write(str(("%.3f" % hits5)) + "\n")
				hits1_file.write(str(("%.3f" % hits1)) + "\n")
				mrr_file.write(str(("%.3f" % mrr)) + "\n")

				self.save()

				if hits10 > best_hits10:
					self.save(self.save_path + '_bestHits10')
					best_hits10 = hits10

			self.batch_nums += 1

			self.scheduler.step()
			if self.batch_nums == self.max_batches:
				self.save()
				break
				hits10_file.close()
				hits5_file.close()
				hits1_file.close()
				mrr_file.close()


	def eval(self, mode='test', meta=False):
		self.matcher.eval()

		symbol2id = self.symbol2id
		few = self.few

		#print (len(symbol2id))

		if mode == 'dev':
			test_tasks = json.load(open(self.datapath + '/dev_tasks.json'))
		else:
			test_tasks = json.load(open(self.datapath + '/test_tasks.json'))

		rel2candidates = self.rel2candidates

		hits10 = []
		hits5 = []
		hits1 = []
		mrr = []

		# for query_ in test_tasks.keys():
		# 	print len(test_tasks[query_])
		# 	if len(test_tasks[query_]) < 4:
		# 		print len(test_tasks[query_])
		# print ("evaluation")

		#print (len(test_tasks.keys()))
		task_embed_f = open(self.datapath + "task_embed.txt", "w")
		#print (len(test_tasks.keys()))
		temp_count = 0
		for query_ in test_tasks.keys():
			#print (query_)
			entity_embed_f = open(self.datapath + str(query_) + "_entity_embed.txt", "w")
			
			task_embed_f.write(str(query_) + ",")

			hits10_ = []
			hits5_ = []
			hits1_ = []
			mrr_ = []

			candidates = rel2candidates[query_]
			# if temp_count == 0:
			# 	print (candidates[500])
			#print (len(candidates))
			support_triples = test_tasks[query_][:few]

			# train_and_test = test_tasks[query_]
			# random.shuffle(train_and_test)
			# support_triples = train_and_test[:few]
			# if temp_count == 0:
			# 	print (support_triples[0][0])
			# 	print (support_triples[0][2])

			temp_count += 1

			support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]

			if meta:
				support_left = [self.ent2id[triple[0]] for triple in support_triples]
				support_right = [self.ent2id[triple[2]] for triple in support_triples]
				support_meta = self.get_meta(support_left, support_right)

			if self.cuda:
				support = Variable(torch.LongTensor(support_pairs)).cuda()
			else:
				support = Variable(torch.LongTensor(support_pairs))

			temp = 0
			#print (len(test_tasks[query_][few:]))
			for triple in test_tasks[query_][few:]:
				temp += 1
				true = triple[2]
				query_pairs = []
				if triple[0] in symbol2id and triple[2] in symbol2id:
					query_pairs.append([symbol2id[triple[0]], symbol2id[triple[2]]])

				if meta:
					query_left = []
					query_right = []
					if triple[0] in self.ent2id and triple[2] in self.ent2id:
						query_left.append(self.ent2id[triple[0]])
						query_right.append(self.ent2id[triple[2]])

				for ent in candidates:
					if (ent not in self.e1rel_e2[triple[0]+triple[1]]) and ent != true:
						query_pairs.append([symbol2id[triple[0]], symbol2id[ent]])
						if meta:
							query_left.append(self.ent2id[triple[0]])
							query_right.append(self.ent2id[ent])

				if self.cuda:
					query = Variable(torch.LongTensor(query_pairs)).cuda()
				else:
					query = Variable(torch.LongTensor(query_pairs))

				if meta:
					query_meta = self.get_meta(query_left, query_right)
					if self.set_aggregator == 'lstmae':
						scores, _, support_g_embed, query_g_embed = self.matcher(query, support, query_meta, support_meta)
						#print (support_g_embed.cpu().data.numpy()[:10])
						support_g_embed = support_g_embed.view(support_g_embed.numel())
						query_g_embed_temp = query_g_embed.cpu().detach().numpy()
						#query_g_embed_temp = query_g_embed.cpu().data.numpy()
						if temp == 1:
							for k in range(1, len(query_g_embed_temp)):
								#query_g_embed_temp = query_g_embed[k].view(query_g_embed[k].numel())
								#query_g_embed_temp = query_g_embed_temp.cpu().data.numpy()
								for l in range(len(query_g_embed_temp[k])):
									entity_embed_f.write(str(query_g_embed_temp[k][l]) + " ")
								entity_embed_f.write("\n")
						
						#query_g_embed_temp = query_g_embed[0].view(query_g_embed[0].numel())
						#query_g_embed_temp = query_g_embed_temp.cpu().data.numpy()
						for l in range(len(query_g_embed_temp[0])):
							entity_embed_f.write(str(query_g_embed_temp[0][l]) + " ")
						entity_embed_f.write("\n")

						if temp == 1:
							embed_temp = support_g_embed.cpu().data.numpy()
							#print (len(embed_temp))
							for l in range(len(embed_temp)):
								task_embed_f.write(str(float(embed_temp[l])) + " ")
							task_embed_f.write("\n")

					else:
						scores, query_g_embed = self.matcher(query, support, query_meta, support_meta)
						query_g_embed_temp = query_g_embed.cpu().detach().numpy()
						#query_g_embed_temp = query_g_embed.cpu().data.numpy()
						if temp == 1:
							for k in range(1, len(query_g_embed_temp)):
								#query_g_embed_temp = query_g_embed[k].view(query_g_embed[k].numel())
								#query_g_embed_temp = query_g_embed_temp.cpu().data.numpy()
								for l in range(len(query_g_embed_temp[k])):
									entity_embed_f.write(str(query_g_embed_temp[k][l]) + " ")
								entity_embed_f.write("\n")
						
						#query_g_embed_temp = query_g_embed[0].view(query_g_embed[0].numel())
						#query_g_embed_temp = query_g_embed_temp.cpu().data.numpy()
						for l in range(len(query_g_embed_temp[0])):
							entity_embed_f.write(str(query_g_embed_temp[0][l]) + " ")
						entity_embed_f.write("\n")

						# if temp == 1:
						# 	embed_temp = support_g_embed.cpu().data.numpy()
						# 	#print (len(embed_temp))
						# 	for l in range(len(embed_temp)):
						# 		task_embed_f.write(str(float(embed_temp[l])) + " ")
						# 	task_embed_f.write("\n")

					scores.detach()
					scores = scores.data
				else:
					if self.set_aggregator == 'lstmae':
						scores, _ = self.matcher(query, support)
					else:
						scores = self.matcher(query, support)
					scores.detach()
					scores = scores.data

				scores = scores.cpu().numpy()
				sort = list(np.argsort(scores))[::-1]
				rank = sort.index(0) + 1

				if rank <= 10:
					hits10.append(1.0)
					hits10_.append(1.0)
				else:
					hits10.append(0.0)
					hits10_.append(0.0)
				if rank <= 5:
					hits5.append(1.0)
					hits5_.append(1.0)
				else:
					hits5.append(0.0)
					hits5_.append(0.0)
				if rank <= 1:
					hits1.append(1.0)
					hits1_.append(1.0)
				else:
					hits1.append(0.0)
					hits1_.append(0.0)
				mrr.append(1.0/rank)
				mrr_.append(1.0/rank)

		print ('hits1: {:.3f}'.format(np.mean(hits1)))
		print ('hits5: {:.3f}'.format(np.mean(hits5)))
		print ('hits10: {:.3f}'.format(np.mean(hits10)))
		print ('mrr: {:.3f}'.format(np.mean(mrr)))

		task_embed_f.close()
		entity_embed_f.close()

		self.matcher.train()

		return np.mean(hits10), np.mean(hits5), np.mean(hits1), np.mean(mrr)


if __name__ == '__main__':
	args = read_args()

	# setup random seeds
	random.seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed_all(args.random_seed)

	# model execution 
	model_run = Model_Run(args)

	# data analysis
	#model_run.data_analysis()
	
	# train/test model
	if args.test:
		model_run.test()
	else:
		model_run.train()



