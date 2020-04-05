import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np


class EmbedMatcher(nn.Module):
	def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout=0.5, batch_size=64, process_steps=4, finetune=False, aggregate='max'):
		super(EmbedMatcher, self).__init__()
		self.embed_dim = embed_dim
		self.pad_idx = num_symbols
		self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
		self.aggregate = aggregate
		self.num_symbols = num_symbols
		self.few = 4

		self.gnn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
		#self.gnn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

		self.dropout = nn.Dropout(dropout)

		self.set_rnn = nn.LSTM(2*self.embed_dim, self.embed_dim, 1, bidirectional = True)
		self.match_rnn = nn.LSTM(4*self.embed_dim, self.embed_dim, 1, bidirectional = True)
		self.match_MLP = nn.Linear(4*self.embed_dim, 2*self.embed_dim)
		#self.neigh_rnn = nn.LSTM(self.embed_dim, 50, 1, bidirectional = True)

		self.neigh_att_W = nn.Linear(2 * self.embed_dim, self.embed_dim)
		self.neigh_att_u = nn.Linear(self.embed_dim, 1)

		self.set_att_W = nn.Linear(2 * self.embed_dim, self.embed_dim)
		self.set_att_u = nn.Linear(self.embed_dim, 1)

		self.aggre_match_att_W = nn.Linear(4 * self.embed_dim, 2 * self.embed_dim)
		self.aggre_match_att_u = nn.Linear(2 * self.embed_dim, 1)

		self.bn = nn.BatchNorm1d(2 * self.embed_dim)
		self.softmax = nn.Softmax(dim = 1)

		init.xavier_normal_(self.gnn_w.weight)
		init.xavier_normal_(self.neigh_att_W.weight)
		init.xavier_normal_(self.neigh_att_u.weight)
		init.xavier_normal_(self.set_att_W.weight)
		init.xavier_normal_(self.set_att_u.weight)
		#init.constant_(self.gnn_b, 0)

		#print (embed[0])
		if use_pretrain:
			self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
			if not finetune:
				self.symbol_emb.weight.requires_grad = False

		d_model = self.embed_dim * 2
		self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
		self.query_encoder = QueryEncoder(d_model, process_steps)
		self.NTNEncoder = NTNEncoder(d_model)


	# def neighbor_encoder(self, connections, num_neighbors):
	# 	num_neighbors = num_neighbors.unsqueeze(1)
	# 	#print num_neighbors
	# 	relations = connections[:,:,0].squeeze(-1)
	# 	#print relations[0]

	# 	entities = connections[:,:,1].squeeze(-1)
	# 	rel_embeds = self.dropout(self.symbol_emb(relations)) # (batch, 200, embed_dim)
	# 	ent_embeds = self.dropout(self.symbol_emb(entities)) # (batch, 200, embed_dim)

	# 	concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1) # (batch, 200, 2*embed_dim)

	# 	#average pooling
	# 	out_0 = self.gnn_w(concat_embeds)
	# 	out = torch.sum(out_0, dim=1) # (batch, embed_dim)
	# 	out = out / num_neighbors

	# 	#attention encoder
	# 	# out = self.neigh_att_W(concat_embeds).tanh()
	# 	# att_w = self.neigh_att_u(out) 
	# 	# att_w = self.softmax(att_w).view(concat_embeds.size()[0], 1, 30)
	# 	# out = torch.bmm(att_w, ent_embeds).view(concat_embeds.size()[0], self.embed_dim)

	# 	# out = out.view(30, -1, self.embed_dim)
	# 	# out, out_state = self.neigh_rnn(out)
	# 	# out = torch.mean(out, 0).view(-1, self.embed_dim)
	# 	#return out


	# 	return out.tanh()	


	def aggre_match(self, support, query):
		support = support.view(3, 2*self.embed_dim)
		support_new = support.expand(query.size()[0], 3, 2*self.embed_dim)

		query_new = query.expand(3, query.size()[0], 2*self.embed_dim).transpose(0, 1)

		concat = torch.cat((support_new, query_new), dim = -1).transpose(0, 1)
		#print concat.size()

		#concat = self.aggre_match_att_W(concat).tanh()
		#att_w = self.aggre_match_att_u(concat)
		#att_w = self.softmax(att_w).view(-1, 3)
		#att_embed = torch.matmul(att_w, support).view(-1, 1, 2*self.embed_dim)
		#query = query.view(-1, 2*self.embed_dim, 1)
		concat, concat_state = self.match_rnn(concat)
		concat = torch.mean(concat, dim=0).view(-1, 2*self.embed_dim)

		#score = torch.bmm(att_embed, concat).squeeze()

		return concat


	def MLP_match(self, support, query):
		support = support.expand(query.size()[0], 2*self.embed_dim)
		concat = torch.cat((support, query), dim = -1)

		concat = self.match_MLP(concat).relu()

		return concat
		

	# def forward(self, query, support, query_meta=None, support_meta=None):
	# 	query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
	# 	support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

	# 	query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
	# 	query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)

	# 	support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
	# 	support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

	# 	query_neighbor = torch.cat((query_left, query_right), dim=-1) # tanh
	# 	support_neighbor = torch.cat((support_left, support_right), dim=-1) # tanh

	# 	support = support_neighbor
	# 	query = query_neighbor

	# 	support_g = self.support_encoder(support) # 1 * 100
	# 	query_g = self.support_encoder(query)

	# 	# mean pooling for reference set
	# 	#support_g = torch.mean(support_g, dim=0, keepdim=True)
	# 	support_g = torch.max(support_g, dim=0, keepdim=True)[0]
	# 	#support_g = torch.median(support_g, dim=0, keepdim=True)[0]
	# 	#print support_g.size()

	# 	# attention aggregation for reference set
	# 	# support_g_att = self.set_att_W(support_g).tanh()
	# 	# att_w = self.set_att_u(support_g_att).transpose(0, 1)
	# 	# att_w = self.softmax(att_w)
	# 	# support_g = torch.matmul(support_g.transpose(0, 1), att_w.transpose(0, 1))
	# 	# support_g = support_g.transpose(0, 1)

	# 	# attention lstm aggregation for reference set 
	# 	# support_g_0 = support_g.view(3, 1, 2*self.embed_dim)
	# 	# support_g, support_g_state = self.set_rnn(support_g_0)
	# 	# support_g = (support_g + support_g_0)

	# 	# support_g_att = self.set_att_W(support_g).tanh()
	# 	# att_w = self.set_att_u(support_g_att).transpose(0, 1)
	# 	# att_w = self.softmax(att_w).view(3, 1)
	# 	# support_g = support_g.view(3, 2*self.embed_dim)
	# 	# support_g = torch.matmul(support_g.transpose(0, 1), att_w)
	# 	# support_g = support_g.transpose(0, 1)

	# 	# NTN aggregation
	# 	#matching_scores = self.NTNEncoder(support_g, query_g).squeeze()
	# 	#support_g = self.NTNEncoder(support_g, query_g)

	# 	query_f = self.query_encoder(support_g, query_g) # 128 * 100

	# 	#cosine similarity
	# 	matching_scores = torch.matmul(query_f, support_g.t()).squeeze()

	# 	return matching_scores, query_g

	# 	# max 
	# 	# matching_scores_list = torch.zeros([self.few, query_g.size()[0]], dtype = torch.float64)
	# 	# for i in range(support_g.size()[0]):
	# 	# 	support_g_temp = support_g[i].view(1, -1)
	# 	# 	query_f = self.query_encoder(support_g_temp, query_g) # 128 * 100
		
	# 	# 	matching_scores = torch.matmul(query_f, support_g_temp.t()).squeeze()

	# 	# 	matching_scores_list[i] = matching_scores

	# 	# matching_scores = torch.max(matching_scores_list, dim=0, keepdim=True)[0].view(-1)

	# 	# return matching_scores


	def forward(self, query, support):
		query_left = self.symbol_emb(query[:,0])
		query_right = self.symbol_emb(query[:,1])

		support_left = self.symbol_emb(support[:,0])
		support_right = self.symbol_emb(support[:,1])

		query_neighbor = torch.cat((query_left, query_right), dim=-1) # tanh
		support_neighbor = torch.cat((support_left, support_right), dim=-1) # tanh

		support = support_neighbor
		query = query_neighbor

		support_g = support
		query_g = query

		# #support_g = self.support_encoder(support) # 1 * 100
		# #query_g = self.support_encoder(query)

		# mean pooling for reference set
		# #support_g = torch.mean(support, dim=0, keepdim=True)
		# support_g = torch.mean(support_g, dim=0, keepdim=True)
		# #support_g = torch.max(support_g, dim=0, keepdim=True)[0]
		# #support_g = torch.median(support_g, dim=0, keepdim=True)[0]

		# #query_f = self.query_encoder(support_g, query_g) # 128 * 100

		# #support_g = support
		# query_f = query
		# #cosine similarity
		# matching_scores = torch.matmul(query_f, support_g.t()).squeeze()

		# return matching_scores

		# max 

		matching_scores_list = torch.zeros([self.few, query_g.size()[0]], dtype = torch.float64)
		for i in range(support_g.size()[0]):
			support_g_temp = support_g[i].view(1, -1)
			query_f = self.query_encoder(support_g_temp, query_g) # 128 * 100
		
			matching_scores = torch.matmul(query_f, support_g_temp.t()).squeeze()

			matching_scores_list[i] = matching_scores

		matching_scores = torch.max(matching_scores_list, dim=0, keepdim=True)[0].view(-1)

		return matching_scores


class LayerNormalization(nn.Module):
	''' Layer normalization module '''

	def __init__(self, d_hid, eps=1e-3):
		super(LayerNormalization, self).__init__()

		self.eps = eps
		self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
		self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

	def forward(self, z):
		if z.size(1) == 1:
			return z

		mu = torch.mean(z, keepdim=True, dim=-1)
		sigma = torch.std(z, keepdim=True, dim=-1)
		ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
		ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

		return ln_out


class SupportEncoder(nn.Module):
	"""docstring for SupportEncoder"""
	def __init__(self, d_model, d_inner, dropout=0.1):
		super(SupportEncoder, self).__init__()
		self.proj1 = nn.Linear(d_model, d_inner)
		self.proj2 = nn.Linear(d_inner, d_model)
		self.layer_norm = LayerNormalization(d_model)

		init.xavier_normal_(self.proj1.weight)
		init.xavier_normal_(self.proj2.weight)

		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()

	def forward(self, x):
		residual = x
		output = self.relu(self.proj1(x))
		output = self.dropout(self.proj2(output))
		return self.layer_norm(output + residual)


class QueryEncoder(nn.Module):
	def __init__(self, input_dim, process_step=4):
		super(QueryEncoder, self).__init__()
		self.input_dim = input_dim
		self.process_step = process_step
		self.process = nn.LSTMCell(input_dim, 2*input_dim)

	def forward(self, support, query):
		'''
		support: (few, support_dim)
		query: (batch_size, query_dim)
		support_dim = query_dim

		return:
		(batch_size, query_dim)
		'''
		assert support.size()[1] == query.size()[1]

		if self.process_step == 0:
			return query

		batch_size = query.size()[0]
		h_r = Variable(torch.zeros(batch_size, 2*self.input_dim)).cuda()
		c = Variable(torch.zeros(batch_size, 2*self.input_dim)).cuda()

		# h_r = Variable(torch.zeros(batch_size, 2*self.input_dim))
		# c = Variable(torch.zeros(batch_size, 2*self.input_dim))

		#print query.size()

		for step in range(self.process_step):
			h_r_, c = self.process(query, (h_r, c))
			h = query + h_r_[:,:self.input_dim] # (batch_size, query_dim)
			attn = F.softmax(torch.matmul(h, support.t()), dim=1)
			r = torch.matmul(attn, support) # (batch_size, support_dim)
			h_r = torch.cat((h, r), dim=1)

		# return h_r_[:, :self.input_dim]

		return h


class NTNEncoder(nn.Module):
	def __init__(self, d_model):
		super(NTNEncoder, self).__init__()
		self.ntn = nn.Parameter(torch.FloatTensor(5, d_model, d_model))
		self.ntn_u = nn.Parameter(torch.FloatTensor(5, 1))
		self.dimen = d_model
		#self.batch = batch_s
		init.xavier_normal_(self.ntn)
		#init.normal_(self.ntn)
		init.xavier_normal_(self.ntn_u)
		self.dropout = nn.Dropout(0.5)

	def forward(self, support_g, query_g):
		support_g = torch.matmul(self.ntn, support_g.transpose(0, 1))

		support_g = support_g.transpose(1, 2).view(5, 1, 3, self.dimen)
		query_g = query_g.view(query_g.size()[0], self.dimen, 1)
		score = (torch.matmul(support_g, query_g)/100.0).tanh().transpose(0, 1)
		score = score.view(score.size()[0], 5, 3)
		score, indices = torch.max(score, 2)
		score = torch.matmul(score, self.ntn_u)
		return score

		# support_g = support_g.transpose(0, 2).view(3, self.dimen, 5)
		# support_g = torch.matmul(support_g, self.ntn_u.view(5, 1)).tanh().view(3, self.dimen)
		# support_g, indices = torch.max(support_g, 0)
		# support_g = support_g.view(1, self.dimen)

		# return support_g







