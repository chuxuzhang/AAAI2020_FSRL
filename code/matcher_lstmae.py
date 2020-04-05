import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

class EmbedMatcher_LSTMAE(nn.Module):
	def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout=0.5, batch_size=64,
				 process_steps=4, finetune=False, aggregate='max'):
		super(EmbedMatcher_LSTMAE, self).__init__()
		self.embed_dim = embed_dim
		self.pad_idx = num_symbols
		self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
		self.aggregate = aggregate
		self.num_symbols = num_symbols
		self.layer_norm = LayerNormalization(2 * self.embed_dim)

		self.gnn_w = nn.Linear(2 * self.embed_dim, self.embed_dim)
		self.gnn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

		self.dropout = nn.Dropout(dropout)

		self.set_rnn_encoder = nn.LSTM(2 * self.embed_dim, 2 * self.embed_dim, 1, bidirectional = False)
		self.set_rnn_decoder = nn.LSTM(2 * self.embed_dim, 2 * self.embed_dim, 1, bidirectional = False)

		self.set_FC_encoder = nn.Linear(3 * 2 * self.embed_dim, 2 * self.embed_dim)
		self.set_FC_decoder = nn.Linear(2 * self.embed_dim, 3 * 2 * self.embed_dim)

		# self.neigh_rnn = nn.LSTM(self.embed_dim, 50, 1, bidirectional = True)

		self.neigh_att_W = nn.Linear(2 * self.embed_dim, self.embed_dim)
		self.neigh_att_u = nn.Linear(self.embed_dim, 1)

		self.set_att_W = nn.Linear(2 * self.embed_dim, self.embed_dim)
		self.set_att_u = nn.Linear(self.embed_dim, 1)

		self.bn = nn.BatchNorm1d(2 * self.embed_dim)
		self.softmax = nn.Softmax(dim=1)

		self.support_g_W = nn.Linear(4 * self.embed_dim, 2 * self.embed_dim)

		self.FC_query_g = nn.Linear(2 * self.embed_dim, 2 * self.embed_dim)
		self.FC_support_g_encoder = nn.Linear(2 * self.embed_dim, 2 * self.embed_dim)

		init.xavier_normal_(self.gnn_w.weight)
		init.xavier_normal_(self.neigh_att_W.weight)
		init.xavier_normal_(self.neigh_att_u.weight)
		init.xavier_normal_(self.set_att_W.weight)
		init.xavier_normal_(self.set_att_u.weight)
		init.xavier_normal_(self.support_g_W.weight)
		init.constant_(self.gnn_b, 0)

		init.xavier_normal_(self.FC_query_g.weight)
		init.xavier_normal_(self.FC_support_g_encoder.weight)

		if use_pretrain:
			self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
			if not finetune:
				self.symbol_emb.weight.requires_grad = False

		d_model = self.embed_dim * 2
		self.support_encoder = SupportEncoder(d_model, 2 * d_model, dropout)
		self.query_encoder = QueryEncoder(d_model, process_steps)

	def neighbor_encoder(self, connections, num_neighbors):
		num_neighbors = num_neighbors.unsqueeze(1)
		# print num_neighbors
		relations = connections[:, :, 0].squeeze(-1)
		# print relations[0]

		entities = connections[:, :, 1].squeeze(-1)
		rel_embeds = self.dropout(self.symbol_emb(relations))  # (batch, 200, embed_dim)
		ent_embeds = self.dropout(self.symbol_emb(entities))  # (batch, 200, embed_dim)

		concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)  # (batch, 200, 2*embed_dim)

		#out_0 = self.gnn_w(concat_embeds)
		# out = torch.sum(out_0, dim=1)  # (batch, embed_dim)
		# out = out / num_neighbors

		# attention aggregation
		# out = self.neigh_att_W(out_0).tanh()
		# att_w = self.neigh_att_u(out)
		# att_w = self.softmax(att_w).view(out_0.size()[0], 1, 30)
		# out = torch.bmm(att_w, out_0).view(out_0.size()[0], 100)

		out = self.neigh_att_W(concat_embeds).tanh()
		att_w = self.neigh_att_u(out)
		att_w = self.softmax(att_w).view(concat_embeds.size()[0], 1, 30)
		out = torch.bmm(att_w, ent_embeds).view(concat_embeds.size()[0], self.embed_dim)

		# print (out.size())

		# out = out.view(30, -1, self.embed_dim)
		# out, out_state = self.neigh_rnn(out)
		# out = torch.mean(out, 0).view(-1, self.embed_dim)
		# return out
		return out.tanh()


	def forward(self, query, support, query_meta=None, support_meta=None):
		query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
		support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

		query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
		query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)

		support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
		support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

		query_neighbor = torch.cat((query_left, query_right), dim=-1)  # tanh
		support_neighbor = torch.cat((support_left, support_right), dim=-1)  # tanh

		support = support_neighbor
		query = query_neighbor

		support_g = self.support_encoder(support) # 1 * 100
		query_g = self.support_encoder(query)

		# mean pooling for reference set
		# support_g = torch.mean(support_g, dim=0, keepdim=True)

		# lstm aggregation for reference set
		#print (support_g.size())

		# lstm autoencoder
		support_g_0 = support_g.view(3, 1, 2 * self.embed_dim)
		support_g_encoder, support_g_state = self.set_rnn_encoder(support_g_0)
		support_g_decoder = support_g_encoder[-1].view(1, -1, 2 * self.embed_dim)
		decoder_set=[]
		support_g_decoder_state = support_g_state
		for idx in range(3):
			support_g_decoder, support_g_decoder_state = self.set_rnn_decoder(support_g_decoder, support_g_decoder_state)
			decoder_set.append(support_g_decoder)
		decoder_set = torch.cat(decoder_set, dim=0)

		# FC autoencoder
		# support_g = support_g.view(-1, 3 * 2 * self.embed_dim)
		# support_g_encoder = self.set_FC_encoder(support_g)
		# support_g_decoder = self.set_FC_decoder(support_g_encoder)

		ae_loss = nn.MSELoss()(support_g_0, decoder_set.detach())
		#ae_loss = 0

		#support_g_encoder = torch.mean(support_g_encoder, 0).view(1, 2*self.embed_dim)
		#support_g_encoder = support_g_encoder[-1].view(1, 2 * self.embed_dim)

		support_g_encoder = support_g_encoder.view(3, 2 * self.embed_dim)
		
		support_g_encoder = support_g_0.view(3, 2 * self.embed_dim) + support_g_encoder
		
		#support_g_encoder = torch.mean(support_g_encoder, dim=0, keepdim=True)		
		
		support_g_att = self.set_att_W(support_g_encoder).tanh()
		att_w = self.set_att_u(support_g_att)
		att_w = self.softmax(att_w)
		support_g_encoder = torch.matmul(support_g_encoder.transpose(0, 1), att_w)
		support_g_encoder = support_g_encoder.transpose(0, 1)

		support_g_encoder = support_g_encoder.view(1, 2 * self.embed_dim)

		#print (support_g_encoder.size())
		#print (query_g.size())

		query_f = self.query_encoder(support_g_encoder, query_g) # 128 * 100

		#print (support_g_encoder.size())

		# cosine similarity
		#query_g = self.FC_query_g(query_g)
		#support_g_encoder = self.FC_support_g_encoder(support_g_encoder)

		matching_scores = torch.matmul(query_f, support_g_encoder.t()).squeeze()

		return matching_scores, ae_loss, support_g_encoder, query_g


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
		self.process = nn.LSTMCell(input_dim, 2 * input_dim)

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
		h_r = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()
		c = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()

		# h_r = Variable(torch.zeros(batch_size, 2*self.input_dim))
		# c = Variable(torch.zeros(batch_size, 2*self.input_dim))

		for step in range(self.process_step):
			h_r_, c = self.process(query, (h_r, c))
			h = query + h_r_[:, :self.input_dim]  # (batch_size, query_dim)
			attn = F.softmax(torch.matmul(h, support.t()), dim=1)
			r = torch.matmul(attn, support)  # (batch_size, support_dim)
			h_r = torch.cat((h, r), dim=1)

		# return h_r_[:, :self.input_dim]
		return h

		