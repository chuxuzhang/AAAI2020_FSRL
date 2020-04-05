import argparse

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", default="../data/NELL", type=str)
	parser.add_argument("--random_seed", default=1, type=int)
	parser.add_argument("--random_embed", default=0, type=int)
	parser.add_argument("--few", default=3, type=int)
	parser.add_argument("--test", default=0, type=int)
	parser.add_argument("--embed_model", default='ComplEx', type=str)
	parser.add_argument("--batch_size", default=128, type=int)
	parser.add_argument("--embed_dim", default=100, type=int)
	parser.add_argument("--dropout", default=0.5, type=float)
	parser.add_argument("--fine_tune", default=0, type=int)
	parser.add_argument("--aggregate", default='max', type=str)
	parser.add_argument("--process_steps", default=2, type=int)
	parser.add_argument("--aggregator", default='max', type=str)
	parser.add_argument("--lr", default=0.0001, type=float)
	parser.add_argument("--weight_decay", default=0, type=float)
	parser.add_argument("--max_neighbor", default=30, type=int)
	parser.add_argument("--train_few", default=1, type=int)
	parser.add_argument("--no_meta", default='0', type=int)
	parser.add_argument("--margin", default=5.0, type=float)
	parser.add_argument("--eval_every", default=10000, type=int)
	parser.add_argument("--max_batches", default=200000, type=int)
	parser.add_argument("--prefix", default='intial', type=str)
	parser.add_argument("--set_aggregator", default="lstmae", type=str)
	parser.add_argument("--ae_weight", default=0.00001, type=float)
	parser.add_argument("--cuda", default=1, type=int)

	args = parser.parse_args()
	args.save_path = 'models/' + args.prefix

	#print (args.embed_dim)

	print("------arguments/parameters-------")
	for k, v in vars(args).items():
		print(k + ': ' + str(v))
	print("---------------------------------")

	return args

