import os
import time
import argparse
import tensorflow as tf
import numpy as np
from sampler import WarpSampler
from model import Model
#from tqdm import tqdm
from util import *
from numpy import linalg as LA

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def calc_dist(X):
    n, k = X.shape
    res = 0.0
    for i in range(1, n):
        xi = X[i, :]
        for j in range(1, n):
            xj = X[j, :]
            res += np.power(LA.norm(xi - xj), 2)
    return res

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--threshold_item', default=1.0, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--gpu', default=0, type=int)


args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = len(user_train) / args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])

print("There are {0} users {1} items".format(usernum, itemnum))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum, 
        batch_size=args.batch_size, maxlen=args.maxlen, 
        threshold_item=args.threshold_item, n_workers=3)
model = Model(usernum, itemnum, args)
sess.run(tf.initialize_all_variables())

T = 0.0
t_test = evaluate(model, dataset, args, sess)
t_valid = evaluate_valid(model, dataset, args, sess)
print("[0, 0.0, {0}, {1}, {2}, {3}],".format(t_valid[0], t_valid[1], t_test[0], t_test[1]))

t0 = time.time()

for epoch in range(1, args.num_epochs + 1):
    for step in range(int(num_batch)):
        u, seq, pos, neg = sampler.next_batch()
        auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True})

    if epoch % 10 == 0:
        t1 = time.time() - t0
        T += t1
        t_test = evaluate(model, dataset, args, sess)
        t_valid = evaluate_valid(model, dataset, args, sess)
        print("[{0}, {1}, {2}, {3}, {4}, {5}],".format(epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
        t0 = time.time()

f.close()
sampler.close()
print("Done")
