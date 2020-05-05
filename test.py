import argparse
import time

from utils import read_pickle, Datagen_set, Datagen_deepcom, Datagen_tree, Datagen_binary, bleu4, load_json
from models import Seq2seqModel, CodennModel, ChildsumModel, MultiwayModel, NaryModel
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
from joblib import delayed, Parallel
import json


# parse argments

parser = argparse.ArgumentParser(description='Source Code Generation')

parser.add_argument('-m', "--method", type=str, nargs="?", required=True,
                    choices=['seq2seq', 'deepcom', 'codenn', 'childsum', 'multiway', "nary"],
                    help='Encoder method')
parser.add_argument('-d', "--dim", type=int, nargs="?", required=False, default=512,
                    help='Representation dimension')
parser.add_argument("--embed", type=int, nargs="?", required=False, default=256,
                    help='Representation dimension')
parser.add_argument("--drop", type=float, nargs="?", required=False, default=.5,
                    help="Dropout rate")
parser.add_argument('-r', "--lr", type=float, nargs="?", required=True,
                    help='Learning rate')
parser.add_argument('-b', "--batch", type=int, nargs="?", required=True,
                    help='Mini batch size')
parser.add_argument('-e', "--epochs", type=int, nargs="?", required=True,
                    help='Epoch number')
parser.add_argument('-g', "--gpu", type=str, nargs="?", required=True,
                    help='What GPU to use')
parser.add_argument('-l', "--layer", type=int, nargs="?", required=False, default=1,
                    help='Number of layers')
parser.add_argument("--val", type=str, nargs="?", required=False, default="BLEU",
                    help='Validation method')
parser.add_argument('--data_dir', type=str, required=False, default="../dataset")

args = parser.parse_args()

name = args.method + "_dim" + str(args.dim) + "_embed" + str(args.embed)
name = name + "_drop" + str(args.drop)
name = name + "_lr" + str(args.lr) + "_batch" + str(args.batch)
name = name + "_epochs" + str(args.epochs) + "_layer" + str(args.layer) + "NEW_skip_size100"

checkpoint_dir = "./models/" + name


# set tf eager

tfe = tf.contrib.eager
config = tf.ConfigProto(
     gpu_options=tf.GPUOptions(
         visible_device_list=args.gpu))
# config = tf.ConfigProto(allow_soft_placement=False)

config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
session = tf.Session(config=config)
os.makedirs("./logs/" + name, exist_ok=True)
writer = tf.contrib.summary.create_file_writer("./logs/" + name, flush_millis=10000)


# load data

train_data = load_json(args.data_dir + '/train.json')
valid_data = load_json(args.data_dir + '/valid.json')
test_data = load_json(args.data_dir + '/test.json')

code_i2w = read_pickle(args.data_dir + '/code_i2w.pkl')
nl_i2w = read_pickle(args.data_dir + '/nl_i2w.pkl')
ast_i2w = read_pickle(args.data_dir + '/ast_i2w.pkl')


code_w2i = {v: k for k, v in code_i2w.items()}
nl_w2i = {v: k for k, v in nl_i2w.items()}
ast_w2i = {v: k for k, v in ast_i2w.items()}

train_ast_path = [x['ast_num'] for x in train_data]
train_code = [x['code'] for x in train_data]
train_nl = [x['nl'] for x in train_data]

valid_ast_path = [x['ast_num'] for x in valid_data]
valid_code = [x['code'] for x in valid_data]
valid_nl = [x['nl'] for x in valid_data]

test_ast_path = [x['ast_num'] for x in test_data]
test_code = [x['code'] for x in test_data]
test_nl = [x['nl'] for x in test_data]

train_y = [[nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in train_nl]
valid_y = [[nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in valid_nl]
test_y = [[nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in test_nl]

train_code_ids = [[code_w2i[t] if t in code_w2i.keys() else code_w2i["<UNK>"] for t in l] for l in train_code]
valid_code_ids = [[code_w2i[t] if t in code_w2i.keys() else code_w2i["<UNK>"] for t in l] for l in valid_code]
test_code_ids = [[code_w2i[t] if t in code_w2i.keys() else code_w2i["<UNK>"] for t in l] for l in test_code]

# setting model

if args.method in ['seq2seq', 'deepcom']:
    Model = Seq2seqModel
elif args.method in ['codenn']:
    Model = CodennModel
elif args.method in ['childsum']:
    Model = ChildsumModel
elif args.method in ['multiway']:
    Model = MultiwayModel
elif args.method in ['nary']:
    Model = NaryModel

print(len(ast_w2i))

model = Model(args.dim, args.dim, args.dim, len(ast_w2i), len(nl_w2i),
              dropout=args.drop, lr=args.lr, layer=args.layer)
epochs = args.epochs
batch_size = args.batch
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(model=model)
root.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Setting Data Generator

if args.method in ['deepcom', 'seq2seq']:
    tst_gen = Datagen_deepcom(test_ast_path, test_y, batch_size, ast_w2i, nl_i2w, path=args.data_dir + '/tree/test/', train=False)
elif args.method in ['codenn']:
    tst_gen = Datagen_set(test_ast_path, test_y, batch_size, ast_w2i, nl_i2w, path=args.data_dir + '/tree/test/', train=False)
elif args.method in ['childsum', 'multiway']:
    tst_gen = Datagen_tree(test_ast_path, test_y, batch_size, ast_w2i, nl_i2w, path=args.data_dir + '/tree/test/', train=False)
elif args.method in ['nary']:
    tst_gen = Datagen_binary(test_ast_path, test_y, batch_size, ast_w2i, nl_i2w, path=args.data_dir + '/tree/test/', train=False)

# evaluation
preds = []
trues = []
nodes_len = []

for x, y, x_raw, y_raw in tqdm(tst_gen(0), "Testing"):
    res = model.translate(x, nl_i2w, nl_w2i)
    preds += res
    trues += [s[1:-1] for s in y_raw]

    nodes_len += [len(s) for s in x_raw]

    print("node_len", nodes_len[0])
    print("predict", preds[0])
    print("true", trues[0])

results = []

for i in range(len(preds)):
    results.append({
        'node_len': str(nodes_len[i]),
        'predict': preds[i],
        'true': trues[i]
    })

with open('./' + args.method + '_predict_results.txt', 'w') as f:
    json.dump(results, f)
