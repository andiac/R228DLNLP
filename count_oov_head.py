import data_utils
from train_definition_model import load_pretrained_embeddings

vocab, rev_vocab = data_utils.initialize_vocabulary("./data/definitions/definitions_100000.vocab")

pre_embs_dict, embedding_length = load_pretrained_embeddings("./embeddings/GoogleWord2Vec.clean.normed.pkl")

count = 0
all_count = 0
fp = open("./data/definitions/train.definitions.ids100000.head")
for line in fp:
  if rev_vocab[int(line.strip())] in pre_embs_dict:
    count += 1
  all_count += 1
fp.close()
print("train:", count , "/", all_count)
count = 0
all_count = 0
fp = open("./data/definitions/dev.definitions.ids100000.head")
for line in fp:
  if rev_vocab[int(line.strip())] in pre_embs_dict:
    count += 1
  all_count += 1
fp.close()
print("dev:", count , "/", all_count)
