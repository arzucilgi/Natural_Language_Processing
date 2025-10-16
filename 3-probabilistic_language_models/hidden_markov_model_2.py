import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000 # nltk içinde bulunan veri seti

# Gerekli veri setini içeri aktar
nltk.download("conll2000")

train_data=conll2000.tagged_sents("train.txt")
test_data=conll2000.tagged_sents("test.txt")

print(f"train_data:{train_data[:1]}")

"""
train_data:[[('Confidence', 'NN'), ('in', 'IN'), ('the', 'DT'), ('pound', 'NN'), ('is', 'VBZ'), ('widely', 'RB'),
             ('expected', 'VBN'), ('to', 'TO'),('take', 'VB'), ('another', 'DT'), ('sharp', 'JJ'), ('dive', 'NN'), 
             ('if', 'IN'), ('trade', 'NN'), ('figures', 'NNS'), ('for', 'IN'), 
             ('September', 'NNP'), (',', ','), ('due', 'JJ'), ('for', 'IN'), ('release', 'NN'), ('tomorrow', 'NN'),
             (',', ','), ('fail', 'VB'), ('to', 'TO'), ('show', 'VB'), ('a', 'DT'), ('substantial', 'JJ'),
             ('improvement', 'NN'), ('from', 'IN'), ('July', 'NNP'), ('and', 'CC'), ('August', 'NNP'), ("'s", 'POS'),
             ('near-record', 'JJ'), ('deficits', 'NNS'), ('.', '.')]]
"""

#train hmm
trainer=hmm.HiddenMarkovModelTrainer()
hmm_tagger=trainer.train(train_data)

#yeni cümle ve test
test_sentence="I like going to school".split()
tags=hmm_tagger.tag(test_sentence)
print(f"test sentence : {tags}")

"""
test sentence : [('I', 'PRP'), ('like', 'IN'), ('going', 'VBG'), ('to', 'TO'), ('school', 'NN')]
"""