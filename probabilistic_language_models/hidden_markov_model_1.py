"""
Part Of Speech POS: Kelimelerin uygun sözcük türünü bulmma çalışması
HMM
I(Zamir) am a teacher(isim)
"""

import nltk
from nltk.tag import hmm

#örnek training data tanımla 
train_data=[
    [("I","PRP"),("am","VBP"),("a","DT"),("teacher","NN")],
    [("You","PRP"),("are","VBP"),("a","DT"),("student","NN")]]

#train hmm
trainer=hmm.HiddenMarkovModelTrainer()
hmm_tagger=trainer.train(train_data)

#yeni bir cümle oluştur ve cümlenin içerisinde bulunna her bir sözcüğün türünü etiketle

test_sentence="I am a student".split()
tags=hmm_tagger.tag(test_sentence)
print(f"Yeni Cümle: {tags}")
"""
Yeni Cümle: [('I', 'PRP'), ('am', 'VBP'), ('a', 'DT'), ('student', 'NN')]
"""

test_sentence="He is a driver".split()
tags=hmm_tagger.tag(test_sentence)
print(f"Yeni Cümle: {tags}")

"""
Yeni Cümle: [('He', 'PRP'), ('is', 'PRP'), ('a', 'PRP'), ('driver', 'PRP')]
Çok az   eğitilmiş veri seti olduğu için yanlış sonuçlar üretti.
"""

