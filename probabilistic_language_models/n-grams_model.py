import nltk
from nltk.util import ngrams  # n gram modeli oluşturmak için
from nltk.tokenize import word_tokenize 
from collections import Counter

#örnek veri seti oluştur.
corpus=[
        "I love apple",
        "I love him",
        "I love NLP",
        "You love me",
        "He loves apple",
        "They love apple",
        "I love you and you love me"
        ]
""" 
Problem tanımı:
    Dil modeli yapmak istiyoruz.
    Amaç :Bir kelimeden sonra gelecek olan kelimeyi tahmin etmek.Metin türetmek veya oluşturmak.
    bunun için n gram dil modelini kullanabiliriz.
    I .... : Olasılıksal olarak I dan sonra 4 defa love gelmiş boşluğu love ile doldurmasını bekliyoruz.
    I love ... : Olasılıksal olarak love dan sonra 3 defa apple gelmiş ve boşluğa apple gelmesini bekliyoruz.
"""

#verileri token haline getir
tokens=[word_tokenize(sentence.lower()) for sentence in corpus]

#bigram
bigrams=[]
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))
bigrams_freq=Counter(bigrams)

#trigram
trigrams=[]
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list, 3)))
trigrams_freq=Counter(trigrams)

#model testing
#"I love" dan sonra "you" veya "apple" gelem olasılıklarını hesaplayalım.
bigram=("i","love")
prob_you=trigrams_freq[("i","love","you")]/bigrams_freq[bigram]
print(f" you kelimesinin olma olasılığı: {prob_you}")

prob_apple=trigrams_freq[("i","love","apple")]/bigrams_freq[bigram]
print(f" apple kelimesinin olma olasılığı: {prob_apple}")
