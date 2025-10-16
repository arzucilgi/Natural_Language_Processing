from sklearn.feature_extraction.text import CountVectorizer


#örnek metin 
document=[
    "Bu çalışma NGram çalışmasıdır.",
    "Bu çalışma Doğal Dil işleme çalışmaısıdır."
    ]

#uni-gram,bi-gram, tri-gram 3 farklı n değerine sahip gram modeli 
vectorizer_unigram=CountVectorizer(ngram_range=(1,1))
vectorizer_bigram=CountVectorizer(ngram_range=(2,2))
vectorizer_trigram=CountVectorizer(ngram_range=(3,3))

#unigram
x_unigram=vectorizer_unigram.fit_transform(document)
unigram_features=vectorizer_unigram.get_feature_names_out()

#bigram
x_bigram=vectorizer_bigram.fit_transform(document)
bigram_features=vectorizer_bigram.get_feature_names_out()

#trigram
x_trigram=vectorizer_trigram.fit_transform(document)
trigram_features=vectorizer_trigram.get_feature_names_out()

#sonuçların incelenmesi
print(f"unigram_features: {unigram_features}")
print(f"bigram_features: {bigram_features}")
print(f"trigram_features: {trigram_features}")

unigram_array=x_unigram.toarray()
bigram_array=x_bigram.toarray()
trigram_array=x_trigram.toarray()
print(bigram_array)