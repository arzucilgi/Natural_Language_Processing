#kütüphane içeri aktar
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#örnek belge oluştur
documents=["köpek çok tatlı bir hayvandır.",
          "köpek ve kuş çok tatlı hayvanlardır.",
          "inekler süt üretirler."]

#vektorizer tanımla
tfidf_vectorizer=TfidfVectorizer()

#metinleri sayısal hale çevir
x=tfidf_vectorizer.fit_transform(documents)

#kelime kümesini incele
feature_names=tfidf_vectorizer.get_feature_names_out()

#vektör temsilini incele
vektor_temsili=x.toarray()
print(f"tf-idf: {vektor_temsili}")

df_tfidf=pd.DataFrame(vektor_temsili,columns=feature_names)

#ortalama Tf-IDF değerlerine bakalım 
tf_idf=df_tfidf.mean(axis=0)

