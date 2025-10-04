# count vectorizer içeriye aktar.

from sklearn.feature_extraction.text import CountVectorizer


#Kendi veri setini oluştur.

documents=["kedi evde ", "kedi bahçede"]



#Vectorizer tanımla.
vectorizer=CountVectorizer()



#Metni sayısal vektörlere çevir.

x=vectorizer.fit_transform(documents)

#Sonuçların incelenmesi ve vektör temsili

feature_names=vectorizer.get_feature_names_out()# kelime kümesi oluştu.
print(f"kelime kümesi:{feature_names}")
vector_temsili=x.toarray()
print(f"vector temsili:{vector_temsili}")