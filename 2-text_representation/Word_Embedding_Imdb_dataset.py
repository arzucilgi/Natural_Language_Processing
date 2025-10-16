import pandas as pd
import matplotlib.pyplot as plt 
import re
from sklearn.decomposition import  PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

# veri setini yükle
df=pd.read_csv("IMDB Dataset.csv")
documents =df["review"]
stop_word_eng=set(stopwords.words("english"))

#metin temizleme 
def clean_text(text):
    text=text.lower()
    text=re.sub(r"\d+", "" ,text)
    text=re.sub(r"[^\w\s]","", text)
    text=" ".join([word for word in text.split() if len(word)>2])
    words = [word for word in text.split() if len(word) > 2 and word not in stop_word_eng]
   # text=simple_preprocess(text)
    return " ".join(words)

cleaned_documents=[clean_text(doc) for doc in documents]

#metin tokenization 
tokenized_documents=[simple_preprocess(doc) for doc in cleaned_documents]

#word2Vec modeli tanımla 
model=Word2Vec(sentences=tokenized_documents,vector_size=50,min_count=1, sg=0, window=5)
word_vectors=model.wv
words=list(word_vectors.index_to_key)[:500]
vectors=[word_vectors[word] for word in words]

#clustering KMeans K=2

kmeans=KMeans(n_clusters=2)
kmeans.fit(vectors)
clusters=kmeans.labels_

#PCA 50 ->2 
pca=PCA(n_components=2)
reduced_vectors=pca.fit_transform(vectors)

# 2 boyutlu görselleştirme
plt.figure()
plt.scatter(reduced_vectors[:,0],reduced_vectors[:,1],c=clusters,cmap="viridis")

centers=pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0],centers[:,1],c="red",marker="x",s=150,label="Center")
plt.legend()

#figure üzerine kelimelerin eklennmesi

for i, word in enumerate(words):
    plt.text(reduced_vectors[i,0],reduced_vectors[i,1],word, fontsize=7)

plt.title("Word2Vec")
