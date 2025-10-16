
# kütüphanelerin aktarılması
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
from nltk.corpus import stopwords

#veri setinin içeriye aktarılması
df=pd.read_csv("IMDB Dataset.csv")

#metin verilerini alalım.
documents=df["review"]
labels=df["sentiment"]  # pozitif veya negatif 
stop_word_eng=set(stopwords.words("english"))

#metin temizleme adımları
def clean_text(text):
    text= text.lower()
    text=re.sub(r"\d+", "", text)
    text= re.sub(r"^\w\s", "", text)
    text= " ".join([word for word in text.split() if len(word)>2])
    text=" ".join([word for word in text.split() if word.lower() not in stop_word_eng ])
    return text

cleaned_doc=[clean_text(row) for row in documents]

#bow
#vectorizer tanımla
vectorizer= CountVectorizer()

#metinleri sayısal hale getir
x=vectorizer.fit_transform(cleaned_doc[:75])

#kelime kümesi göster
feature_names=vectorizer.get_feature_names_out()

#vektör temsilini göster
vektor_temsili2=x.toarray()

print(f"vektor_temsili:{vektor_temsili2}")
df_bow=pd.DataFrame(vektor_temsili2,columns=feature_names)

#kelime frekanslarını göster
word_counts= x.sum(axis=0).A1
word_freq=dict(zip(feature_names,word_counts))

most_common_5_words=Counter(word_freq).most_common(5)
print(f"most_common_5_words: {most_common_5_words}")