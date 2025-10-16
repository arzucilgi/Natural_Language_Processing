#kütüphaneleri içeir aktar

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
import string

#veri setini içeri aktar
df = pd.read_csv("spam.csv",encoding="latin-1")

stop_word_eng=set(stopwords.words("english"))
documents=df["v2"]

#metin temizleme
#metin temizleme adımları
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Küçük harfe çevir
    text = text.lower()
    
    # URL'leri kaldır
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Email adreslerini kaldır
    text = re.sub(r"\S+@\S+", "", text)
    
    # Sayıları kaldır
    text = re.sub(r"\d+", "", text)
    
    # Noktalama ve özel karakterleri kaldır
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Emoji ve özel karakterleri kaldır
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    
    # Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()
    
    # Stopword ve kısa kelimeleri kaldır
    words = [word for word in text.split() if len(word) > 2 and word not in stop_word_eng]
    
    return " ".join(words)

cleaned_doc=[clean_text(row) for row in documents]

#tf-idf vectorizer 
vectorizer=TfidfVectorizer()
x=vectorizer.fit_transform(cleaned_doc)

#kelime kümesini incele
feature_names=vectorizer.get_feature_names_out()
tfidf_score=x.mean(axis=0).A1 #Her kelimenin ortalama tf-idf değerleri


#tfidf skorlarını içeren bir df oluştur.
df_tfidf=pd.DataFrame({"word": feature_names, "tfidf_score":tfidf_score})

#skorları sırala ve sonuçları incele.
df_tfidf_sorted=df_tfidf.sort_values(by="tfidf_score",ascending=False)
print(df_tfidf_sorted.head(10))
