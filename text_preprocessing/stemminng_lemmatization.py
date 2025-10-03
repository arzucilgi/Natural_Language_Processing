import nltk

nltk.download("wordnet") # Wordnet:Lemmatization işlemi için gerekli veri tabanı

from nltk.stem import PorterStemmer # stemming için fonksiyon

#porterStemmer nesnesi oluştur
stemmer=PorterStemmer()

words=["running", "runner", "runs", "ran", "better", "go", "went"]

#Kelimlerin köklerini buluyoruz. Bunu yaparken de PorterStemmer in stem() fonksiyonunu kullanıyoruz.
stems= [stemmer.stem(w) for w in words]

print(stems)

#lemmatization
 
from nltk.stem import WordNetLemmatizer

lemmatizer= WordNetLemmatizer()
lemmas= [lemmatizer.lemmatize(w, pos="v") for w in words]

print(lemmas)

