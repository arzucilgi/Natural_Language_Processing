
import nltk

from nltk.corpus import stopwords
nltk.download("stopwords")  #farklı dillerde en çok kullanılan stop words içeren veri seti

# ingilizce stop words analizi (nltk)

stop_word_eng=set(stopwords.words("english"))

text="There are some examples of handling stop words from some texts"
text_list= text.split()
filtered_words=[word for word in text_list if word.lower() not in stop_word_eng ]
print(filtered_words)


# türkçe stop words analizi  (nltk)

stop_word_tr=set(stopwords.words("turkish"))

text1="Merhaba arkadaşlar çok güzel bir ders işliyoruz. Bu ders faydalı mı?"
text_list2=text1.split()
filtered_words_tr=[word for word in text_list2 if word.lower() not in stop_word_tr]
print(filtered_words_tr)


# kütüphanesiz stop words çıkarımı

tr_stopwords=["için", "bu", "ile", "mu", "mi", "özel"]

metin="Bu bir denemedir. Amacımız bu metinde bulunan özel karakterleri elemek mi acaba?"
filtered_word= [word for word in metin.split() if word.lower() not in tr_stopwords]
filtered_stopwords=set([word.lower() for word in metin.split() if word.lower() in tr_stopwords])
print(filtered_word)
print(filtered_stopwords)


