import nltk  # natural language toolkit
nltk.download("punkt") #metni kelime ve cümle bazında tokenlarına ayırmak için gereklidir.
nltk.download("punkt_tab")

text= "Hello World! How are you? Helloo,  hi ..."

#kelime tokenizasyonu:word_tokenize: Metni kelimelere ayırır.
# Noktalama işaretleri ve boşluklar ayrı birer token olarak elde edilecektir.

word_tokens=nltk.word_tokenize(text)


#Cümle tokenizasyonu:sent_tokenize:Metni kelimelere ayırır. Her bir cümle token olur.


