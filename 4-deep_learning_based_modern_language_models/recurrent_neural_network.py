"""
RNN İLE SINIFLANDIRMA PROBLEMİ ÇÖZÜMÜ(DUYGU ANALİZİ )
duygu analizi: bir cümlenin etiketlenmesi(pozitive ve negative )
restaurant yorum değerlendirilmesi. 
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec #netin temsili
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential #Keras’ta sinir ağı oluşturmak için kullanılan en temel model tipi
from keras.layers import SimpleRNN, Dense,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split #Veri setini test ve train olarak bölmeyi sağlar.
from sklearn.preprocessing import LabelEncoder#Metin türündeki kategorik etiketleri (string) → sayılara (integer) dönüştürmek için kullanılır.

#veri setini oluştur.
data = {
    "text": [
        # --- Positive (50) ---
        "Yemekler harikaydı, servis çok hızlıydı.",
        "Garsonlar çok ilgiliydi, tekrar gelirim.",
        "Tatlılar efsaneydi, kesinlikle tavsiye ederim.",
        "Yemekler sıcaktı ve lezzet mükemmeldi.",
        "Atmosfer çok hoştu, kendimi evde gibi hissettim.",
        "Fiyatlar makuldü, porsiyonlar doyurucuydu.",
        "Sunum çok şıktı, yemekler göz kamaştırıcıydı.",
        "Personel güler yüzlüydü, hizmet kusursuzdu.",
        "Mekan çok temizdi, hijyen açısından içim rahat etti.",
        "Müzikler ve ortam çok keyifliydi.",
        "Kebap mükemmeldi, eti tam kıvamında pişmişti.",
        "Servis çok hızlıydı, bekletmediler.",
        "Çorba sıcaktı ve çok lezzetliydi.",
        "Pizza çok tazeydi, malzemeleri boldu.",
        "Garsonlar kibar ve profesyoneldi.",
        "Yemeklerin sunumu çok estetikti.",
        "Mekanın ambiyansı romantikti, çok beğendim.",
        "Tatlılar tam kıvamındaydı, şeker oranı dengeliydi.",
        "Yemekleri çok beğendik, tekrar geleceğiz.",
        "Servis mükemmeldi, çalışanlar çok saygılıydı.",
        "Yemek muhteşemdi, şefin eline sağlık.",
        "Patates kızartması çok çıtırdı, bayıldım.",
        "Pizzanın peyniri uzuyordu, enfesti.",
        "Tatlı mükemmeldi, porsiyonu da büyüktü.",
        "Çalışanlar çok yardımseverdi.",
        "Mekan çok huzurluydu, sakin müzikler çalıyordu.",
        "Yemekler taze ve sıcaktı, her şey dört dörtlüktü.",
        "Fiyat performans olarak gayet iyi.",
        "Garsonlar çok nazikti.",
        "Sunum özenliydi, tabaklar tertemizdi.",
        "Kebap lezzetliydi, yanında gelen mezeler harikaydı.",
        "Servis hızlıydı, siparişimiz kısa sürede geldi.",
        "Tatlılar çok tazeydi, şeker oranı tam yerindeydi.",
        "Mekanın dekorasyonu çok hoştu.",
        "Garsonlar güler yüzlüydü, ortam samimiydi.",
        "Porsiyonlar yeterliydi, fiyat da uygundu.",
        "Lezzet muhteşemdi, kesinlikle tavsiye ederim.",
        "Tatlılar harikaydı, şefin ellerine sağlık.",
        "Sunum mükemmeldi, fotoğraflık bir tabaktı.",
        "Garsonlar çok özenliydi, ilgi alaka süperdi.",
        "Mekan ferah ve rahattı.",
        "Yemek sıcak ve çok lezzetliydi.",
        "Çorba nefisti, baharat oranı tam yerindeydi.",
        "Tatlılar efsane, kahveyle birlikte harika gitti.",
        "Servis hızlı ve profesyoneldi.",
        "Garsonlar samimi ve güler yüzlüydü.",
        "Porsiyonlar doyurucuydu, fiyat da uygundu.",
        "Lezzetli yemekler, hoş ortam, kesinlikle tavsiye ederim.",
        "Garsonlar çok ilgiliydi, hizmet kalitesi mükemmeldi.",
        "Restoran çok konforluydu, kendimi özel hissettim.",
        #"Tatlılar enfesti, sunum da harikaydı.",

        # --- Negative (50) ---
        "Yemekler çok pişmişti, tadı kalmamıştı.",
        "Garsonlar ilgisizdi, siparişim yanlış geldi.",
        "Tatlı bayattı, hiç beğenmedim.",
        "Servis çok yavaştı, yarım saat bekledik.",
        "Yemek tuzluydu, yiyemedim.",
        "Masalar kirliydi, hijyen sıfırdı.",
        "Garson surat asıyordu, rahatsız ediciydi.",
        "Porsiyonlar çok küçüktü, doymadım.",
        "Fiyatlar çok yüksekti, verdiğimiz paraya değmedi.",
        "Tat kötüydü, malzemeler bayat gibiydi.",
        "Hamburger ekmeği yanmıştı.",
        "Siparişim eksik geldi, ilgilenmediler.",
        "Garsonlar çok kaba davrandı.",
        "Çatal bıçaklar temiz değildi.",
        "Yemekler geç geldi, soğumuştu.",
        "Tatlı aşırı şekerliydi, yenmiyordu.",
        "Salata taze değildi, limon suyu bile yoktu.",
        "Çorba ılıktı, sıcak bekliyordum.",
        "Restoran çok gürültülüydü, rahat edemedim.",
        "Servis berbat, bir daha asla gelmem.",
        "Yemek soğuktu, tadı da kötüydü.",
        "Servis çok kötüydü, ilgilenen kimse yoktu.",
        "Tatlı bayattı, param boşa gitti.",
        "Garson yanlış sipariş getirdi.",
        "Mekan çok kalabalıktı, gürültüden duramadık.",
        "Yemekler yağ içindeydi, midemi bozdu.",
        "Masalar yapış yapıştı, hijyen hiç yok.",
        "Fiyatlar aşırı pahalıydı.",
        "Garson kaba konuştu.",
        "Tatlı çok tuhaftı, tadını anlamadım.",
        "Yemek geç geldi, üstelik soğuktu.",
        "Servis rezaletti.",
        "Yemek çok tuzluydu, yiyemedim.",
        "Tatlı bayattı, hayal kırıklığı yaşadım.",
        "Servis berbattı, ilgilenmediler.",
        "Garson ilgisizdi, yanlış sipariş getirdi.",
        "Mekan çok kirliydi, hijyen sıfır.",
        "Fiyatlar çok yüksekti, değmez.",
        "Tat kötüydü, et sertti.",
        "Garson kaba davrandı, çok rahatsız ediciydi.",
        "Servis çok yavaştı.",
        "Masalar lekeliydi, hijyen eksikti.",
        "Yemek lezzetsizdi, baharat yoktu.",
        "Tatlı kötüydü, şeker oranı fazla.",
        "Servis ilgisizdi, hiç memnun kalmadım.",
        "Yemek yanmıştı, tadı kötüydü.",
        "Tatlı donuktu, taze değildi.",
        "Mekan havasızdı, rahatsız oldum.",
        "Garsonlar ilgisizdi, hizmet yetersizdi.",
        "Porsiyonlar küçüktü, doymadık."
    ],
    "label": [
        *["positive"] * 50,
        *["negative"] * 50
    ]
}

df= pd.DataFrame(data)

#metin temizleme işlemleri ve preprocessing: tokenization, padding, label encodeing, train test split 

tokenizer=Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences=tokenizer.texts_to_sequences(df["text"])
word_index=tokenizer.word_index

#padding process
maxlen=max(len(seq) for seq in sequences)
X=pad_sequences(sequences,maxlen=maxlen)
print(X.shape)

#label encoding
label_encoder= LabelEncoder()
y=label_encoder.fit_transform(df["label"])

#train test split

x_train, x_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)

#Metin temsili: word embeddinng: word2vec 

sentences=[text.split() for text in df["text"]]
word2vec_model=Word2Vec(sentences, window=5,min_count=1,vector_size=50)

embedding_dim=50
embedding_matrix=np.zeros((len(word_index)+1,embedding_dim))
for word,i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i]=word2vec_model.wv[word]
#modelling: buil. train ve test: rrn model 

#build model
model=Sequential()

#embedding
model.add(Embedding(input_dim=len(word_index)+1,output_dim=embedding_dim,weights=[embedding_matrix],input_length=maxlen, trainable=False))

#RNN layer
model.add(SimpleRNN(50, return_sequences=False))

#Output layer
model.add(Dense(1, activation="sigmoid"))

#compile layer
model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])

#train model
model.fit(x_train,y_train,epochs=20,batch_size=2,validation_data=(x_test,y_test))

#evaluate rrn model
test_loss , test_accuracy=model.evaluate(x_test,y_test)
print(f"test loss:{test_loss}")
print(f"test_accuracy:{test_accuracy}")

#cümle sınıflandırma çalışması

def classify_sentence(sentence):
    seq=tokenizer.texts_to_sequences([sentence])
    padded_seq=pad_sequences(seq,maxlen=maxlen)
    prediction=model.predict(padded_seq)
    predicted_class=(prediction >0.5).astype(int)
    label="positive" if predicted_class[0][0]==1 else "negative"
    return label


sentence="Restaurant çok temizdi ve yemekler çok güzeldi."
result=classify_sentence(sentence)
print(f"result:{result}")




