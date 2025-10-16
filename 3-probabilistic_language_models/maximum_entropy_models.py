from nltk.classify import MaxentClassifier


# veri seti tanımlama
train_Data=[({"love":True," amazing":True, "happy":True, "terrible":False},"positive"),
            ({"hate":True, "terrible":True},"negative"),
            ({"joy":True,"happy":True, "hate":False},"positive"),
            ({"sad":True, "depressed":True, "love":False},"negative")
            ]

#train max entropy classifier

classifier=MaxentClassifier.train(train_Data, max_iter=10)

# yeni test cümlesi

test_sentence="I  love this movie and it was amazing"
features={word :(word in test_sentence.lower().split()) for word in ["love","amazing","terrible","happy","hate","joy","sad","depressed"]}

label=classifier.classify(features)
print(f"Result : {label}")