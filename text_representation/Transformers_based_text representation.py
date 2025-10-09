
from transformers import AutoTokenizer, AutoModel
import torch

#model ve tokenizer yükle 
model_name= "bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModel.from_pretrained(model_name)
#imput text (metni) tanımla
text=" Transformers can be used for natural language processing. "

#metni tokenlara çevir
inputs=tokenizer(text,return_tensors="pt")  #çıktı pytorch tensoru olarak  return edilir.

#modeli kullanarak metin temsili oluştur.
with torch.no_grad(): # gradyanların hesaplanması durdurulur, böylece belleği daha verimli kullanırız.
    outputs=model(**inputs)

#modelin çıkışından son gizli durumu alalım
last_hidden_state=outputs.last_hidden_state

#ilk tokenin embedding i al ve print ettir.
first_token_embedding=last_hidden_state[0,0,:].numpy()
print(f"Metin Temsili:{first_token_embedding}")
