
#VERİ TEMİZLEME

text= " Hello,    World!     2025"

cleaned_text1= " ".join(text.split())
print(cleaned_text1)

cleaned_text2= cleaned_text1.lower()
print(cleaned_text2)

import string

cleaned_text3=cleaned_text2.translate(str.maketrans("","",string.punctuation))
print(cleaned_text3)


import re
text2= " Hello,    World!     2025^#"
cleaned_text4= re.sub(r"[^A-Za-z0-9\s]","", text2)
print(cleaned_text4)


from textblob import TextBlob #metin analizlerinde kullanılan bir kütüphane
text3="Hellio Wirld  2035"
cleaned_text5=TextBlob(text3).correct() # correct: yazım hatalarını düzeltir.
print(cleaned_text5)

from bs4 import BeautifulSoup
html_text= "<div>Hello,    World!     2025</div>"
cleaned_text6=BeautifulSoup(html_text,"html.parser").get_text()
print(cleaned_text6)


