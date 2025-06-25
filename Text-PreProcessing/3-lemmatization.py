# Lemmatizer: It is the technique like stemming. The output we will get after lemmzatization is called "lemma". "Lemma" is the root word rather than root stem.

# POS
# Noun - n
# Verb - v
# adjactive - a
# adver - r

from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
words = ["History", "Fairly", "Sportingly", "Eating", "Eats", "Eaten", "Writing", "Write"]
for word in words:
    lemm_word = lemmatizer.lemmatize(word, pos='v')
    print(word + "---->"+ lemm_word)


