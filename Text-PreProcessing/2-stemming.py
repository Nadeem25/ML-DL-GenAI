# Stemming: It is the process of reducing word to its word stem that affixes to suffixes and prefixesto or to the roots of words knows as lemma.
# 1. Snowball Stemmber
from nltk.stem import SnowballStemmer
snowball_stemmber = SnowballStemmer("english")

words = ["History", "Fairly", "Sportingly", "Eating", "Eats", "Eaten"]
for word in words:
    stem_word = snowball_stemmber.stem(word)
    print(word + "---->"+ stem_word)
