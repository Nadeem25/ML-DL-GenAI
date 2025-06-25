
import nltk
nltk.download('punkt_tab')  # Download the punkt tokenizer
corpus = """ My name is Nadeem. I am pursuing MSC from Berlin Univeristy."""

# Corpus (Paragraph) into documents(sentences)
#import nltk.tokenize as tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
sentences = sent_tokenize(corpus, "english")
print(f"Paragraph Tokenization {sentences}")


# Pragraph to word
words = word_tokenize(corpus)
print(f"Pragraph to word ${words}")
