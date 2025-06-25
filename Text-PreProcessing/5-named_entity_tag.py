
import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')


sentence = "The Eiffel Tower was built from 1887 to 1889 by French engineer Gustave Eiffel, whose company specialized in building metal frameworks and structures."
words = nltk.word_tokenize(sentence)
tag_elemnts = nltk.pos_tag(words)
named_entity_reg = nltk.ne_chunk(tag_elemnts).draw()
