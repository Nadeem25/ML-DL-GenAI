import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

data = {
    "label": ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "ham", "spam"],
    "message": [
        "Win money now by clicking this link!",
        "Hi, how are you doing today?",
        "You have been selected for a prize!",
        "Let's meet for lunch tomorrow.",
        "Claim your free gift card now!",
        "Can you call me later?",
        "Congratulations! You've won a lottery.",
        "Don't forget the meeting at 3 PM.",
        "Are we still going to the cinema?",
        "Get rich quick with this new method!"
    ]
}
df = pd.DataFrame(data)
message = df.message
#print(df.message)

# Step 1: Data Cleaning and Pre Processing
corpus = []
for i in range(0, len(df.message)):
    review = re.sub('[^a-zA-Z]', ' ', message[i]) # Replace any char aprt from a-zA-Z replace with " "
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # Consider word which are not in stopwords and find out the word stem
    review = " ".join(review)
    corpus.append(review)
print(f"Review: {corpus}")

# Step 2: Convert the text into vector using bag of words technique
cv = CountVectorizer(max_features=2500, binary=True, ngram_range=(1,2))

# cv.fit(corpus)
# print(f"Vocabulary: {cv.vocabulary_}")

vector = cv.fit_transform(corpus).toarray()
print(f"Vocabulary: {cv.vocabulary_}")
print(f"Vector Value: {vector}")





