from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class SparseEmbedding:
    def __init__(self, sparse_embedding_method="tfidf-word"):

        if sparse_embedding_method == "tfidf-word":
            self.vectorizer = TfidfVectorizer(analyzer="word")
        elif sparse_embedding_method == "tfidf-char":
            self.vectorizer = TfidfVectorizer(analyzer="char_wb")
        elif sparse_embedding_method == "cv":
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer = None

    def train(self, train_sents):
        self.vectorizer.fit(train_sents)

    def get_embeddings(self, sents=[]):
        return self.vectorizer.transform(sents)
