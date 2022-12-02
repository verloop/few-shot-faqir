from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class SparseEmbedding:
    def __init__(self, sparse_embedding_method="TFIDF_WORD_EMBEDDINGS"):

        if sparse_embedding_method == "TFIDF_WORD_EMBEDDINGS":
            self.vectorizer = TfidfVectorizer(analyzer="word")
        elif sparse_embedding_method == "TFIDF_CHAR_EMBEDDINGS":
            self.vectorizer = TfidfVectorizer(analyzer="char_wb")
        elif sparse_embedding_method == "CV_EMBEDDINGS":
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer = None

    def train(self, train_sents):
        self.vectorizer.fit(train_sents)

    def get_embeddings(self, sents=[]):
        return self.vectorizer.transform(sents)
