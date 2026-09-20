from bertopic import BERTopic

topic_model = BERTopic.load("results-1/bertopic_model")

vocab = topic_model.vectorizer_model.vocabulary_
#print("the" in vocab, "and" in vocab)
print(topic_model.vectorizer_model.get_params()["stop_words"])
