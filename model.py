from sentence_transformers import SentenceTransformer


def model(params):
    pre_trained_model = params.pre_trained_model
    return SentenceTransformer(pre_trained_model)