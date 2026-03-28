# import numpy as np
# import torch
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk

# nltk.download('punkt')

# # Checking if CUDA device is available
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# sbert_model = SentenceTransformer(
#     "all-MiniLM-L6-v2",
#     device=DEVICE
# )

# # Splitting Sentences
# def split_sentences(text):
#     return [s.strip() for s in nltk.sent_tokenize(text) if len(s.strip()) > 0]


# def self_consistency(sampled_answers, batch_size=32):

#     if len(sampled_answers) < 2:
#         return 0.0

#     base_answer = sampled_answers[0]
#     other_answers = sampled_answers[1:]

#     # Splitting Sentences
#     base_sents = split_sentences(base_answer)
#     sample_sents = [split_sentences(ans) for ans in other_answers]

#     if len(base_sents) == 0:
#         return 0.0

#     # Flattening all sentences for batch encoding
#     all_sentences = base_sents + [s for sample in sample_sents for s in sample]


#     # Encoding in a single batch
#     embeddings = sbert_model.encode(
#         all_sentences,
#         batch_size=batch_size,
#         convert_to_numpy=True,
#         normalize_embeddings=True
#     )

#     # Split embeddings back
#     base_embs = embeddings[:len(base_sents)]
#     idx = len(base_sents)

#     sample_embs = []
#     for sample in sample_sents:
#         n = len(sample)
#         sample_embs.append(embeddings[idx:idx+n])
#         idx += n


#     # Computing consistency
#     sentence_scores = []

#     for i, base_emb in enumerate(base_embs):
#         base_emb = base_emb.reshape(1, -1)

#         per_sample_scores = []

#         for sample_emb in sample_embs:
#             if len(sample_emb) == 0:
#                 continue

#             sims = np.dot(base_emb, sample_emb.T)

#             # best matching sentence
#             max_sim = np.max(sims)
#             per_sample_scores.append(max_sim)

#         if len(per_sample_scores) > 0:
#             sentence_scores.append(np.mean(per_sample_scores))

#     if len(sentence_scores) == 0:
#         return 0.0


#     # Converting to inconsistency
#     consistency_score = np.mean(sentence_scores)
#     inconsistency_score = 1.0 - consistency_score

#     return float(inconsistency_score)

import numpy as np
import nltk

def split_sentences(text):
    return [s.strip() for s in nltk.sent_tokenize(text) if len(s.strip()) > 0]


def self_consistency(sampled_answers, sbert_model, batch_size=32):
    if len(sampled_answers) < 2:
        return 0.0

    base_answer = sampled_answers[0]
    other_answers = sampled_answers[1:]

    base_sents = split_sentences(base_answer)
    sample_sents = [split_sentences(ans) for ans in other_answers]

    if len(base_sents) == 0:
        return 0.0

    all_sentences = base_sents + [s for sample in sample_sents for s in sample]

    embeddings = sbert_model.encode(
        all_sentences,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    base_embs = embeddings[:len(base_sents)]
    idx = len(base_sents)

    sample_embs = []
    for sample in sample_sents:
        n = len(sample)
        sample_embs.append(embeddings[idx:idx+n])
        idx += n

    sentence_scores = []

    for base_emb in base_embs:
        base_emb = base_emb.reshape(1, -1)
        per_sample_scores = []

        for sample_emb in sample_embs:
            if len(sample_emb) == 0:
                continue

            sims = np.dot(base_emb, sample_emb.T)
            max_sim = np.max(sims)
            per_sample_scores.append(max_sim)

        if len(per_sample_scores) > 0:
            sentence_scores.append(np.mean(per_sample_scores))

    if len(sentence_scores) == 0:
        return 0.0

    consistency_score = np.mean(sentence_scores)
    return float(consistency_score)