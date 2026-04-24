import math
from collections import Counter, defaultdict
import re

def preprocess(sentence):
    """小写 + 去除标点"""
    return re.sub(r'[^\w\s]', '', sentence.lower()).split()

def get_ngrams(tokens, n):
    """生成n-gram"""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def compute_tf(ngrams):
    """计算TF"""
    tf = Counter(ngrams)
    total = sum(tf.values())
    return {ng: count / total for ng, count in tf.items()}

def compute_df(refs, n):
    """计算文档频率"""
    df = defaultdict(int)
    for ref in refs:
        tokens = preprocess(ref)
        ngrams = set(get_ngrams(tokens, n))
        for ng in ngrams:
            df[ng] += 1
    return df

def compute_idf(df, N):
    """IDF = log(N / (df + 1))"""
    return {ng: math.log(N / (df[ng] + 1.0)) for ng in df}

def vectorize(tf, idf):
    """TF-IDF 向量"""
    return {ng: tf[ng] * idf.get(ng, 0) for ng in tf}

def cosine_similarity(v1, v2):
    """余弦相似度"""
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in set(v1) | set(v2))
    norm1 = math.sqrt(sum(v**2 for v in v1.values()))
    norm2 = math.sqrt(sum(v**2 for v in v2.values()))
    return dot / (norm1 * norm2 + 1e-8)

def compute_cider(hypo, refs, n_max=4):
    scores = []
    N = len(refs)

    for n in range(1, n_max+1):
        df = compute_df(refs, n)
        idf = compute_idf(df, N)

        ref_vectors = []
        for ref in refs:
            ref_ngrams = get_ngrams(preprocess(ref), n)
            tf = compute_tf(ref_ngrams)
            ref_vectors.append(vectorize(tf, idf))

        hypo_tokens = preprocess(hypo)
        hypo_ngrams = get_ngrams(hypo_tokens, n)
        hypo_tf = compute_tf(hypo_ngrams)
        hypo_vector = vectorize(hypo_tf, idf)

        sims = [cosine_similarity(hypo_vector, ref_vec) for ref_vec in ref_vectors]
        scores.append(sum(sims) / len(sims))

    return sum(scores) / len(scores)