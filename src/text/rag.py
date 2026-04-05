import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from retriever import retrieve
from extractor import extract


# We define three possible verdict labels that the RAG module can return.
# SUPPORT means the Wikipedia evidence backs up the claim.
# REFUTE means evidence was found but the claim terms do not match it.
# UNKNOWN means there is not enough evidence to make a determination.
SUPPORT = "support"
REFUTE  = "refute"
UNKNOWN = "unknown"

# We define the similarity thresholds that control our verdict decisions.
# A similarity above SUPPORT_THRESHOLD with matching claim terms gives SUPPORT.
# A similarity above REFUTE_THRESHOLD but with no matching terms gives REFUTE.
# Below REFUTE_THRESHOLD we always return UNKNOWN because the evidence is too weak.
SUPPORT_THRESHOLD = 0.15
REFUTE_THRESHOLD  = 0.05
MIN_PASSAGES      = 1


def compute_similarity(claim, passages):
    # We use TF-IDF cosine similarity to measure how topically related
    # the claim is to each retrieved Wikipedia passage.
    # TF-IDF weights terms by how frequently they appear in the claim
    # relative to how common they are across all passages,
    # which makes it a good measure of topical overlap.
    if not passages:
        return 0.0, []

    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        # We fit the vectorizer on the claim plus all passages together
        # so the term frequencies are computed in the same vector space.
        all_texts = [claim] + passages
        tfidf     = vectorizer.fit_transform(all_texts)
        # We compute the cosine similarity between the claim (index 0)
        # and each of the passages (indices 1 onwards).
        sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    except Exception:
        return 0.0, []

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    # We return the passages ranked by similarity so the best evidence
    # is always in the first position.
    ranked = [passages[i] for i in np.argsort(sims)[::-1]]

    return best_sim, ranked


def claim_terms_in_evidence(claim, passages):
    # We check how many content words from the claim appear in the evidence.
    # This gives us a second signal complementary to TF-IDF similarity:
    # a high similarity score combined with matching claim terms strongly
    # suggests that the evidence is talking about the same topic as the claim.
    claim_words = set(w.lower() for w in claim.split() if len(w) > 3)
    evidence    = " ".join(passages).lower()

    matches = sum(1 for w in claim_words if w in evidence)
    return matches, len(claim_words)


def determine_verdict(similarity, matches, total_claim_words, num_passages):
    # We need at least one passage to make any verdict at all.
    if num_passages < MIN_PASSAGES:
        return UNKNOWN, 0.0

    # If the similarity is below our minimum threshold, the evidence is too
    # weakly related to the claim and we cannot draw any conclusion.
    if similarity < REFUTE_THRESHOLD:
        return UNKNOWN, similarity

    # If the similarity is strong enough and claim terms appear in the evidence,
    # we conclude that the evidence supports the claim.
    # We scale the confidence proportionally to the similarity score.
    if similarity >= SUPPORT_THRESHOLD and matches > 0:
        confidence = min(similarity * 2, 1.0)
        return SUPPORT, round(confidence, 4)

    # If there is some similarity but none of the claim terms appear in the evidence,
    # we interpret this as the evidence being about a related but different topic,
    # which we treat as a refutation signal.
    if similarity >= REFUTE_THRESHOLD and matches == 0:
        confidence = min(similarity * 1.5, 1.0)
        return REFUTE, round(confidence, 4)

    return UNKNOWN, round(similarity, 4)


def run_rag(text):
    # We run the full RAG pipeline in five steps.

    # Step 1: we extract named entities and keywords from the text
    # and build a clean Wikipedia search query.
    extraction = extract(text)
    query      = extraction["query"]

    # If the extraction produced an empty query we cannot search for anything
    # and we return UNKNOWN immediately.
    if not query.strip():
        return {
            "verdict"      : UNKNOWN,
            "confidence"   : 0.0,
            "query"        : query,
            "source_title" : None,
            "source_url"   : None,
            "best_passage" : None,
            "similarity"   : 0.0,
            "entities"     : extraction["entities"],
            "keywords"     : extraction["keywords"],
        }

    # Step 2: we search Wikipedia for relevant pages and retrieve text passages.
    retrieval = retrieve(query)

    # If no relevant Wikipedia page was found we return UNKNOWN.
    if not retrieval["found"] or not retrieval["passages"]:
        return {
            "verdict"      : UNKNOWN,
            "confidence"   : 0.0,
            "query"        : query,
            "source_title" : None,
            "source_url"   : None,
            "best_passage" : None,
            "similarity"   : 0.0,
            "entities"     : extraction["entities"],
            "keywords"     : extraction["keywords"],
        }

    # Step 3: we compute TF-IDF cosine similarity between the claim
    # and each retrieved passage, and keep the best score.
    similarity, ranked_passages = compute_similarity(text, retrieval["passages"])

    # Step 4: we check how many claim terms appear in the evidence
    # to use as a second signal alongside the similarity score.
    matches, total = claim_terms_in_evidence(text, retrieval["passages"])

    # Step 5: we combine the similarity score and term matches
    # to produce the final verdict and confidence score.
    verdict, confidence = determine_verdict(
        similarity, matches, total, len(retrieval["passages"])
    )

    return {
        "verdict"      : verdict,
        "confidence"   : confidence,
        "query"        : query,
        "source_title" : retrieval["title"],
        "source_url"   : retrieval["url"],
        "best_passage" : ranked_passages[0] if ranked_passages else None,
        "similarity"   : round(similarity, 4),
        "entities"     : extraction["entities"],
        "keywords"     : extraction["keywords"],
    }


def print_rag_result(result):
    print("\n" + "="*50)
    print("  RAG Text Analysis")
    print("="*50)
    print(f"  Query          : {result['query']}")
    print(f"  Source         : {result['source_title']}")
    print(f"  Similarity     : {result['similarity']}")
    print(f"  Verdict        : {result['verdict'].upper()}")
    print(f"  Confidence     : {result['confidence']}")
    if result["best_passage"]:
        print(f"\n  Best passage   :\n  {result['best_passage'][:200]}...")
    print("="*50)


if __name__ == "__main__":
    text = (
        sys.argv[1] if len(sys.argv) > 1
        else "NASA confirmed that the moon landing in 1969 was led by Neil Armstrong."
    )

    print(f"Input text: {text}\n")
    result = run_rag(text)
    print_rag_result(result)