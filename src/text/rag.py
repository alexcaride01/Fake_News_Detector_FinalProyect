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
SUPPORT = "support"
REFUTE  = "refute"
UNKNOWN = "unknown"

# We provide two modes for the final verdict step.
# When USE_LLM is True we use Ollama to reason over the evidence,
# which gives more accurate results but takes longer on CPU.
# When USE_LLM is False we fall back to TF-IDF cosine similarity,
# which is faster but less capable of understanding semantic meaning.
USE_LLM      = True
OLLAMA_MODEL = "llama3.2:1b"

# We define the TF-IDF thresholds used when USE_LLM is False.
SUPPORT_THRESHOLD = 0.15
REFUTE_THRESHOLD  = 0.05
MIN_PASSAGES      = 1


def compute_similarity(claim, passages):
    # We use TF-IDF cosine similarity to measure topical overlap
    # between the claim and each retrieved Wikipedia passage.
    if not passages:
        return 0.0, []

    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        all_texts = [claim] + passages
        tfidf     = vectorizer.fit_transform(all_texts)
        sims      = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    except Exception:
        return 0.0, []

    best_sim = float(sims[np.argmax(sims)])
    ranked   = [passages[i] for i in np.argsort(sims)[::-1]]

    return best_sim, ranked


def claim_terms_in_evidence(claim, passages):
    # We check how many content words from the claim appear in the evidence
    # as a second signal alongside TF-IDF similarity.
    claim_words = set(w.lower() for w in claim.split() if len(w) > 3)
    evidence    = " ".join(passages).lower()
    matches     = sum(1 for w in claim_words if w in evidence)
    return matches, len(claim_words)


def tfidf_verdict(claim, passages):
    # We combine similarity score and term matching to produce a verdict.
    # This is our fallback method when Ollama is not available.
    similarity, ranked = compute_similarity(claim, passages)
    matches, total     = claim_terms_in_evidence(claim, passages)

    if len(passages) < MIN_PASSAGES or similarity < REFUTE_THRESHOLD:
        return UNKNOWN, round(similarity, 4), "", ranked

    if similarity >= SUPPORT_THRESHOLD and matches > 0:
        return SUPPORT, round(min(similarity * 2, 1.0), 4), "", ranked

    if similarity >= REFUTE_THRESHOLD and matches == 0:
        return REFUTE, round(min(similarity * 1.5, 1.0), 4), "", ranked

    return UNKNOWN, round(similarity, 4), "", ranked


def llm_verdict(claim, passages):
    # We pass the claim and the Wikipedia passages to Ollama as context.
    # The model reads both and decides whether the evidence supports,
    # refutes or is insufficient to verify the claim.
    # This is the Augmented Generation step of our RAG pipeline:
    # we retrieved the evidence, augmented the prompt with it,
    # and now the model generates the verdict and explains its reasoning.
    try:
        import ollama

        # We join all passages into a single evidence block
        # so the model has the full context in one place.
        evidence = "\n\n".join(passages)

        prompt = f"""You are a fact-checking assistant. Your task is to determine whether a claim is supported or refuted by the Wikipedia evidence provided.

Claim: {claim}

Wikipedia evidence:
{evidence}

Instructions:
- Read the claim and the evidence carefully.
- Start your response with exactly one of these words: support, refute, or unknown.
- Then on a new line, write one sentence explaining why, based only on the evidence.
- Be concise and specific. Mention the key fact from the evidence that led to your decision.

Example format:
refute
The evidence states that X is located in Paris, France, not in London as the claim suggests.

Your answer:"""

        response = ollama.chat(
            model    = OLLAMA_MODEL,
            messages = [{"role": "user", "content": prompt}]
        )

        # We parse the first line for the verdict and the second line for the explanation.
        # The model is instructed to put the verdict on the first line and the
        # explanation on the second line so we can parse them reliably.
        raw         = response["message"]["content"].strip()
        lines       = [l.strip() for l in raw.split("\n") if l.strip()]
        first_line  = lines[0].lower() if lines else ""
        explanation = lines[1] if len(lines) > 1 else ""

        if "support" in first_line:
            verdict    = SUPPORT
            confidence = 0.80
        elif "refute" in first_line:
            verdict    = REFUTE
            confidence = 0.75
        else:
            verdict    = UNKNOWN
            confidence = 0.0

        return verdict, confidence, explanation

    except ImportError:
        # We warn the user if the ollama library is not installed
        # and fall back to TF-IDF automatically.
        print("Warning: ollama library not found. Falling back to TF-IDF.")
        return None, None, ""

    except Exception as e:
        # We catch any other errors such as the Ollama server not running
        # and fall back to TF-IDF so the pipeline does not crash.
        print(f"Warning: Ollama error ({e}). Falling back to TF-IDF.")
        return None, None, ""


def run_rag(text):
    # Step 1: we extract entities and keywords and build a Wikipedia search query.
    extraction = extract(text)
    query      = extraction["query"]

    if not query.strip():
        return {
            "verdict"         : UNKNOWN,
            "confidence"      : 0.0,
            "query"           : query,
            "source_title"    : None,
            "source_url"      : None,
            "best_passage"    : None,
            "similarity"      : 0.0,
            "llm_explanation" : "",
            "entities"        : extraction["entities"],
            "keywords"        : extraction["keywords"],
            "llm_used"        : False,
        }

    # Step 2: we search Wikipedia and retrieve relevant text passages.
    retrieval = retrieve(query)

    if not retrieval["found"] or not retrieval["passages"]:
        return {
            "verdict"         : UNKNOWN,
            "confidence"      : 0.0,
            "query"           : query,
            "source_title"    : None,
            "source_url"      : None,
            "best_passage"    : None,
            "similarity"      : 0.0,
            "llm_explanation" : "",
            "entities"        : extraction["entities"],
            "keywords"        : extraction["keywords"],
            "llm_used"        : False,
        }

    # Step 3: we determine the verdict using either Ollama or TF-IDF.
    llm_used        = False
    similarity      = 0.0
    best_passage    = retrieval["passages"][0]
    llm_explanation = ""

    if USE_LLM:
        # We try to get the verdict from Ollama first.
        # If it fails for any reason we fall back to TF-IDF automatically.
        print("  Sending evidence to Ollama for reasoning...")
        verdict, confidence, llm_explanation = llm_verdict(text, retrieval["passages"])

        if verdict is not None:
            sim, ranked  = compute_similarity(text, retrieval["passages"])
            similarity   = round(sim, 4)
            best_passage = ranked[0] if ranked else retrieval["passages"][0]
            llm_used     = True
        else:
            # We fall back to TF-IDF if Ollama is not available.
            verdict, confidence, _, ranked = tfidf_verdict(text, retrieval["passages"])
            similarity   = round(compute_similarity(text, retrieval["passages"])[0], 4)
            best_passage = ranked[0] if ranked else retrieval["passages"][0]
    else:
        verdict, confidence, _, ranked = tfidf_verdict(text, retrieval["passages"])
        similarity   = round(compute_similarity(text, retrieval["passages"])[0], 4)
        best_passage = ranked[0] if ranked else retrieval["passages"][0]

    return {
        "verdict"         : verdict,
        "confidence"      : confidence,
        "query"           : query,
        "source_title"    : retrieval["title"],
        "source_url"      : retrieval["url"],
        "best_passage"    : best_passage,
        "similarity"      : similarity,
        "llm_explanation" : llm_explanation,
        "entities"        : extraction["entities"],
        "keywords"        : extraction["keywords"],
        "llm_used"        : llm_used,
    }


def print_rag_result(result):
    print("\n" + "="*50)
    print("  RAG Text Analysis")
    print("="*50)
    print(f"  Query          : {result['query']}")
    print(f"  Source         : {result['source_title']}")
    print(f"  Similarity     : {result['similarity']}")
    print(f"  LLM used       : {result['llm_used']}")
    print(f"  Verdict        : {result['verdict'].upper()}")
    print(f"  Confidence     : {result['confidence']}")
    if result["llm_explanation"]:
        print(f"  LLM reasoning  : {result['llm_explanation']}")
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