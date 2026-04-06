import sys
import re
import nltk


# We download the required NLTK resources the first time the module is loaded.
# These resources are small and only need to be downloaded once.
# averaged_perceptron_tagger is used for part-of-speech tagging.
# maxent_ne_chunker and words are used for named entity recognition.
# punkt is used for sentence and word tokenization.
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("maxent_ne_chunker_tab", quiet=True)
nltk.download("words", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag


# We define the set of English stopwords we want to filter out.
STOP_WORDS = set(stopwords.words("english"))

# We filter out words that appear frequently in news headlines but add no
# factual value to a Wikipedia search query.
STOP_QUERY_WORDS = {
    "breaking", "news", "official", "officially", "report", "reports",
    "according", "sources", "say", "says", "claim", "claims", "allegedly",
    "confirm", "confirms", "confirmed", "announce", "announces", "announced",
    "just", "exclusive", "update", "latest",
}

# We define articles and determiners to remove from search terms.
ARTICLES = {"the", "a", "an", "its", "their", "our", "this", "that", "these", "those"}

# We map NLTK NE chunk labels to readable entity type names.
ENTITY_LABEL_MAP = {
    "PERSON"       : "PERSON",
    "ORGANIZATION" : "ORG",
    "GPE"          : "GPE",
    "LOCATION"     : "LOC",
    "FACILITY"     : "FAC",
    "GSP"          : "GPE",
}


def extract_entities(text):
    # We use NLTK's named entity chunker to identify named entities in the text.
    # We tokenize and POS-tag the text first, then run the NE chunker on the result.
    # The chunker groups tokens into named entity chunks with labels like
    # PERSON, ORGANIZATION and GPE (geopolitical entity).
    tokens   = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tree     = ne_chunk(pos_tags)

    entities = []
    seen     = set()

    for subtree in tree:
        if hasattr(subtree, "label"):
            label = ENTITY_LABEL_MAP.get(subtree.label(), subtree.label())
            name  = " ".join(word for word, tag in subtree.leaves()).strip()
            if name.lower() not in seen:
                seen.add(name.lower())
                entities.append({"text": name, "label": label})

    return entities


def extract_keywords(text, max_keywords=10):
    # We extract keywords by keeping only nouns and proper nouns
    # that are not stopwords and not in our query stop list.
    # We use POS tags to identify nouns: NN (noun), NNS (plural noun),
    # NNP (proper noun) and NNPS (plural proper noun).
    tokens   = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    keywords = []
    seen     = set()

    for word, tag in pos_tags:
        clean = word.strip().lower()
        if (tag in {"NN", "NNS", "NNP", "NNPS"}
                and clean not in STOP_WORDS
                and clean not in STOP_QUERY_WORDS
                and clean not in ARTICLES
                and len(clean) > 2
                and clean not in seen):
            seen.add(clean)
            keywords.append(clean)

        if len(keywords) >= max_keywords:
            break

    return keywords


def clean_term(term):
    # We remove articles from the beginning of a term and strip
    # punctuation that could interfere with the Wikipedia search.
    words = [w for w in term.split() if w.lower() not in ARTICLES]
    clean = " ".join(words)
    clean = re.sub(r"[!?\"'()]", "", clean).strip()
    return clean


def build_search_query(entities, keywords, original_text, max_terms=5):
    # We build the search query in three steps, prioritizing the most
    # informative terms first and falling back to less specific ones if needed.
    terms = []
    seen  = set()

    # Step 1: we add named entities first because they are the most specific
    # and informative terms for a Wikipedia search.
    for e in entities:
        cleaned = clean_term(e["text"])
        if (cleaned
                and len(cleaned) > 2
                and cleaned.lower() not in STOP_QUERY_WORDS
                and cleaned.lower() not in seen):
            seen.add(cleaned.lower())
            terms.append(cleaned)
        if len(terms) >= 3:
            break

    # Step 2: we fill the remaining slots with keywords.
    for kw in keywords:
        if len(terms) >= max_terms:
            break
        cleaned = clean_term(kw)
        if (cleaned
                and len(cleaned) > 2
                and cleaned.lower() not in seen
                and cleaned.lower() not in STOP_QUERY_WORDS
                and not any(cleaned.lower() in t.lower() or t.lower() in cleaned.lower()
                            for t in terms)):
            seen.add(cleaned.lower())
            terms.append(cleaned)

    # Step 3: if we still do not have enough terms, we fall back to
    # extracting individual nouns directly from the original text.
    if len(terms) < 2:
        tokens   = word_tokenize(original_text)
        pos_tags = pos_tag(tokens)
        for word, tag in pos_tags:
            if tag in {"NN", "NNS", "NNP", "NNPS"}:
                cleaned = clean_term(word.strip())
                if (len(cleaned) > 3
                        and cleaned.lower() not in STOP_QUERY_WORDS
                        and cleaned.lower() not in seen):
                    seen.add(cleaned.lower())
                    terms.append(cleaned)
                if len(terms) >= max_terms:
                    break

    return " ".join(terms[:max_terms])


def extract(text):
    # We run the full extraction pipeline and return all results in a dictionary.
    # This dictionary is consumed by the RAG module to build the Wikipedia query.
    entities = extract_entities(text)
    keywords = extract_keywords(text)
    query    = build_search_query(entities, keywords, text)

    return {
        "entities": entities,
        "keywords": keywords,
        "query"   : query,
    }


if __name__ == "__main__":
    sample = (
        sys.argv[1] if len(sys.argv) > 1
        else "Breaking news: The United States has officially replaced the bald eagle with the golden retriever as its national bird!!!"
    )

    print(f"Input: {sample}\n")
    result = extract(sample)

    print("Entities:")
    for e in result["entities"]:
        print(f"  {e['label']:8s} -> {e['text']}")

    print(f"\nKeywords: {result['keywords']}")
    print(f"\nSearch query: {result['query']}")