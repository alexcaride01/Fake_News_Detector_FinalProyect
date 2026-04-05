import sys
import re
import spacy


# We load the small English spaCy model which provides named entity recognition,
# part-of-speech tagging and noun chunk detection.
# It is lightweight enough to run on CPU without any noticeable delay.
nlp = spacy.load("en_core_web_sm")

# We define the entity types we want to extract for fact-checking purposes.
# These types cover the most informative elements in news claims:
# people, organizations, locations, dates, events and products.
RELEVANT_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT", "NORP", "FAC", "PRODUCT"}

# We filter out words that appear frequently in news headlines but add no
# factual value to a Wikipedia search query.
# Including these words tends to lead the search to irrelevant pages.
STOP_QUERY_WORDS = {
    "breaking", "news", "official", "officially", "report", "reports",
    "according", "sources", "say", "says", "claim", "claims", "allegedly",
    "confirm", "confirms", "confirmed", "announce", "announces", "announced",
    "breaking news", "just in", "exclusive", "update", "latest",
}

# We define a set of articles and determiners that we want to remove
# from our search terms because they add noise to the Wikipedia query
# without contributing any factual information.
ARTICLES = {"the", "a", "an", "its", "their", "our", "this", "that", "these", "those"}


def extract_entities(text):
    # We run the spaCy NLP pipeline on the input text to identify named entities.
    # We deduplicate entities using a lowercase set so we do not include
    # the same entity twice with different capitalization.
    doc      = nlp(text)
    entities = []
    seen     = set()

    for ent in doc.ents:
        if ent.label_ in RELEVANT_ENTITY_TYPES:
            clean = ent.text.strip()
            if clean.lower() not in seen:
                seen.add(clean.lower())
                entities.append({"text": clean, "label": ent.label_})

    return entities


def extract_keywords(text, max_keywords=10):
    # We extract noun chunks first because they capture multi-word concepts
    # like "bald eagle" or "national bird" that are more informative than
    # individual words on their own.
    # We then add individual nouns and proper nouns that were not already captured.
    doc      = nlp(text)
    keywords = []
    seen     = set()

    for chunk in doc.noun_chunks:
        # We remove articles from the beginning of each noun chunk
        # to avoid including "the" in our search query terms.
        words = [w for w in chunk.text.strip().split() if w.lower() not in ARTICLES]
        clean = " ".join(words).lower()
        if len(clean) > 2 and clean not in STOP_QUERY_WORDS:
            if clean not in seen:
                seen.add(clean)
                keywords.append(clean)

    # We also add individual content words that were not already covered
    # by the noun chunks to ensure we do not miss any important terms.
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
            clean = token.text.strip().lower()
            if (len(clean) > 2
                    and clean not in seen
                    and clean not in STOP_QUERY_WORDS):
                seen.add(clean)
                keywords.append(clean)

    return keywords[:max_keywords]


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
    # We limit to 3 entities so the query does not become too long.
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

    # Step 2: we fill the remaining slots with keywords, preferring
    # multi-word terms over single words because they are more specific.
    # We also avoid adding terms that are already covered by existing ones.
    multi  = [kw for kw in keywords if len(kw.split()) > 1]
    single = [kw for kw in keywords if len(kw.split()) == 1]

    for kw in multi + single:
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

    # Step 3: if we still do not have enough terms, we fall back to extracting
    # individual nouns directly from the original text as a last resort.
    if len(terms) < 2:
        doc = nlp(original_text)
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
                cleaned = clean_term(token.text.strip())
                if (len(cleaned) > 3
                        and cleaned.lower() not in STOP_QUERY_WORDS
                        and cleaned.lower() not in seen):
                    seen.add(cleaned.lower())
                    terms.append(cleaned)
                if len(terms) >= max_terms:
                    break

    return " ".join(terms[:max_terms])


def extract(text):
    # We run the full extraction pipeline and return all results in a single dictionary.
    # This dictionary is then consumed by the RAG module to build the Wikipedia query.
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