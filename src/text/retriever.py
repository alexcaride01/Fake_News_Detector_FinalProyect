import sys
import wikipedia


# We set the Wikipedia language to English so all searches and page content
# are returned in English, which is the language our NLP models expect.
wikipedia.set_lang("en")

MAX_PASSAGES = 5
PASSAGE_LEN  = 300


def get_page_text(page):
    # We use the full page content instead of just the summary because
    # the summary often starts with infobox data that confuses TF-IDF.
    try:
        full  = page.content
        lines = [l for l in full.split("\n") if not l.strip().startswith("==")]
        text  = " ".join(lines)
        text  = " ".join(text.split())
        return text[:4000]
    except Exception:
        return page.summary


def split_into_passages(text):
    # We split the page text into passages of approximately PASSAGE_LEN characters.
    passages  = []
    sentences = text.split(". ")
    current   = ""

    for sentence in sentences:
        current += sentence + ". "
        if len(current) >= PASSAGE_LEN:
            passages.append(current.strip())
            current = ""
            if len(passages) >= MAX_PASSAGES:
                break

    if current.strip() and len(passages) < MAX_PASSAGES:
        passages.append(current.strip())

    return passages


def try_load_page(title):
    # We try to load a Wikipedia page by title, handling disambiguation pages
    # by selecting the first suggested option.
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            return wikipedia.page(e.options[0], auto_suggest=False)
        except Exception:
            return None
    except Exception:
        return None


def query_similarity_score(query, page):
    # We compute a relevance score based on how many query words appear
    # in the page title and summary. We weight title matches more heavily
    # because a title match is a stronger signal of relevance.
    query_words  = set(w.lower() for w in query.split() if len(w) > 3)
    title_words  = set(w.lower() for w in page.title.split())
    summary      = page.summary.lower()

    title_hits   = sum(1 for w in query_words if w in title_words)
    summary_hits = sum(1 for w in query_words if w in summary)

    return title_hits * 3 + summary_hits


def build_candidate_queries(query):
    # We build candidate queries starting with the most specific named entities.
    # The key insight is that for fact-checking we want to find a page about
    # the SUBJECT of the claim, not about the location or other details.
    # So we try shorter versions of the query progressively, which tends to
    # isolate the main subject (usually at the beginning of the query).
    words = [w for w in query.split() if len(w) > 2]

    candidates = []

    # We add the full query first
    candidates.append(query)

    # We try progressively shorter versions from the start
    # since the most important terms tend to come first
    for n in range(min(len(words), 3), 0, -1):
        short = " ".join(words[:n])
        if short not in candidates:
            candidates.append(short)

    # We also try the longest individual words as standalone queries
    sorted_words = sorted(words, key=len, reverse=True)
    for word in sorted_words[:3]:
        if word not in candidates:
            candidates.append(word)

    return candidates


def retrieve(query):
    # We try each candidate query and keep the page with the highest
    # relevance score. We prioritize pages that match the beginning
    # of the query since that usually contains the main subject.
    best_page  = None
    best_score = -1

    for cq in build_candidate_queries(query):
        try:
            search_results = wikipedia.search(cq, results=3)
        except Exception:
            continue

        for title in search_results:
            page = try_load_page(title)
            if page is None:
                continue

            score = query_similarity_score(query, page)

            if score > best_score:
                best_score = score
                best_page  = page

            # We stop early if we find a strong enough match
            if best_score >= 4:
                break

        # We stop searching if we already have a strong match
        if best_score >= 4:
            break

    if best_page is None or best_score < 1:
        return {
            "query"    : query,
            "found"    : False,
            "title"    : None,
            "url"      : None,
            "passages" : [],
        }

    text     = get_page_text(best_page)
    passages = split_into_passages(text)

    return {
        "query"    : query,
        "found"    : True,
        "title"    : best_page.title,
        "url"      : best_page.url,
        "passages" : passages,
    }


if __name__ == "__main__":
    query  = sys.argv[1] if len(sys.argv) > 1 else "Eiffel Tower London England"
    print(f"Query: {query}\n")
    result = retrieve(query)

    if not result["found"]:
        print("No relevant Wikipedia page found.")
    else:
        print(f"Page found : {result['title']}")
        print(f"URL        : {result['url']}")
        print(f"Passages   : {len(result['passages'])}\n")
        for i, p in enumerate(result["passages"]):
            print(f"  [{i+1}] {p[:200]}...")