import sys
import wikipedia


# We set the Wikipedia language to English so all searches and page content
# are returned in English, which is the language our NLP models expect.
wikipedia.set_lang("en")

# We retrieve up to 5 passages per page to give the RAG module enough
# context to make a good similarity comparison with the claim.
MAX_PASSAGES = 5
# We aim for passages of around 300 characters each, which is long enough
# to contain meaningful sentences but short enough to stay focused.
PASSAGE_LEN  = 300


def get_page_text(page):
    # We use the full page content instead of just the summary because
    # the summary often starts with infobox data (coordinates, dates, dimensions)
    # that contains very few meaningful sentences and confuses TF-IDF similarity.
    # The full content gives us much richer introductory paragraphs to compare against.
    try:
        full  = page.content
        # We remove section headers (lines starting with ==) because they are
        # structural markers and do not contribute useful content for similarity.
        lines = [l for l in full.split("\n") if not l.strip().startswith("==")]
        text  = " ".join(lines)
        text  = " ".join(text.split())
        # We limit to the first 4000 characters because the most relevant
        # information is always in the introductory section of the article.
        return text[:4000]
    except Exception:
        # If the full content is not available we fall back to the summary.
        return page.summary


def split_into_passages(text):
    # We split the page text into passages of approximately PASSAGE_LEN characters.
    # We split on sentence boundaries rather than fixed character counts
    # to avoid cutting sentences in the middle.
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

    # We add any remaining text as a final passage if we have not reached the limit.
    if current.strip() and len(passages) < MAX_PASSAGES:
        passages.append(current.strip())

    return passages


def try_load_page(title):
    # We try to load a Wikipedia page by its title.
    # If the page is a disambiguation page, we take the first suggested option
    # since it is usually the most common meaning of the term.
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
    # We compute a relevance score for a Wikipedia page given our search query.
    # We check how many query words appear in the page title and summary.
    # We weight title matches more heavily because a title match is a much
    # stronger signal of relevance than a summary match.
    query_words  = set(w.lower() for w in query.split() if len(w) > 3)
    title_words  = set(w.lower() for w in page.title.split())
    summary      = page.summary.lower()

    title_hits   = sum(1 for w in query_words if w in title_words)
    summary_hits = sum(1 for w in query_words if w in summary)

    return title_hits * 3 + summary_hits


def retrieve(query):
    # We build a list of candidate queries ranging from the full query
    # to progressively shorter versions and finally individual key words.
    # This strategy allows us to find a relevant page even when the full
    # query does not match any Wikipedia title directly.
    words = [w for w in query.split() if len(w) > 2]

    candidate_queries = [query]

    for n in range(len(words) - 1, 1, -1):
        candidate_queries.append(" ".join(words[:n]))

    # We also add the longest individual words as standalone queries
    # in case the multi-word queries all fail to find a relevant page.
    sorted_words = sorted(words, key=len, reverse=True)
    for word in sorted_words[:3]:
        if word not in candidate_queries:
            candidate_queries.append(word)

    best_page  = None
    best_score = -1

    # We try each candidate query and keep the page with the highest relevance score.
    # We stop early as soon as we find a page with a strong enough score
    # to avoid making unnecessary Wikipedia API calls.
    for cq in candidate_queries:
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

            if best_score >= 3:
                break

        if best_score >= 3:
            break

    # We only return a result if we found a page with at least a minimal
    # relevance score. A score below 1 means none of the query terms
    # appeared in the page, which means the page is likely irrelevant.
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
    query  = sys.argv[1] if len(sys.argv) > 1 else "Eiffel Tower Paris France"
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