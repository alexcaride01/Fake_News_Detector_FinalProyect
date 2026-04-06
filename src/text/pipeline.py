import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ocr import extract_text, has_text
from rag import run_rag


def run_text_pipeline(image_path):
    # We run the full text analysis pipeline in two steps.
    # First we extract any text embedded in the image using OCR.
    # Then, if we found enough text, we run the RAG pipeline to verify
    # the claim against external evidence from Wikipedia.

    # Step 1: we attempt to extract text from the image using our
    # multi-strategy OCR module. The result may be an empty string
    # if the image contains no readable text.
    text       = extract_text(image_path)
    text_found = has_text(text)

    # If we did not find enough text to work with, we return an UNKNOWN verdict
    # with zero confidence. The decision engine will then rely entirely
    # on the CNN visual signal to make the final decision.
    if not text_found:
        return {
            "text_found"    : False,
            "extracted_text": "",
            "verdict"       : "unknown",
            "confidence"    : 0.0,
            "query"         : None,
            "source_title"  : None,
            "source_url"    : None,
            "similarity"    : 0.0,
            "entities"      : [],
            "keywords"      : [],
        }

    # Step 2: we pass the extracted text through the full RAG pipeline,
    # which extracts entities, searches Wikipedia and computes the verdict.
    rag_result = run_rag(text)

    # We return a unified dictionary that contains all the information
    # the decision engine needs to make its final verdict.
    return {
        "text_found"    : True,
        "extracted_text": text,
        "verdict"       : rag_result["verdict"],
        "confidence"    : rag_result["confidence"],
        "query"         : rag_result["query"],
        "source_title"  : rag_result["source_title"],
        "source_url"    : rag_result["source_url"],
        "similarity"      : rag_result["similarity"],
        "llm_explanation" : rag_result.get("llm_explanation", ""),
        "entities"      : rag_result["entities"],
        "keywords"      : rag_result["keywords"],
    }


def print_pipeline_result(result):
    print("\n" + "="*50)
    print("  Text Pipeline Result")
    print("="*50)
    print(f"  Text found      : {result['text_found']}")
    if result["text_found"]:
        print(f"  Extracted text  : {result['extracted_text'][:100]}...")
        print(f"  Query           : {result['query']}")
        print(f"  Source          : {result['source_title']}")
        print(f"  Similarity      : {result['similarity']}")
        print(f"  Verdict         : {result['verdict'].upper()}")
        print(f"  Confidence      : {result['confidence']}")
    print("="*50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/text/pipeline.py <path_to_image>")
        print("Example: python src/text/pipeline.py dataset/test/fake/img001.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: image not found at '{image_path}'")
        sys.exit(1)

    print(f"Processing image: {image_path}")
    result = run_text_pipeline(image_path)
    print_pipeline_result(result)