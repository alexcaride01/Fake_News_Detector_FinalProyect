import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# We define the four possible final verdict labels.
# FAKE means we have strong evidence of manipulation or a false claim.
# REAL means both the image and the claim appear to be authentic.
# DOUBTFUL means the signals are mixed and the user should investigate further.
# UNKNOWN means we do not have enough evidence to make any determination.
FAKE     = "fake"
REAL     = "real"
DOUBTFUL = "doubtful"
UNKNOWN  = "unknown"

# We define the CNN probability thresholds that separate our confidence levels.
# If p_fake is above HIGH_FAKE_THRESHOLD we consider the image very likely fake.
# If p_fake is below LOW_FAKE_THRESHOLD we consider the image likely real.
# Values in between are treated as uncertain.
HIGH_FAKE_THRESHOLD = 0.75
LOW_FAKE_THRESHOLD  = 0.40

# We only trust the RAG verdict if its confidence is above this threshold.
# Below this value the text evidence is too weak to influence the final decision
# and we rely primarily on the CNN signal instead.
MIN_TEXT_CONFIDENCE = 0.20


def decide(p_fake, text_verdict, text_confidence, text_found):
    # We receive four inputs from the two upstream modules.
    # p_fake is the probability that the image is fake according to the CNN.
    # text_verdict is the RAG verdict: "support", "refute" or "unknown".
    # text_confidence is how confident the RAG module is in its verdict.
    # text_found indicates whether the OCR module found usable text in the image.

    # We build a list of reasons that we will return alongside the verdict
    # so the final output is fully explainable to the user.
    reasons = []

    # Case 1: no readable text was found in the image.
    # In this case we rely entirely on the CNN visual signal to make the decision.
    # We cannot cross-check any claim so we acknowledge this limitation in the explanation.
    if not text_found:
        if p_fake >= HIGH_FAKE_THRESHOLD:
            verdict    = FAKE
            confidence = p_fake
            reasons.append(f"Image manipulation probability is high ({p_fake:.2f}). No text found to cross-check.")
        elif p_fake <= LOW_FAKE_THRESHOLD:
            verdict    = REAL
            confidence = 1 - p_fake
            reasons.append(f"Image appears authentic ({1-p_fake:.2f} confidence). No text found to cross-check.")
        else:
            verdict    = DOUBTFUL
            confidence = 0.5
            reasons.append(f"CNN is uncertain (p_fake={p_fake:.2f}) and no text was found.")
        return verdict, round(confidence, 4), reasons

    # Case 2: text was found but the RAG confidence is too low to be useful.
    # This happens when Wikipedia did not find a relevant article for the claim.
    # We fall back to the CNN signal but we note that the text could not be verified.
    if text_confidence < MIN_TEXT_CONFIDENCE:
        if p_fake >= HIGH_FAKE_THRESHOLD:
            verdict    = FAKE
            confidence = p_fake
            reasons.append(f"Image manipulation probability is high ({p_fake:.2f}).")
            reasons.append("Text evidence is insufficient to confirm or deny the claim.")
        elif p_fake <= LOW_FAKE_THRESHOLD:
            verdict    = DOUBTFUL
            confidence = 0.4
            reasons.append(f"Image appears authentic but text evidence is insufficient.")
            reasons.append("Cannot verify the claim without external evidence.")
        else:
            verdict    = UNKNOWN
            confidence = 0.3
            reasons.append("Neither the image analysis nor the text evidence is conclusive.")
        return verdict, round(confidence, 4), reasons

 # Case 3: both signals are available and the RAG confidence is high enough.
    # We now apply a set of rules that cover all meaningful combinations of
    # the CNN and RAG signals to produce the most informative final verdict.

    # Rule 3a: the image looks manipulated AND the claim is contradicted by Wikipedia.
    # Both signals point in the same direction, so we return FAKE with high confidence.
    if p_fake >= HIGH_FAKE_THRESHOLD and text_verdict == "refute":
        verdict    = FAKE
        confidence = min((p_fake + text_confidence) / 2 + 0.1, 1.0)
        reasons.append(f"Image shows signs of manipulation (p_fake={p_fake:.2f}).")
        reasons.append(f"The textual claim is contradicted by external evidence (confidence={text_confidence:.2f}).")
        return verdict, round(confidence, 4), reasons

    # Rule 3b: the image looks manipulated BUT the claim is supported by Wikipedia.
    # The signals disagree, which could mean the image is real but used out of context.
    # We return DOUBTFUL to flag this ambiguity to the user.
    if p_fake >= HIGH_FAKE_THRESHOLD and text_verdict == "support":
        verdict    = DOUBTFUL
        confidence = 0.55
        reasons.append(f"Image shows signs of manipulation (p_fake={p_fake:.2f}).")
        reasons.append("However, the textual claim is supported by external evidence.")
        reasons.append("The image may be real but used out of context.")
        return verdict, round(confidence, 4), reasons

    # Rule 3c: the image looks authentic AND the claim is supported by Wikipedia.
    # Both signals agree that the content is legitimate, so we return REAL.
    if p_fake <= LOW_FAKE_THRESHOLD and text_verdict == "support":
        verdict    = REAL
        confidence = min((1 - p_fake + text_confidence) / 2 + 0.1, 1.0)
        reasons.append(f"Image appears authentic (p_fake={p_fake:.2f}).")
        reasons.append(f"The textual claim is supported by external evidence (confidence={text_confidence:.2f}).")
        return verdict, round(confidence, 4), reasons

    # Rule 3d: the image looks authentic BUT the claim is refuted by Wikipedia.
    # The image itself may be real, but the associated text claim is false.
    # This is a common disinformation pattern where a genuine image is paired
    # with a fabricated caption or headline.
    if p_fake <= LOW_FAKE_THRESHOLD and text_verdict == "refute":
        verdict    = FAKE
        confidence = min(text_confidence + 0.1, 1.0)
        reasons.append(f"Image appears authentic (p_fake={p_fake:.2f}).")
        reasons.append(f"However, the textual claim is refuted by external evidence (confidence={text_confidence:.2f}).")
        reasons.append("The image may be real but the associated claim is false.")
        return verdict, round(confidence, 4), reasons

    # Rule 3e: the RAG verdict is UNKNOWN, meaning no relevant evidence was found.
    # Without external evidence we cannot confirm or deny the claim,
    # so we return DOUBTFUL regardless of what the CNN says.
    if text_verdict == "unknown":
        verdict    = DOUBTFUL
        confidence = 0.4
        reasons.append(f"CNN is uncertain (p_fake={p_fake:.2f}).")
        reasons.append("No sufficient external evidence found to verify the claim.")
        return verdict, round(confidence, 4), reasons

    # Rule 3f: the image is suspicious but the RAG returned UNKNOWN.
    # We lean towards DOUBTFUL because the image signal is concerning
    # but we cannot confirm it with text evidence.
    if p_fake >= HIGH_FAKE_THRESHOLD and text_verdict == "unknown":
        verdict    = DOUBTFUL
        confidence = p_fake * 0.8
        reasons.append(f"Image shows signs of manipulation (p_fake={p_fake:.2f}).")
        reasons.append("No sufficient external evidence found to verify the claim.")
        return verdict, round(confidence, 4), reasons

    # Fallback: if none of the rules above matched we return DOUBTFUL
    # with low confidence to signal that the result is inconclusive.
    verdict    = DOUBTFUL
    confidence = 0.4
    reasons.append("Signals are mixed or inconclusive.")
    return verdict, round(confidence, 4), reasons


def print_decision(p_fake, text_verdict, text_confidence, text_found, verdict, confidence, reasons):
    print("\n" + "="*50)
    print("  Decision Engine")
    print("="*50)
    print(f"  CNN p_fake       : {p_fake:.4f}")
    print(f"  Text found       : {text_found}")
    print(f"  Text verdict     : {text_verdict}")
    print(f"  Text confidence  : {text_confidence:.4f}")
    print(f"\n  FINAL VERDICT    : {verdict.upper()}")
    print(f"  Confidence       : {confidence:.4f}")
    print(f"\n  Explanation:")
    for reason in reasons:
        print(f"    - {reason}")
    print("="*50)


if __name__ == "__main__":
    # We define a set of test cases covering the main scenarios the system can encounter.
    # Each case has a known expected output so we can quickly verify the rules are working.
    test_cases = [
        (0.92, "refute",  0.45, True,  "CNN fake + RAG refutes -> should be FAKE"),
        (0.85, "support", 0.50, True,  "CNN fake + RAG supports -> should be DOUBTFUL"),
        (0.20, "support", 0.40, True,  "CNN real + RAG supports -> should be REAL"),
        (0.15, "refute",  0.55, True,  "CNN real + RAG refutes -> should be FAKE"),
        (0.88, "unknown", 0.10, True,  "CNN fake + no evidence -> should be DOUBTFUL"),
        (0.80, "unknown", 0.00, False, "CNN fake + no text -> should be FAKE"),
        (0.30, "unknown", 0.00, False, "CNN real + no text -> should be REAL"),
    ]

    for p_fake, text_verdict, text_conf, text_found, description in test_cases:
        print(f"\nScenario: {description}")
        verdict, confidence, reasons = decide(p_fake, text_verdict, text_conf, text_found)
        print_decision(p_fake, text_verdict, text_conf, text_found, verdict, confidence, reasons)   