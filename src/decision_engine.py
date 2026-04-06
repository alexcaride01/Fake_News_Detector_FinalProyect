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

    