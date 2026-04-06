import sys
import pytesseract
from PIL import Image, ImageEnhance
import cv2
import numpy as np


# We set the path to the Tesseract executable on Windows.
# Tesseract is an external OCR engine that pytesseract calls under the hood.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# We define multiple Tesseract page segmentation modes to try on each image.
# Different modes work better depending on how the text is laid out in the image.
# psm 6 assumes a single uniform block of text.
# psm 11 finds as much text as possible even if it is scattered.
# psm 3 is fully automatic page segmentation.
# psm 4 assumes a single column of text.
CONFIGS = [
    "--psm 6",
    "--psm 11",
    "--psm 3",
    "--psm 4",
]


def upscale(img, min_width=1400):
    # We upscale images that are too small because Tesseract performs significantly
    # better on larger images where the characters have more pixels to work with.
    # We only upscale if the width is below our minimum threshold.
    h, w = img.shape[:2]
    if w < min_width:
        scale = min_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


def run_tesseract(pil_img):
    # We try all our Tesseract configurations and keep the result that
    # produces the most words. Different configurations work better for
    # different image layouts so trying all of them gives us the best coverage.
    best = ""
    for cfg in CONFIGS:
        try:
            text = pytesseract.image_to_string(pil_img, lang="eng", config=cfg)
            # We clean up extra whitespace and newlines to get a single clean string.
            text = " ".join(text.split())
            if len(text.split()) > len(best.split()):
                best = text
        except Exception:
            continue
    return best


def get_variants(img_bgr):
    # We generate multiple preprocessed versions of the image.
    # Each variant is designed to make a specific type of text more readable
    # for Tesseract. We return all variants and try each one, keeping
    # the result with the most extracted words.
    variants = []
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # We create four yellow text masks with different HSV ranges.
    # Yellow text on dark backgrounds is very common in news images and memes.
    # We use the HSV color space because it separates color from brightness,
    # making it easier to isolate specific colors regardless of lighting conditions.
    for lo_h, hi_h, lo_s, lo_v in [
        (15, 45, 60, 60),    # Standard yellow range
        (10, 50, 40, 40),    # Wider range including golden tones
        (18, 38, 80, 100),   # Pure bright yellow
        (20, 35, 100, 150),  # Highly saturated yellow
    ]:
        mask = cv2.inRange(hsv,
            np.array([lo_h, lo_s, lo_v]),
            np.array([hi_h, 255, 255])
        )
        # We dilate the mask slightly to reconnect characters that may have
        # been split due to anti-aliasing or compression artifacts.
        kernel = np.ones((2, 2), np.uint8)
        mask   = cv2.dilate(mask, kernel, iterations=1)
        # We invert the mask so the text appears black on a white background,
        # which is the format Tesseract works best with.
        inv = cv2.bitwise_not(upscale(mask))
        variants.append((f"yellow_{lo_h}_{hi_h}", inv))

    # We identify and remove the blue background to isolate the text.
    # Blue in HSV falls roughly in the range 95-145.
    # We replace blue pixels with white so only the text remains visible.
    blue_mask = cv2.inRange(hsv, np.array([95, 50, 30]), np.array([145, 255, 255]))
    no_blue   = img_bgr.copy()
    no_blue[blue_mask > 0] = [255, 255, 255]
    gray_no_blue = cv2.cvtColor(no_blue, cv2.COLOR_BGR2GRAY)
    _, thresh_no_blue = cv2.threshold(upscale(gray_no_blue), 180, 255, cv2.THRESH_BINARY_INV)
    variants.append(("no_blue_thresh", thresh_no_blue))

    # We extract the red channel because yellow pixels have high values in both
    # the red and green channels. Using the red channel alone can help isolate
    # yellow text from darker backgrounds.
    r = upscale(img_bgr[:, :, 2])
    _, r_thresh = cv2.threshold(r, 160, 255, cv2.THRESH_BINARY)
    variants.append(("red_thresh", cv2.bitwise_not(r_thresh)))

    # We also try the green channel for a similar reason.
    g = upscale(img_bgr[:, :, 1])
    _, g_thresh = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)
    variants.append(("green_thresh", cv2.bitwise_not(g_thresh)))

    # We combine the red and green channels by taking the minimum value at each pixel.
    # Pixels that are bright in both channels correspond to yellow regions,
    # which makes this a strong indicator of yellow text.
    rg_combined = cv2.min(img_bgr[:, :, 2], img_bgr[:, :, 1])
    rg_combined = upscale(rg_combined)
    _, rg_thresh = cv2.threshold(rg_combined, 140, 255, cv2.THRESH_BINARY)
    variants.append(("rg_combined", cv2.bitwise_not(rg_thresh)))

    # We include standard grayscale variants as general-purpose fallbacks
    # that work well for black text on light backgrounds.
    gray_up = upscale(gray)
    variants.append(("gray", gray_up))
    variants.append(("gray_inv", cv2.bitwise_not(gray_up)))

    # CLAHE (Contrast Limited Adaptive Histogram Equalization) improves contrast
    # locally across the image, which helps with uneven lighting conditions.
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    variants.append(("clahe",     clahe.apply(gray_up)))
    variants.append(("clahe_inv", cv2.bitwise_not(clahe.apply(gray_up))))

    # Otsu's thresholding automatically finds the optimal threshold value
    # to separate text from background in a bimodal image.
    _, otsu = cv2.threshold(gray_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu",     otsu))
    variants.append(("otsu_inv", cv2.bitwise_not(otsu)))

    # Adaptive thresholding computes a local threshold for each region of the image,
    # which handles images where the background brightness varies across the image.
    adp = cv2.adaptiveThreshold(
        gray_up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
    )
    variants.append(("adaptive",     adp))
    variants.append(("adaptive_inv", cv2.bitwise_not(adp)))

    # We isolate white text by masking pixels with high values in all three channels.
    # White pixels have high R, G and B values simultaneously.
    # This works well for white text on any colored background including red.
    white_mask = cv2.inRange(img_bgr, np.array([180, 180, 180]), np.array([255, 255, 255]))
    white_up   = upscale(white_mask)
    variants.append(("white_text", cv2.bitwise_not(white_up)))

    # We suppress the red background by masking red pixels in HSV
    # and replacing them with white, leaving only non-red content visible.
    # Red in HSV wraps around 0 and 180 so we need two ranges.
    red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]),   np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_mask  = cv2.bitwise_or(red_mask1, red_mask2)
    no_red    = img_bgr.copy()
    no_red[red_mask > 0] = [255, 255, 255]
    gray_no_red  = cv2.cvtColor(no_red, cv2.COLOR_BGR2GRAY)
    _, no_red_thresh = cv2.threshold(upscale(gray_no_red), 200, 255, cv2.THRESH_BINARY_INV)
    variants.append(("no_red_thresh", no_red_thresh))

    # We also try inverting the blue channel because red pixels have low blue values
    # while white pixels have high blue values, creating good contrast.
    b_inv = cv2.bitwise_not(upscale(img_bgr[:, :, 0]))
    _, b_thresh = cv2.threshold(b_inv, 100, 255, cv2.THRESH_BINARY)
    variants.append(("blue_inv_thresh", b_thresh))

    return variants


def extract_text(image_path):
    # We start by trying Tesseract directly on the original PIL image
    # as our initial baseline result.
    pil_orig  = Image.open(image_path).convert("RGB")
    best_text = run_tesseract(pil_orig)

    # We also try a contrast and sharpness enhanced version of the original image.
    # This often helps when the text is slightly blurry or low contrast.
    try:
        enhanced = ImageEnhance.Contrast(pil_orig).enhance(2.0)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
        t = run_tesseract(enhanced)
        if len(t.split()) > len(best_text.split()):
            best_text = t
    except Exception:
        pass

    # We then try all our preprocessed variants and keep the one
    # that gives us the most words extracted.
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return best_text

        for name, processed in get_variants(img_bgr):
            try:
                pil_img = Image.fromarray(processed)
                text    = run_tesseract(pil_img)
                if len(text.split()) > len(best_text.split()):
                    best_text = text
                # We stop early if we already have a solid result
                # to avoid spending extra time on remaining variants.
                if len(best_text.split()) >= 20:
                    break
            except Exception:
                continue

    except Exception:
        pass

    return best_text


def has_text(text, min_words=3):
    # We consider text valid if it contains at least min_words words.
    # We use a low threshold of 3 to be as permissive as possible
    # and avoid discarding images that do contain text.
    return len(text.strip().split()) >= min_words


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/text/ocr.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    text       = extract_text(image_path)

    print(f"Image     : {image_path}")
    print(f"Has text  : {has_text(text)}")
    print(f"Word count: {len(text.split())}")
    print(f"Extracted :\n{text}")