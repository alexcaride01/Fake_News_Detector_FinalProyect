import sys
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# We try multiple Tesseract page segmentation modes because different modes
# work better depending on how the text is laid out in the image.
CONFIGS = [
    "--psm 6",   # Single uniform block of text
    "--psm 11",  # Sparse text, find as much as possible
    "--psm 3",   # Fully automatic page segmentation
    "--psm 4",   # Single column of text
    "--psm 7",   # Single text line
]


def upscale(img, min_width=1400):
    # We upscale small images because Tesseract performs significantly
    # better when characters have more pixels to work with.
    h, w = img.shape[:2]
    if w < min_width:
        scale = min_width / w
        img   = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    return img


def run_tesseract(pil_img):
    # We try all Tesseract configurations and keep the result with most words.
    best = ""
    for cfg in CONFIGS:
        try:
            text = pytesseract.image_to_string(pil_img, lang="eng", config=cfg)
            text = " ".join(text.split())
            if len(text.split()) > len(best.split()):
                best = text
        except Exception:
            continue
    return best


def isolate_color(img_bgr, hsv, lo, hi, dilate=True):
    # We create a binary mask isolating pixels within the given HSV range.
    # We invert the result so the isolated color becomes black on white,
    # which is the format Tesseract reads best.
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    if dilate:
        kernel = np.ones((2, 2), np.uint8)
        mask   = cv2.dilate(mask, kernel, iterations=1)
    return cv2.bitwise_not(upscale(mask))


def suppress_color(img_bgr, hsv, lo, hi):
    # We replace pixels of the given color with white to remove a background color,
    # leaving only the text visible against a clean white background.
    mask             = cv2.inRange(hsv, np.array(lo), np.array(hi))
    result           = img_bgr.copy()
    result[mask > 0] = [255, 255, 255]
    gray             = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh        = cv2.threshold(upscale(gray), 180, 255, cv2.THRESH_BINARY_INV)
    return thresh


def get_variants(img_bgr):
    # We generate a comprehensive set of preprocessed image variants.
    # Each variant is designed to make a specific type of text more readable.
    # We cover text isolation by color, background suppression, individual
    # channel extraction and standard grayscale techniques.
    variants = []
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # We isolate white text by finding pixels with high values in all channels.
    # This works for white text on any dark or colored background.
    white = cv2.inRange(img_bgr, np.array([170, 170, 170]), np.array([255, 255, 255]))
    variants.append(("white",        cv2.bitwise_not(upscale(white))))
    white2 = cv2.inRange(img_bgr, np.array([200, 200, 200]), np.array([255, 255, 255]))
    variants.append(("white_strict", cv2.bitwise_not(upscale(white2))))

    # We isolate black text by finding pixels with low values in all channels.
    # This works for black text on any light background.
    black = cv2.inRange(img_bgr, np.array([0, 0, 0]), np.array([80, 80, 80]))
    variants.append(("black", upscale(cv2.bitwise_not(black))))

    # We isolate yellow text using multiple HSV ranges to cover standard yellow,
    # golden tones and highly saturated yellow that appear in news images and memes.
    variants.append(("yellow_wide",   isolate_color(img_bgr, hsv, [10, 40, 40],   [45, 255, 255])))
    variants.append(("yellow_pure",   isolate_color(img_bgr, hsv, [18, 80, 100],  [38, 255, 255])))
    variants.append(("yellow_gold",   isolate_color(img_bgr, hsv, [10, 60, 60],   [30, 255, 255])))
    variants.append(("yellow_bright", isolate_color(img_bgr, hsv, [20, 100, 150], [35, 255, 255])))

    # We also combine red and green channels to isolate yellow regions,
    # since yellow pixels have high values in both the red and green channels.
    rg = cv2.min(img_bgr[:, :, 2], img_bgr[:, :, 1])
    _, rg_t = cv2.threshold(upscale(rg), 140, 255, cv2.THRESH_BINARY)
    variants.append(("rg_combined", cv2.bitwise_not(rg_t)))

    # We suppress the blue background and let yellow text stand out.
    variants.append(("no_blue_bg_yellow", suppress_color(img_bgr, hsv, [95, 50, 30], [145, 255, 255])))

    # We isolate red text using two HSV ranges because red wraps around
    # 0 and 180 degrees in the HSV color wheel.
    red1 = cv2.inRange(hsv, np.array([0, 100, 100]),   np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red  = cv2.bitwise_or(red1, red2)
    kernel = np.ones((2, 2), np.uint8)
    red  = cv2.dilate(red, kernel, iterations=1)
    variants.append(("red", cv2.bitwise_not(upscale(red))))

    # We isolate blue text using two ranges to cover both light and dark blues.
    variants.append(("blue",      isolate_color(img_bgr, hsv, [100, 80,  80],  [130, 255, 255])))
    variants.append(("blue_dark", isolate_color(img_bgr, hsv, [100, 100, 50],  [140, 255, 200])))

    # We isolate green text covering a broad range of green shades.
    variants.append(("green",       isolate_color(img_bgr, hsv, [40, 60,  60],  [80, 255, 255])))
    variants.append(("green_dark",  isolate_color(img_bgr, hsv, [40, 80,  40],  [75, 255, 180])))

    # We isolate orange text which sits between red and yellow in HSV.
    variants.append(("orange",      isolate_color(img_bgr, hsv, [5,  100, 100], [20, 255, 255])))

    # We isolate pink and magenta text which appears in many memes and graphics.
    variants.append(("pink",        isolate_color(img_bgr, hsv, [140, 60, 100], [170, 255, 255])))
    variants.append(("magenta",     isolate_color(img_bgr, hsv, [145, 80, 80],  [175, 255, 255])))

    # We isolate cyan and teal text.
    variants.append(("cyan",        isolate_color(img_bgr, hsv, [80,  60, 100], [100, 255, 255])))

    # We suppress common background colors to isolate text of any color on them.
    variants.append(("no_blue_bg",  suppress_color(img_bgr, hsv, [95,  50,  30],  [145, 255, 255])))
    variants.append(("no_red_bg",   suppress_color(img_bgr, hsv, [0,   100, 100], [10,  255, 255])))
    variants.append(("no_green_bg", suppress_color(img_bgr, hsv, [40,  60,  60],  [80,  255, 255])))
    variants.append(("no_black_bg", suppress_color(img_bgr, hsv, [0,   0,   0],   [180, 255, 50])))
    variants.append(("no_dark_bg",  suppress_color(img_bgr, hsv, [0,   0,   0],   [180, 255, 80])))

    # We extract individual color channels because each channel can reveal
    # text that is invisible or hard to read in the combined grayscale image.
    r = upscale(img_bgr[:, :, 2])
    g = upscale(img_bgr[:, :, 1])
    b = upscale(img_bgr[:, :, 0])
    _, r_t = cv2.threshold(r, 160, 255, cv2.THRESH_BINARY)
    _, g_t = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)
    _, b_t = cv2.threshold(b, 150, 255, cv2.THRESH_BINARY)
    variants.append(("red_ch",      cv2.bitwise_not(r_t)))
    variants.append(("green_ch",    cv2.bitwise_not(g_t)))
    variants.append(("blue_ch",     cv2.bitwise_not(b_t)))
    variants.append(("blue_ch_inv", b_t))

    # We apply standard grayscale variants as general-purpose fallbacks
    # that work well for dark text on light backgrounds.
    gray_up = upscale(gray)
    variants.append(("gray",     gray_up))
    variants.append(("gray_inv", cv2.bitwise_not(gray_up)))

    # CLAHE improves local contrast which helps with uneven lighting conditions.
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    ce    = clahe.apply(gray_up)
    variants.append(("clahe",     ce))
    variants.append(("clahe_inv", cv2.bitwise_not(ce)))

    # Otsu's thresholding finds the optimal global threshold automatically
    # and works well when there is a clear separation between text and background.
    _, otsu = cv2.threshold(gray_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu",     otsu))
    variants.append(("otsu_inv", cv2.bitwise_not(otsu)))

    # Adaptive thresholding computes a local threshold for each region,
    # which handles images where background brightness varies across the image.
    adp = cv2.adaptiveThreshold(
        gray_up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
    )
    variants.append(("adaptive",     adp))
    variants.append(("adaptive_inv", cv2.bitwise_not(adp)))

    # We apply a sharpening kernel to make blurry text edges crisper.
    kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray_up, -1, kernel)
    variants.append(("sharp",     sharpened))
    variants.append(("sharp_inv", cv2.bitwise_not(sharpened)))

    # Contrast stretching normalizes the pixel range to use the full 0-255 range,
    # which helps when the image has a narrow range of pixel values.
    norm = cv2.normalize(gray_up, None, 0, 255, cv2.NORM_MINMAX)
    variants.append(("contrast",     norm))
    variants.append(("contrast_inv", cv2.bitwise_not(norm)))

    return variants


def extract_text(image_path):
    # We start with the original PIL image as our initial baseline.
    pil_orig  = Image.open(image_path).convert("RGB")
    best_text = run_tesseract(pil_orig)

    # We try a contrast, sharpness and brightness enhanced version with PIL
    # because these adjustments often make text significantly more readable.
    try:
        enhanced = ImageEnhance.Contrast(pil_orig).enhance(2.5)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.1)
        t = run_tesseract(enhanced)
        if len(t.split()) > len(best_text.split()):
            best_text = t
    except Exception:
        pass

    # We try a median-filtered version which removes noise while
    # preserving edges, helping Tesseract read noisy or compressed images.
    try:
        filtered = pil_orig.filter(ImageFilter.MedianFilter(size=3))
        t = run_tesseract(filtered)
        if len(t.split()) > len(best_text.split()):
            best_text = t
    except Exception:
        pass

    # We try all OpenCV preprocessed variants and keep the best result.
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
                # We stop early once we have a solid result to save time.
                if len(best_text.split()) >= 20:
                    break
            except Exception:
                continue

    except Exception:
        pass

    return best_text


def has_text(text, min_words=3):
    # We consider text valid if it contains at least min_words words.
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