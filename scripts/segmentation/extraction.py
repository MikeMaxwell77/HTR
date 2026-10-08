"""Reading-order sorting and padded word extraction."""
import cv2
import numpy as np
from .detection import RegionFocusedTextDetector
from .preprocessing import preprocess_image

def segment_words(image_path):
    detector = RegionFocusedTextDetector()
    img = cv2.imread(image_path)
    img = img[650:2800, :]
    image_copy = img.copy()

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu1 = cv2.threshold(grey, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]

    horz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 1))
    lines = cv2.morphologyEx(otsu1, cv2.MORPH_OPEN, horz_kernel, iterations=2) # Corrected line
    cnts = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (255, 255, 255), 2)

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu2 = cv2.threshold(grey, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 9))
    dilation = cv2.dilate(otsu2, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img_list = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        #how sensitive we want to be to abnormalities
        if w > 30 and h > 20:
            cropped_word = otsu2[y:y + h, x:x + w]
            resized_word = cv2.resize(cropped_word, (128, 32))  #initial resize
            resized_word = np.expand_dims(resized_word, axis=-1) #add channel dimension
            preprocessed_word = preprocess_image(resized_word) #apply full preprocessing
            img_list.append(preprocessed_word)
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_list, image_copy

def sort_contours(contours):
    """Group by line with v5's 70% height tolerance, then sort left to right."""
    items = sorted(((c, cv2.boundingRect(c)) for c in contours), key=lambda item: item[1][1])
    if not items:
        return [], 0
    lines = []
    current_line = [items[0]]
    last_y_center = items[0][1][1] + items[0][1][3] / 2
    y_tolerance = items[0][1][3] * 0.7
    for item in items[1:]:
        box = item[1]
        current_y_center = box[1] + box[3] / 2
        if abs(current_y_center - last_y_center) < y_tolerance:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
            last_y_center = current_y_center
            y_tolerance = box[3] * 0.7
    lines.append(current_line)
    for line in lines:
        line.sort(key=lambda item: item[1][0])
    return [item[0] for line in lines for item in line], len(lines)


def extract_words(image, contours):
    """Return preprocessed crops and their corresponding padded boxes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    words, boxes = [], []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_pad, y_pad = max(0, x - 2), max(0, y - 2)
        w_pad = min(gray.shape[1] - x_pad, w + 4)
        h_pad = min(gray.shape[0] - y_pad, h + 4)
        crop = gray[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            words.append(preprocess_image(np.expand_dims(crop, axis=-1)))
            boxes.append((x_pad, y_pad, w_pad, h_pad))
        else:
            print(f"Warning: Skipped invalid contour at x={x}, y={y}, w={w}, h={h}.")
    return words, boxes
