

import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import cv2
import keras
from keras.layers import StringLookup
from keras import ops
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from pathlib import Path
import pandas as pd

#///////////////IMAGE PREPROCESSING TO MATCH AI TRAINING///////////////////////
image_width = 128
image_height = 32

def distortion_free_resize(image, img_size=(image_width, image_height)):
    w, h = img_size
    #https://www.tensorflow.org/api_docs/python/tf/image/resize
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    #check tha amount of padding needed to be done.
    pad_height = h - ops.shape(image)[0]
    pad_width = w - ops.shape(image)[1]

    #add padding to both sides
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )
    #we transpose the image because handwriting is more of an up down
    image = ops.transpose(image, (1, 0, 2))
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image, img_size=(image_width, image_height)):
    #convert to tensor for CNN and other tensorflwo functions
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = distortion_free_resize(image, img_size)
    #scale
    image = ops.cast(image, tf.float32) / 255.0
    return image

#///////////////////MAKE THE BOUNDING BOXES////////////////////////////////////
"""
resources used:
    https://medium.com/%40maniksingh256/word-segmentation-from-handwritten-paragraphs-using-opencv-tools-6ba05dee13b8
    https://www.geeksforgeeks.org/python-opencv-morphological-operations/
    https://www.indusmic.com/post/digital-image-processing-using-opencv
    https://www.geeksforgeeks.org/image-segmentation-using-morphological-operation/
    https://www.mo4tech.com/opencv-digital-image-processing.html
"""
class RegionFocusedTextDetector:
    #parameters fine tuned to make good boxes
    def __init__(self):
        #parameters for text detection
        self.block_size = 15    #parameter for adaptive thresholding
        self.c_value = 9        #parameter for adaptive thresholding
        self.min_text_height = 5  #lowered to catch smaller words
        self.min_text_width = 8   #lowered to catch smaller words
        self.min_text_area = 40   #lowered to catch smaller words
        self.max_text_area = 15000
        self.horizontal_kernel_width = 15  # Reduced for better small word detection

    def detect_text_regions(self, image_path, roi=None):
        """
        Detect text regions in an image

        Parameters:
        image_path (str): Path to the image
        roi (tuple): Region of interest as (x, y, width, height), None for entire image

        Returns:
        tuple: (result image with boxes, list of contours)
        """
        #read the image
        img = cv2.imread(image_path)
        #this way it doesn't get distracted
        #img = img[670:2800, 100:2500]
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        #copy of the image for future visual
        result = img.copy()

        """
        ROI or region of interest
        I tried to get this to work, but messes with the bounding boxes for some reason
        """
        if roi is not None:
            x, y, w, h = roi
            #ensure ROI is within image bounds
            x = max(0, min(x, img.shape[1] - 1))
            y = max(0, min(y, img.shape[0] - 1))
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)

            #extract ROI
            img_roi = img[y:y+h, x:x+w]

            #process only the ROI
            processed_roi, contours = self._process_image_region(img_roi)

            #djust contour coordinates to the original image space
            adjusted_contours = []
            for contour in contours:
                contour_shifted = contour.copy()
                contour_shifted[:, :, 0] += x
                contour_shifted[:, :, 1] += y
                adjusted_contours.append(contour_shifted)

            #draw rectangles
            for contour in adjusted_contours:
                bx, by, bw, bh = cv2.boundingRect(contour)
                padding = 2
                bx = max(0, bx - padding)
                by = max(0, by - padding)
                bw = min(img.shape[1] - bx, bw + 2*padding)
                bh = min(img.shape[0] - by, bh + 2*padding)
                cv2.rectangle(result, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

            #draw the ROI boundary in blue
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

            return result, adjusted_contours
        else:
            #process the entire image
            return self._process_full_image(img)


    def _process_image_region(self, img_region):
        """Process a specific region of an image"""
        #convert to grayscale
        gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)

        #apply Gaussian blur to reduce noise
        #we agetting a lot of dots
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        #apply adaptive thresholding to get binary image
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.block_size, self.c_value
        )

        #create two different horizontal kernels - one for small words, one for longer words
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.horizontal_kernel_width, 1))
        large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))

        #connect characters horizontally (two passes with different scales)
        connected_small = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, small_kernel)
        connected_large = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, large_kernel)

        #combine the results
        connected = cv2.bitwise_or(connected_small, connected_large)

        #create vertical kernel for separating text lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

        #separate text lines
        separated = cv2.morphologyEx(connected, cv2.MORPH_OPEN, vertical_kernel)

        #find contours
        contours, _ = cv2.findContours(
            separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        #extra processing for small words: find contours directly on binary image
        small_contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        """
        I was having trouble getting smaller words, and I could not fine tune it
        It would get a thousand dots, not enough of the words to make it work,
        or it would split the larger words.
        
        I included two for loops with different regions. It still MAKES the boxes
        spliting words, but they are deleted by 
        
        I still have the same issue with random dots but that is cleaned on the backend 
        """
        #filter contours for standard text lines
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / float(h) if h > 0 else 0

            #standard text line criteria
            if (h >= self.min_text_height and
                w >= self.min_text_width and
                self.min_text_area <= area <= self.max_text_area and
                aspect_ratio > 1.2):  # Relaxed aspect ratio for shorter words
                valid_contours.append(contour)

        #filter small contours that might be individual words
        for contour in small_contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            #criteria for small words
            if (4 <= h <= 12 and  # Height range for small words
                5 <= w <= 40 and  # Width range for small words
                30 <= area <= 200 and  # Area range for small words
                not self._is_contained_in_contours(contour, valid_contours)):
                valid_contours.append(contour)

        #sort contours by y-position (top to bottom)
        valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[1])

        result = img_region.copy()
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            padding = 2
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_region.shape[1] - x, w + 2*padding)
            h = min(img_region.shape[0] - y, h + 2*padding)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return result, valid_contours

    def _is_contained_in_contours(self, contour, contour_list):
        #check if a contour is contained within any contour in the list
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2

        for other in contour_list:
            if other is contour:
                continue

            other_x, other_y, other_w, other_h = cv2.boundingRect(other)

            #if contors overlaps with other contor
            if (other_x <= center_x <= other_x + other_w and
                other_y <= center_y <= other_y + other_h):
                return True

        return False

    def _process_full_image(self, img):
        #process the entire image
        result, contours = self._process_image_region(img)
        return result, contours

    def visualize_sample(self, image_path, roi=None):
        #visualize for debugging
        original = cv2.imread(image_path)
        if original is None:
            print(f"Could not read image at {image_path}")
            return

        #proccess the image above
        result, contours = self.detect_text_regions(image_path, roi)

        #generate intermediate visualizations
        if roi is not None:
            x, y, w, h = roi
            # Ensure ROI is within image bounds
            x = max(0, min(x, original.shape[1] - 1))
            y = max(0, min(y, original.shape[0] - 1))
            w = min(w, original.shape[1] - x)
            h = min(h, original.shape[0] - y)

            #extract ROI for visualization
            img_roi = original[y:y+h, x:x+w]
            gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.block_size, self.c_value
        )

        #create horizontal kernels
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.horizontal_kernel_width, 1))
        large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))

        #connect characters horizontally
        connected_small = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, small_kernel)
        connected_large = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, large_kernel)
        connected = cv2.bitwise_or(connected_small, connected_large)

        #convert from BGR to RGB for matplotlib
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        #create visualization with intermediate steps
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(original_rgb)
        plt.title("Original Image" + (" with ROI" if roi else ""))
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(binary, cmap='gray')
        plt.title("Binary Threshold")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(connected_small, cmap='gray')
        plt.title("Small Words Connected")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(connected_large, cmap='gray')
        plt.title("Large Words Connected")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(connected, cmap='gray')
        plt.title("Combined Connected")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(result_rgb)
        plt.title(f"Detected Text Regions ({len(contours)} found)")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig("detection_visualization.png")
        plt.show()

        print(f"Found {len(contours)} text regions")
        return result, contours

    def process_dataset(self, dataset_dir, output_dir, roi=None, max_samples=None):
        """Process multiple images from the IAM dataset"""
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        #find all image files
        image_extensions = ['.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(dataset_path.glob(f"**/*{ext}")))

        if max_samples is not None:
            image_files = image_files[:max_samples]

        processed_count = 0
        for img_path in image_files:
            try:
                #create relative output path
                rel_path = img_path.relative_to(dataset_path)
                out_file = output_path / rel_path
                out_file.parent.mkdir(exist_ok=True, parents=True)

                #process image with optional ROI
                result_img, _ = self.detect_text_regions(str(img_path), roi)

                #save result
                cv2.imwrite(str(out_file), result_img)

                processed_count += 1

                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{len(image_files)} images")

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

        print(f"Processing complete. Processed {processed_count} images.")

# --- 3. Modified segment_words Function ---
#https://medium.com/%40maniksingh256/word-segmentation-from-handwritten-paragraphs-using-opencv-tools-6ba05dee13b8
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

#///////////////////////////////////GUESS WORDS BASED ON MODEL/////////////////////

#load trained model
model = load_model("pred.keras")

#decode predictions
characters = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = ops.nn.ctc_decode(pred, sequence_lengths=input_len)[0][0][
        :, :21
    ]
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = (
            tf.strings.reduce_join(num_to_char(res))
            .numpy()
            .decode("utf-8")
            .replace("[UNK]", "")
        )
        output_text.append(res)
    return output_text
#///////validation/////////////////////////
def score(pred, target):
    None

#///////////////////MAIN/////////////////////////////////////////////////////////
#load information from csv
forms_data = pd.read_csv('forms_data.csv')
host="C:/Users/mikey/OneDrive/Documents/USCB/Data Mining/Project/Forms/"

results_forms = []
results_pred =[]

results_sentences = []
results_cleaned = []
for index, row in forms_data.iterrows():
    form_id = row['Form ID']
    form_info = row['Form Data']
    
    #It was made with "sample path", and I am not chaning it
    sample_path = host + form_id + ".png"
    
    #-------using text detector------------
    detector = RegionFocusedTextDetector()
    original_image = cv2.imread(sample_path)
    """
    here we have to get a list of contors and sort them
    
    they has typically been challenging.  Here we try to group them.
    """
    if original_image is None:
        print(f"Error: Could not read image at {sample_path}")
    else:
        #process the image and 
        image_with_boxes, detected_contours = detector.detect_text_regions(sample_path)
        img_to_extract_from = original_image # Use the whole image for extraction
    
        print(f"Detected {len(detected_contours)} potential text regions.")
    
        #get the bounding boxes
        if detected_contours:
            # (x, y, w, h)
            bounding_boxes = [cv2.boundingRect(c) for c in detected_contours]
            contours_with_boxes = list(zip(detected_contours, bounding_boxes))
        
            #sort them into groups based on y axis
            contours_with_boxes.sort(key=lambda item: item[1][1])
            """
            We are basically going to go through ths like a four year old.
            
            We made some lines earlier and we are now sorting through them
            """
            lines = []
            if contours_with_boxes:
                current_line = [contours_with_boxes[0]]
                #use center y for comparison, and allow some tolerance based on height
                last_y_center = contours_with_boxes[0][1][1] + contours_with_boxes[0][1][3] / 2
                #allow overlap up to 70% of height
                y_tolerance = contours_with_boxes[0][1][3] * 0.7 
        
                for item in contours_with_boxes[1:]:
                    contour, box = item
                    current_y_center = box[1] + box[3] / 2
                    #check if current y centerpoint is close to the last one
                    if abs(current_y_center - last_y_center) < y_tolerance:
                         current_line.append(item)
                         # Update line's representative y? Maybe not needed if primarily sorted by y
                    else:
                        #start a new line
                        lines.append(current_line)
                        current_line = [item]
                        last_y_center = current_y_center
                        y_tolerance = box[3] * 0.7 #update tolerance based on new line start
                lines.append(current_line) #add the last line
        
            #sort by x within each group of borders(lines)
            for line in lines:
                line.sort(key=lambda item: item[1][0]) # Sort by x
            """
            combine the lines back into a single list
            grouped by line y(height) and then wthin(already sorted by line y, then within-line x)
            
            
            """
            #combine the lines back into a single list
            #grouped by line y(height) and then wthin(already sorted by line y, then within-line x)
            sorted_contours_with_boxes = [item for line in lines for item in line]
            sorted_contours = [item[0] for item in sorted_contours_with_boxes]
            sorted_boxes = [item[1] for item in sorted_contours_with_boxes] # Keep boxes if useful
        
            print(f"Contours sorted into {len(lines)} lines.")
    
        else:
            print("No contours detected to sort.")
            sorted_contours = []
            sorted_boxes = []
    
    
        # now that we have the contors we 
        word_images = []
        # To store final coordinates used for extraction, IN SORTED ORDER
        processed_boxes_final = [] 
    
        # Convert the image used for extraction to grayscale
        gray_image_for_extraction = cv2.cvtColor(img_to_extract_from, cv2.COLOR_BGR2GRAY)
    
        #go through all of our bounding-boxes
        for contour in sorted_contours: 
            #get bounding box coordinates within the image
            x, y, w, h = cv2.boundingRect(contour)
    
            #get the grayscale of the original immage and add padding
            #add padding similar to detector's internal drawing logic
            padding = 2
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)         #ensure coordinates are within the bounds of the *extraction* image
            w_pad = min(gray_image_for_extraction.shape[1] - x_pad, w + 2*padding)
            h_pad = min(gray_image_for_extraction.shape[0] - y_pad, h + 2*padding)
    
            #crop the word region from the grayscale image
            cropped_word = gray_image_for_extraction[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
    
            #check if the cropped area is valid aka not zero
            if cropped_word.shape[0] > 0 and cropped_word.shape[1] > 0:
                #add channel dimension for image preprocessing function
                cropped_word = np.expand_dims(cropped_word, axis=-1)
                #preprocess the image like in keras
                preprocessed_word = preprocess_image(cropped_word)
                word_images.append(preprocessed_word)
                #get the bounding boxes info
                processed_boxes_final.append((x_pad, y_pad, w_pad, h_pad))
            else:
                print(f"Warning: Skipped invalid contour (originally at x={x}, y={y}, w={w}, h={h}) after padding and sorting.")
    
    
        if word_images:
            print(f"successfully preprocessed {len(word_images)} word images for Keras model.")
            # Prepare the word images for the model
            word_images_array = np.array(word_images, dtype=np.float32)
    
            #make predictions
            predictions = model.predict(word_images_array)
            predicted_texts = decode_batch_predictions(predictions)
    
            #print predictions (should now be in reading order)
            print("\n--- Predicted Text (Sorted Order) ---")
            for i, text in enumerate(predicted_texts):
                # Use the processed_boxes_final which corresponds to the sorted order
                print(f"word {i + 1} (box: {processed_boxes_final[i]}): {text}")
            print("-------------------------------------\n")
    
    #/////////////////////VISUALIZE///////////////////////////////////////////////
            #put the bounding boxes on the original image
            visualization_image = original_image.copy()
            #go through each box and 
            for idx, (bx, by, bw, bh) in enumerate(processed_boxes_final):
                 #red boxes for extracted regions 
                 cv2.rectangle(visualization_image, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)
                 #add text number to the box
                 cv2.putText(visualization_image, str(idx+1), (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    
            plt.figure(figsize=(15, 10))
    
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis("off")
    
            plt.subplot(1, 2, 2)
            # Use visualization_image to show red extraction boxes IN SORTED ORDER
            plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
            
            plt.title(f"etected ({len(detected_contours)}) / extracted & sorted ({len(word_images)}) text regions")
            plt.axis("off") 
    
            plt.tight_layout() 
            plt.show() 
    
        else:
            #detector is currently named "RegionFocusedTextDetector"
            print("Error: no valid word regions were extracted using WordDector.")
    
    
    #///////////////////Text to sentece//////////////////////////////////////////////
        #remove the first 3 (sentence, database, and formID)
        del predicted_texts[0:3]
        #remove the last one which is "Name:"
        del predicted_texts[-1]
        #we remove all the periods, dashes and, quotes in single form
        predicted_texts = [item for item in predicted_texts if item != '.']
        predicted_texts = [item for item in predicted_texts if item != '"']
        predicted_texts = [item for item in predicted_texts if item != ',']
        
        #cat all values with a space
        sentences = ' '.join(predicted_texts)
        
        #send this off to a third party AI tool or Textblob
        """
        I would like to use an AI tool, which got 100% accuracy on my test samples,
        but that isn't scalable and it costs money.
        
        I decided to use textblob instead.
        """
        
    import re
    from textblob import TextBlob
    #from language_tool_python import LanguageTool
    
    def clean_text(text):
        #basic cleaning with regex
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        #spell correction with TextBlob
        text = str(TextBlob(text).correct())
        
        """
        #grammar correction with LanguageTool
        tool = LanguageTool('en-US')
        matches = tool.check(text)
        text = tool.correct(text)
        """
        
        return text
    
    #example usage
    
    cleaned = clean_text(sentences)
    print(sentences)
    print(cleaned)
    
    #just to plug it into chatgpt
    results_sentences.append(sentences)
    results_cleaned.append(cleaned)
    #compare cleaned to target
    def compare_strings(str1, str2):
        #make two sets from the words
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        # Find the common words
        common_words = words1.intersection(words2)
        
        # Return the count of common words
        return len(common_words)
    ratio = compare_strings(cleaned, form_info)/len(form_info)
    results_forms.append(ratio)
    print(f'Percent of words from the form in common{ratio}')
    ratio = ratio*len(form_info)
    ratio =ratio/len(cleaned)
    results_pred.append(ratio)
    print(f'Percent of words from the predictions in common{ratio}')
    
