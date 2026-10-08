"""OpenCV text-region detection and detector debug visualizations."""
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from .config import PROJECT_ROOT

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
        plt.savefig(str(PROJECT_ROOT / "outputs/detection_visualization.png"))
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
