"""Display extraction boxes numbered in reading order."""
import cv2
import matplotlib.pyplot as plt


def show_extracted_words(image, boxes, detected_count):
    #put the bounding boxes on the original image
    visualization_image = image.copy()
    #go through each box and 
    for idx, (bx, by, bw, bh) in enumerate(boxes):
         #red boxes for extracted regions 
         cv2.rectangle(visualization_image, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)
         #add text number to the box
         cv2.putText(visualization_image, str(idx+1), (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    plt.figure(figsize=(15, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    # Use visualization_image to show red extraction boxes IN SORTED ORDER
    plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))

    plt.title(f"etected ({detected_count}) / extracted & sorted ({len(boxes)}) text regions")
    plt.axis("off") 

    plt.tight_layout() 
    plt.show()
