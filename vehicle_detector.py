import cv2


# img = cv2.imread(r"test_images\test1.jpg")


# Get names of the classes from 'coco.names' file
class_names = []
with open("coco.names", "rt") as file:
    class_names = file.read().rstrip("\n").split("\n")

# List that contains indices of the desired classes - 'bicycle', 'car', 'motorcycle', 'bus'
desired_classes = [2, 3, 4, 6]

config_path = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weights_path = "frozen_inference_graph.pb"

# Load model configuration and weights file and create the network
net = cv2.dnn_DetectionModel(weights_path, config_path)

net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def vehicle_detector(img):
    # Feed the image to the model and get information of class ids, confidence values and bounding boxes of detected objects
    # Consider good detection if above 50% confidence (confThreshold=0.5)
    # Apply non-maximum suppression to eliminate overlapping boxes (nmsThreshold=0.3)
    class_ids, confidence_values, bounding_boxes = net.detect(
        img, confThreshold=0.5, nmsThreshold=0.3)
    # print(class_ids)
    # print(confidence_values)
    # print(bounding_boxes)

    # Draw bounding box for each detected object with label and confidence value
    for class_id, confidence, bbox in zip(class_ids, confidence_values, bounding_boxes):
        # Check if the detected class belongs to the desired classes
        if class_id in desired_classes:
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f'{class_names[class_id-1].upper()} {int(confidence*100)}%',
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    return img


# # Test code
# # Feed the image to the model and get information of class ids, confidence values and bounding boxes of detected objects
# # Consider good detection if above 50% confidence (confThreshold=0.5)
# # Apply non-maximum suppression to eliminate overlapping boxes (nmsThreshold=0.3)
# class_ids, confidence_values, bounding_boxes = net.detect(img, confThreshold=0.5, nmsThreshold=0.3)
# # print(class_ids)
# # print(confidence_values)
# # print(bounding_boxes)

# # Draw bounding box for each detected object with label and confidence value
# for class_id, confidence, bbox in zip(class_ids, confidence_values, bounding_boxes):
#     # Check if the detected class belongs to the desired classes
#     if class_id in desired_classes:
#         x, y, w, h = bbox
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(img, f'{class_names[class_id-1].upper()} {int(confidence*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# cv2.imshow("Image", img)
# cv2.waitKey(0)
