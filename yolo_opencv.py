
#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

#open output text file
number_cars = open("./caroutput/number_cars.txt", "r+")


space_available = [True, True, True, True, True]

#this is the formatting for the specfic image and car parking space

space_min_h = 2100 
space_max_h = 2900

space_1_min_x = 70
space_1_max_x = 750

space_4_min_x = 700
space_4_max_x = 1300

space_2_min_x = 1450
space_2_max_x = 2020

space_3_min_x = 2040
space_3_max_x = 2790

space_5_min_x = 2800
space_5_max_x = 2900


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



def image_output(img, x, y, x_plus_w, y_plus_h, space):
    
    crop_img = img[y:y_plus_h, x:x_plus_w]

    cv2.imwrite("./caroutput/car_space_%s.jpg" %(space), crop_img)




def available_space( img, x, y, x_plus_w, y_plus_h):

    object_location_x = (x_plus_w + x)/2
    object_location_y = (y_plus_h + y)/2
    
    if object_location_y > space_min_h and object_location_y < space_max_h:
    
        if object_location_x > space_1_min_x and object_location_x < space_1_max_x:
            space = "1"
            image_output(img, x, y, x_plus_w, y_plus_h, space)
            space_available[0]= False
            return space_available
            
        elif object_location_x > space_2_min_x and object_location_x < space_2_max_x:
            space = "2"
            image_output(img, x, y, x_plus_w, y_plus_h, space)
            space_available[1]= False
            return space_available
            
        elif object_location_x > space_3_min_x and object_location_x < space_3_max_x:
            space = "3"
            image_output(img, x, y, x_plus_w, y_plus_h, space)
            space_available[2]= False
            return space_available
            
        elif object_location_x > space_4_min_x and object_location_x < space_4_max_x:
               
            space = "4"
            image_output(img, x, y, x_plus_w, y_plus_h, space)            
            space_available[3]= False
            return space_available
            
        elif object_location_x > space_5_min_x and object_location_x < space_5_max_x:
            space = "5"
            image_output(img, x, y, x_plus_w, y_plus_h, space)
            space_available[4]= False
            return space_available
        
        return space_available
    
image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.1
nms_threshold = 0.4



for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

car_counter = 0

for i in indices:
    i = i[0]
    if class_ids[i]==2 :
       car_counter = car_counter +1 
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    space_available = available_space( image, round(x), round(y), round(x+w), round(y+h))
    
no_parking = 5
no_free = no_parking - car_counter

cv2.imwrite("./caroutput/object-detection.jpg", image)

#writing the informmation to a .txt file 
#number of cars
#number of free spaces
number_cars = open("./caroutput/number_cars.txt", "r+")
number_cars_string = "number of cars: " + str(car_counter)
number_cars.write(number_cars_string)
number_cars_string = "\nnumber of free spaces: " + str(no_free)
number_cars.write(number_cars_string)
number_cars.write(str(space_available))
number_cars.close()

print("Complete")

cv2.destroyAllWindows()
