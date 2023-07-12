# import numpy as np
# import cv2
# import os

# ##knn code

# def distance(v1,v2):
#     #eucledian distance
#     return np.sqrt(((v1-v2)**2).sum())

# def knn(train, test,k=5):
#     dist = []

#     for i in range(train.shape[0]):
#         #get the vector and label 
#         ix = train[i, :-1]
#         iy = train[i,-1]

#         #compute the distance from  test point
#         d = distance(test,ix)
#         dist.append([d,iy])
    
#     #sort based on distance and get top k
#     dk = sorted(dist, key=lambda x: x[0])[:k]

#     #retrieve only the label
#     labels = np.array(dk)[:, -1]

#     #ger frequencies of each labels
#     output = np.unique(labels, return_counts = True)

#     #find max frequency and corresponding labels
#     index = np.argmax(output[1])
    
#     return output[0][index]

#     #################################################################################3

#     # Read a video stream using opencv 

#     #init camera
# cap = cv2.VideoCapture(0)

# #face detection and extraction
# face_cascade = cv2.CascadeClassifier("/Users/kunalkumar/Desktop/project1/Face-Recognition/haarcascade_frontalface_alt.xml")

# skip = 0

# dataset_path = '/Users/kunalkumar/Desktop/project1/images of face'
# # face_data = []
# face_data = np.array([])
# labels = []
# class_id = 0 #labels for the given file
# names = {} #mapping between id and name

# #data preparation

# for fx in os.listdir(dataset_path):
#     if fx.endswith('.npy'):
#         names[class_id] = fx[:-4] 
#     #crete a mapping between class_id and name
#         print("Loaded" +fx)
        
#         data_item = np.load(dataset_path + fx)
#         face_data.append(data_item)

#         #create labels for the class
#         target = class_id*np.ones((data_item.shape[0],))
#         class_id += 1
#         labels.append(target)

# print(len((face_data)))

# # face_dataset = np.concatenate(face_data,axis = 0)
# face_dataset = np.concatenate(face_data,axis=0)
# # face_labels = np.concatenate(labels,axis = 0).reshape((-1,1))
# face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

# print(face_dataset.shape)
# print(face_labels.shape)

# trainset = np.concatenate((face_dataset,face_labels),axis = 1)
# print(trainset.shape)



# ### testing ###

# while True:
#     ret,frame = cap.read()
#     if ret == False:
#         continue

#     faces = face_cascade.detectMultiScale(frame,1.3,5)

#     for face in faces:
#         x,y,w,h = face

#         #get the face ROI 
#         offset = 10
#         face_section = frame[y-offset:y+h+offset , x-offset:x+w+offset]
#         face_section = cv2.resize(face_section,(100,100))

#         #predict label (out)
#         out = knn(trainset,face_section.flatten())

#         #display on the screen the name and rectangle around it
#         pred_name = names[int(out)]
#         cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2,cv2.LINE_AA)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)


#     cv2.imshow("Faces",frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import numpy as np
import cv2
import os

## knn code

def distance(v1, v2):
    # eucledian distance
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # get the vector and label 
        ix = train[i, :-1]
        iy = train[i, -1]

        # compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])

    # sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]

    # retrieve only the label
    labels = np.array(dk)[:, -1]

    # get frequencies of each label
    output = np.unique(labels, return_counts=True)

    # find max frequency and corresponding label
    index = np.argmax(output[1])

    return output[0][index]

# Read a video stream using opencv 
cap = cv2.VideoCapture(0)

# face detection and extraction
face_cascade = cv2.CascadeClassifier("/Users/kunalkumar/Desktop/project1/Face-Recognition/haarcascade_frontalface_alt.xml")

skip = 0

dataset_path = "/Users/kunalkumar/Desktop/project1/images of face"
face_data = []
labels = []
class_id = 0  # labels for the given file
names = {}  # mapping between id and name

# data preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4] 
        # create a mapping between class_id and name
        print("Loaded " + fx)
        
        data_item = np.load(os.path.join(dataset_path, fx))
        if data_item.size == 0:
            continue

        face_data.append(data_item)

        # create labels for the class
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

if len(face_data) == 0:
    print("No face data found in the specified directory.")
    exit()

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

### testing ###

while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for face in faces:
        x, y, w, h = face

        # get the face ROI 
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        # predict label (out)
        out = knn(trainset, face_section.flatten())

        # display on the screen the name and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow("Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

