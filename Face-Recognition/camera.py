# import cv2
# import numpy as np

# #init camera
# cap = cv2.VideoCapture(0)

# #face detection
# face_cascade = cv2.CascadeClassifier("/Users/kunalkumar/Desktop/project1/Face-Recognition/haarcascade_frontalface_alt.xml")
# skip = 0
# face_data = []
# dataset_path = '/Users/kunalkumar/Desktop/project1/images of face'
# file_name = input("Enter the name of the person : ")
# while True:

#     ret,frame = cap.read()

#     if ret == False:
#         continue

#     gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    

 

#     faces = face_cascade.detectMultiScale(frame,1.3,5)      #w = 1.3 and h = 5
#     faces = sorted(faces, key = lambda f:f[2] * f[3])

#     #boxes and pick the largest hace 
#     for face in faces[-1:]:
#         x,y,w,h = face
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    
#         #extract and crop out the required face c
#         offset = 20          #padding
#         face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
#         face_section = cv2.resize(face_section,(100,100))

#         skip += 1
#         if skip % 10 == 0:
#             face_data.append(face_section)
#             print(len(face_data))


#     cv2.imshow("Frame",frame)

#     cv2.imshow("Face Section", face_section)

#     #store every 10th capture (face)

# #    if skip % 10 == 0:
# #       #store the 10th face here later on
# #       pass


#     key_pressed = cv2.waitKey(1) & 0xFF
#     if key_pressed == ord('q'):
#         break

# #convert our face list arrray into a numpy arrray
# face_data = np.asarray(face_data)
# face_data = face_data.reshape(face_data.shape[0],-1)
# print(face_data.shape)

# #save this data into file system
# np.save(dataset_path + file_name + '.npy',face_data)
# print("Data successfully save at" + dataset_path + file_name + '.npy')

# cap.release()
# cv2.destroyAllWindows() 
import cv2
import numpy as np

#init camera
cap = cv2.VideoCapture(0)

#face detection
face_cascade = cv2.CascadeClassifier("/Users/kunalkumar/Desktop/project1/Face-Recognition/haarcascade_frontalface_alt.xml")
skip = 0
face_data = []
dataset_path = '/Users/kunalkumar/Desktop/project1/images of face'
file_name = input("Enter the name of the person: ")
face_section = None

while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)  # w = 1.3 and h = 5
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    #boxes and pick the largest face 
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
        #extract and crop out the required face
        offset = 20  # padding
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        skip += 1
        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow("Frame", frame)

    if face_section is not None:
        cv2.imshow("Face Section", face_section)

    #store every 10th capture (face)

#    if skip % 10 == 0:
#       #store the 10th face here later on
#       pass

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

#convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape(face_data.shape[0], -1)
print(face_data.shape)

#save this data into file system
np.save(dataset_path + '/' + file_name + '.npy', face_data)
print("Data successfully saved at " + dataset_path + '/' + file_name + '.npy')

cap.release()
cv2.destroyAllWindows()
