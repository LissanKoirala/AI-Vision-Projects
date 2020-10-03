from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np


# Classifier for detecting whether there is a face in the frame or not!
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Classifier for detecting emotions on the frame
classifier =load_model('./Emotion_Detection.h5')

# Labels for the emotions
class_labels = ['Angry','Happy','Neutral','Sad','Surprise']


# Starting the capture from the webcam! It can be form video source if wanted! Just changing the "(0)" to the 
# video source, example link or a file source
cap = cv2.VideoCapture(0)



while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []

    # Converting the color of the frame to gray as it is easy to process the the normal color
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Then classflying if there is face in the frame
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction, then lookup the class


        	# Now here we start making predictions if there is any emotion captured or not!

            preds = classifier.predict(roi)[0]

            # Printing the predictions on the console
            label = class_labels[preds.argmax()]

            print(label)
            label_position = (x,y)


            if label == "Happy":
                color = (0,255,0)

            if label == "Angry":
                color = (0,0,255)

            if label == "Neutral":
                color = (255,0,)

            if label == "Surprise":
                color = (0,255,255)

            if label == "Sad":
                color = (255,255,0)


            # Writing this on the frame for the user to view!
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, color ,3)
        
        # It can happen that the face is not there in the frame of the video capture for that reason,
        # We need to include this to inform the user.
        else:
        	
            cv2.putText(frame, 'No Face Found', (20,60), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
        print("\n\n")
    

    # Displaying the frame on the screen, with the label of the emotion in it!
    cv2.imshow('Emotion Detector', frame)

    # Quit the program if the user presses q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the webcam and destroying all the windows!
cap.release()
cv2.destroyAllWindows()


























