from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import time
import pyaudio  
import wave

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

ear_thresh = 0.21
frames_thresh = 48
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
count=0
chunk=1024

while True:
    start_time = time.time()
    f = wave.open(r"alarm.wav","rb")
    p = pyaudio.PyAudio()            
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
    channels = f.getnchannels(),  
    rate = f.getframerate(),  
    output = True)
    ret, frame=cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect(gray, 0)
    for face in faces:
        shape = predict(gray, face)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        #leftEyeHull = cv2.convexHull(leftEye)
        #rightEyeHull = cv2.convexHull(rightEye)
        #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < ear_thresh:
            count += 1
            print (count)
            cv2.imwrite("Drowsy_Dataset/Drowsy(%f).jpg"%ear,frame)
            if count >= frames_thresh:                
                cv2.putText(frame, "DROWSINESS ALERT!", (125, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                data = f.readframes(chunk)                   
                stream.write(data)            
        else:
            count = 0
            stream.stop_stream()  
            stream.close()           
            cv2.imwrite("Non_Drowsy_Dataset/Non_Drowsy(%f).jpg"%ear,frame)            

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0,0), 2)
            
    end_time = time.time() 
    fps = 1 / float(end_time - start_time)
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (30, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0,0), 2)                    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
        p.terminate()      
        break


