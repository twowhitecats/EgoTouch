import cv2
import time
import math
import signal
import threading
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import serial
import os
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
import pandas as pd






class landmarker_and_result():
   def __init__(self):
      self.result = mp.tasks.vision.HandLandmarkerResult
      self.landmarker = mp.tasks.vision.HandLandmarker
      self.createLandmarker()
   
   def createLandmarker(self):
      # callback function
      def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
         self.result = result

      # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
      options = mp.tasks.vision.HandLandmarkerOptions( 
         base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"), # path to model
         running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
         num_hands = 2, # track both hands
         min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
         min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
         min_tracking_confidence = 0.3, # lower than value to get predictions more often
         result_callback=update_result)
      
      # initialize landmarker
      self.landmarker = self.landmarker.create_from_options(options)
   
   def detect_async(self, frame):
      # convert np frame to mp image
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      # detect landmarks
      self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

   def close(self):
      # close landmarker
      self.landmarker.close()

def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   """Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
   try:
      if detection_result.hand_landmarks == []:
         return rgb_image
      else:
         hand_landmarks_list = detection_result.hand_landmarks
         annotated_image = np.copy(rgb_image)

         # Loop through the detected hands to visualize.
         for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
               landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            mp.solutions.drawing_utils.draw_landmarks(
               annotated_image,
               hand_landmarks_proto,
               mp.solutions.hands.HAND_CONNECTIONS,
               mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
               mp.solutions.drawing_styles.get_default_hand_connections_style())
            
         return annotated_image
   except:
      return rgb_image

def count_fingers_raised(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   """Iterate through each hand, checking if fingers (and thumb) are raised.
   Hand landmark enumeration (and weird naming convention) comes from
   https://developers.google.com/mediapipe/solutions/vision/hand_landmarker."""
   try:
      # Get Data
      hand_landmarks_list = detection_result.hand_landmarks

      # Code to count numbers of fingers raised will go here
      numRaised = 0
      thumb, ind, mid, ring, lit = 0, 0, 0, 0, 0

    #   raisedFin=[False, False, False, False, False]
      # for each hand...
      for idx in range(len(hand_landmarks_list)):
         # hand landmarks is a list of landmarks where each entry in the list has an x, y, and z in normalized image coordinates
         hand_landmarks = hand_landmarks_list[idx]
         # for each fingertip... (hand_landmarks 4, 8, 12, and 16)
         for i in range(8,21,4):
            # make sure finger is higher in image the 3 proceeding values (2 finger segments and knuckle)
            tip_y = hand_landmarks[i].y
            dip_y = hand_landmarks[i-1].y
            pip_y = hand_landmarks[i-2].y
            mcp_y = hand_landmarks[i-3].y
            if tip_y < min(dip_y,pip_y,mcp_y):
               numRaised += 1
                
               if i == 8 : ind = 1
               elif i == 12 : mid = 1
               elif i == 16 : ring = 1
               else : lit = 1
         # for the thumb
         # use direction vector from wrist to base of thumb to determine "raised"
         tip_x = hand_landmarks[4].x
         dip_x = hand_landmarks[3].x
         pip_x = hand_landmarks[2].x
         mcp_x = hand_landmarks[1].x
         palm_x = hand_landmarks[0].x
         if mcp_x > palm_x:
            if tip_x > max(dip_x,pip_x,mcp_x):
               numRaised += 1
               thumb = 1
         else:
            if tip_x < min(dip_x,pip_x,mcp_x):
               numRaised += 1
               thumb = 1

      # Code to display the number of fingers raised will go here
      annotated_image = np.copy(rgb_image)
      height, width, _ = annotated_image.shape
      text_x = int(hand_landmarks[0].x * width) - 100
      text_y = int(hand_landmarks[0].y * height) + 50
    #   cv2.putText(img = annotated_image, text = str(numRaised) + " Fingers Raised",
    #       org = (text_x, text_y), fontFace = cv2.FONT_HERSHEY_DUPLEX,
    #       fontScale = 1, color = (0,0,255), thickness = 2, lineType = cv2.LINE_4)
      cv2.putText(img = annotated_image, text = str(thumb) + str(ind) + str(mid) + str(ring) + str(lit),
          org = (text_x, text_y), fontFace = cv2.FONT_HERSHEY_DUPLEX,
          fontScale = 1, color = (0,0,255), thickness = 2, lineType = cv2.LINE_4)
      return annotated_image
   except:
      return rgb_image

def get_images_activated_index_fingers(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   try:
      # Get Data
      hand_landmarks_list = detection_result.hand_landmarks
      
      height, width = rgb_image.shape[:2]
      dst = np.array([[height,0],[0,0],[0,width],[height,width]], np.float32)
      # dst = np.array([[height,0],[height,width],[0,width],[0,0]], np.float32)
      # dst = np.array([[0,0],[height,0],[height,width],[0,width]], np.float32)


      
      # Code to count numbers of fingers raised will go here
      numRaised = 0
      thumb, ind, mid, ring, lit = 0, 0, 0, 0, 0
    #   raisedFin=[False, False, False, False, False]
      if len(hand_landmarks_list) < 2: return rgb_image, False, (0,0)
      # for each hand...
      
      for idx in range(len(hand_landmarks_list)):
         hand_landmarks = hand_landmarks_list[idx]
         if detection_result.handedness[idx][0].category_name == 'Left':
            wrist = (hand_landmarks[0].x, hand_landmarks[0].y)

            continue 
         
         # hand landmarks is a list of landmarks where each entry in the list has an x, y, and z in normalized image coordinates
         # for each fingertip... (hand_landmarks 4, 8, 12, and 16)

            # make sure finger is higher in image the 3 proceeding values (2 finger segments and knuckle)
         tip_y = hand_landmarks[8].y
         dip_y = hand_landmarks[3].y
         pip_y = hand_landmarks[6].y
         mcp_y = hand_landmarks[5].y
         if tip_y < min(dip_y,pip_y,mcp_y):
            numRaised += 1
               
            ind = 1


         # for the thumb
         # use direction vector from wrist to base of thumb to determine "raised"
         # tip_x = hand_landmarks[4].x
         # dip_x = hand_landmarks[3].x
         # pip_x = hand_landmarks[2].x
         # mcp_x = hand_landmarks[1].x
         # palm_x = hand_landmarks[0].x
         # if mcp_x > palm_x:
         #    if tip_x > max(dip_x,pip_x,mcp_x):
         #       numRaised += 1
         #       thumb = 1
         # else:
         #    if tip_x < min(dip_x,pip_x,mcp_x):
         #       numRaised += 1
         #       thumb = 1

      if ind == 1:
         pixel = 0.1
         dirVec = [hand_landmarks[8].x - hand_landmarks[7].x, hand_landmarks[8].y - hand_landmarks[7].y]
         dirVec = (dirVec / np.linalg.norm(dirVec)) * pixel
         perpVec = [dirVec[1], -1 * dirVec[0]]
         perpVec = (perpVec / np.linalg.norm(perpVec)) * pixel
         # print(dirVec)
         # print(perpVec)
         # print(hand_landmarks[8])
         src = np.array([[(hand_landmarks[8].x + perpVec[0] + dirVec[0])*width , height*(hand_landmarks[8].y + perpVec[1] + dirVec[1])],
                [width* (hand_landmarks[8].x - perpVec[0] + dirVec[0]) , height* (hand_landmarks[8].y - perpVec[1] + dirVec[1])],
                [width* (hand_landmarks[7].x - perpVec[0] - dirVec[0]) , height* (hand_landmarks[7].y - perpVec[1]- dirVec[1])],
                [width* (hand_landmarks[7].x + perpVec[0] - dirVec[0]) , height* (hand_landmarks[7].y + perpVec[1]- dirVec[1])]], np.float32)
         pers = cv2.getPerspectiveTransform(src,dst)
         image = cv2.flip(cv2.warpPerspective(rgb_image, pers, (height, width)),1 )
         axis = np.array([wrist[0] - hand_landmarks[5].x, wrist[1] - hand_landmarks[5].y])
         r_vector = np.array([wrist[0] - hand_landmarks[8].x, wrist[1] - hand_landmarks[8].y])
         
         r = np.linalg.norm(r_vector)
         theta = np.arccos(np.inner(axis,r_vector) / (np.linalg.norm(axis) * r))
         return image, True, (r,theta)
      return rgb_image, False, (0,0) # return transformed image and activated or not
   except Exception as e:
      print(e)
      return rgb_image, False, (0,0)
    
    
def get_images_activated_thumb(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   try:
      # Get Data
      hand_landmarks_list = detection_result.hand_landmarks
      
      height, width = rgb_image.shape[:2]
      dst = np.array([[height,0],[0,0],[0,width],[height,width]], np.float32)
      
      # Code to count numbers of fingers raised will go here
      thumb = 0
    #   raisedFin=[False, False, False, False, False]
      if len(hand_landmarks_list) < 2: return rgb_image, False, (0,0)
      # for each hand...
      
      #left hand is selected for surface
      if detection_result.handedness[0][0].category_name == 'Left':
         wrist = (hand_landmarks_list[0][0].x, hand_landmarks_list[0][0].y)
         hand_landmarks = hand_landmarks_list[1]
      elif detection_result.handedness[1][0].category_name == 'Left':
         wrist = (hand_landmarks_list[1][0].x, hand_landmarks_list[1][0].y)
         hand_landmarks = hand_landmarks_list[0]
      else: 
         return rgb_image, False, (0,0)
      # for idx in range(len(hand_landmarks_list)):
         
      #    hand_landmarks = hand_landmarks_list[idx]
      #    if detection_result.handedness[idx][0].category_name == 'Left':
      #       wrist = (hand_landmarks[0].x, hand_landmarks[0].y)

      #       continue 
         
         #for the thumb
         #use direction vector from wrist to base of thumb to determine "raised"
      tip_x = hand_landmarks[4].x
      dip_x = hand_landmarks[3].x
      pip_x = hand_landmarks[2].x
      mcp_x = hand_landmarks[1].x
      palm_x = hand_landmarks[0].x
      if mcp_x > palm_x:
         if tip_x > max(dip_x,pip_x,mcp_x):
            thumb = 1
      else:
         if tip_x < min(dip_x,pip_x,mcp_x):
            thumb = 1
      
      
      if thumb == 1:
      
         pixel = 0.1
         dirVec = [hand_landmarks[4].x - hand_landmarks[3].x, hand_landmarks[4].y - hand_landmarks[3].y]
         dirVec = (dirVec / np.linalg.norm(dirVec)) * pixel
         perpVec = [dirVec[1], -1 * dirVec[0]]
         perpVec = (perpVec / np.linalg.norm(perpVec)) * pixel
         # print(dirVec)
         # print(perpVec)
         # print(hand_landmarks[4])
         src = np.array([[(hand_landmarks[4].x + perpVec[0] + dirVec[0])*width , height*(hand_landmarks[4].y + perpVec[1] + dirVec[1])],
                [width* (hand_landmarks[4].x - perpVec[0] + dirVec[0]) , height* (hand_landmarks[4].y - perpVec[1] + dirVec[1])],
                [width* (hand_landmarks[3].x - perpVec[0] - dirVec[0]) , height* (hand_landmarks[3].y - perpVec[1]- dirVec[1])],
                [width* (hand_landmarks[3].x + perpVec[0] - dirVec[0]) , height* (hand_landmarks[3].y + perpVec[1]- dirVec[1])]], np.float32)
         pers = cv2.getPerspectiveTransform(src,dst)
         image = cv2.flip(cv2.warpPerspective(rgb_image, pers, (height, width)),1 )
         axis = np.array([wrist[0] - hand_landmarks[1].x, wrist[1] - hand_landmarks[1].y])
         r_vector = np.array([wrist[0] - hand_landmarks[4].x, wrist[1] - hand_landmarks[4].y])
         
         r = np.linalg.norm(r_vector)
         theta = np.arccos(np.inner(axis,r_vector) / (np.linalg.norm(axis) * r))
         return image, True, (r,theta)
      return rgb_image, False, (0,0) # return transformed image and activated or not
   except Exception as e:
      print(e)
      return rgb_image, False, (0,0)
    
def getAngleBetVector(A, B):
   
   ip = A[0] * B[0] + A[1] * B[1]
   
   ip2 = np.linalg.norm(A) * np.linalg.norm(B)
   
   cosX = ip / ip2
   
   x = math.acos(cosX)
   degx = math.degrees(x)
   return degx




port = 'COM5' # 시리얼 포트
baud = 19200 # 시리얼 보드레이트(통신속도)



cap = cv2.VideoCapture(0)
try:
   
   ser = serial.Serial(port, baud)
except Exception as e:
   print(e)
   
# thread = threading.Thread(target=readThread, args=(ser), daemon=True)
hand_landmarker = landmarker_and_result()
unused_frame = []

mode = input() #Folder to save, train or eval
name = input() #batch name

df = pd.DataFrame(columns = ['force', 'touch'])
r_theta_list = []
force_touch_list = []
time.sleep(1)
print("Recording Start")
# ser.write(str.encode("S"))
time.sleep(1)
frame_num = 0
# thread.start()


while(cap.isOpened()):
   ret,frame = cap.read()
   frame_num+=1
   
   frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
   hand_landmarker.detect_async(frame)
   
   # frame = draw_landmarks_on_image(frame,hand_landmarker.result)
   # frame = count_fingers_raised(frame,hand_landmarker.result)

   # frame, read, r_theta = get_images_activated_index_fingers(frame,hand_landmarker.result)
   frame, read, r_theta = get_images_activated_thumb(frame,hand_landmarker.result)
   if not read : 
      unused_frame.append(frame_num)
      ser.reset_input_buffer()
      
   else:
      r_theta_list.append({'r': r_theta[0], 'theta' : r_theta[1]})
      if ser.readable():
         x = ser.readline()
         temp = str(x.decode('utf-8')).rstrip().split(',')
         force_touch_list.append(temp)
      
      cv2.imwrite(os.path.join('dataset', mode,'image',name+'_'+str(frame_num)+'.jpg'), frame)
      print('saved!')
      
      
   cv2.imshow('frame', cv2.resize(frame,(900, 1600 )))
   if cv2.waitKey(33) == ord('q'):
      break
   if frame_num >= 900:
      print('timeisover')
      break
   

hand_landmarker.close()    
cap.release()

cv2.destroyAllWindows()


df_r_theta = pd.DataFrame(r_theta_list, columns=['r','theta'])
df_r_theta.to_csv(os.path.join('dataset', mode, 'angle', name+'.csv'), sep=',', na_rep='NaN') 

# df = pd.read_csv(os.path.join('dataset', mode, 'label', name+'.csv'))
# df.drop(unused_frame, inplace=True)
# df.to_csv(os.path.join('dataset', mode, 'label', name+'.csv'), sep=',', na_rep='NaN') 

df_force_touch = pd.DataFrame(force_touch_list, columns=['force','touch'])

df_force_touch.to_csv(os.path.join('dataset', mode, 'label', name+'.csv'), sep=',', na_rep='NaN') 

