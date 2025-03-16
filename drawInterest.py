import cv2
import numpy as np

# area1 = Red
# area2 = Blue
#right
lab6_z1 = [(781,392), (809,391), (850,423), (823,434)]
lab6_z2 = [(744,393), (765,393), (810,442), (781, 445)]

shomrom_z1 = [(753,330), (770,331 ), (798,367), (782,369)]
shomrom_z2 = [(724, 368),(712,334 ) ,(735,333 ), (751,367 )]

lab8_z1 = [(704, 258), (716,256), (730, 273), (715,277)]
lab8_z2 = [(681,260), (692,260), (700,278), (686,280)]

#left
# x2 x4 x3 x1
# เก็บของ
smo2_z1 = [(349, 371), (380, 323), (399, 325), (368, 372)]
smo2_z2 = [(411, 326), (387, 375), (414, 380), (435, 327)]

smo1_z1 = [(237, 446), (279, 396), (301, 404), (261, 452)]
smo1_z2 = [(276, 457), (314, 406), (337, 409), (302, 459)]

lab7_z1 = [(374,286),(406,291),(434,253),(403,249)]
lab7_z2 = [(418,293),(444,298),(468,255),(447,250)]
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', RGB)

cap = cv2.VideoCapture('rtsp://admin:admin1234@172.17.18.36')

count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    
    cv2.polylines(frame, [np.array(lab6_z1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(lab6_z2, np.int32)], True, (255, 0, 0), 2)
        
    cv2.polylines(frame, [np.array(shomrom_z1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(shomrom_z2, np.int32)], True, (255, 0, 0), 2)
    
    cv2.polylines(frame, [np.array(smo1_z1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(smo1_z2, np.int32)], True, (255, 0, 0), 2)
    
    cv2.polylines(frame, [np.array(smo2_z1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(smo2_z2, np.int32)], True, (255, 0, 0), 2)
    
    cv2.polylines(frame, [np.array(lab7_z1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(lab7_z2, np.int32)], True, (255, 0, 0), 2)
    
    cv2.polylines(frame, [np.array(lab8_z1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(lab8_z2, np.int32)], True, (255, 0, 0), 2)
    
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
