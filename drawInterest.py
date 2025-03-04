import cv2
import numpy as np

# area1 = Red
# area2 = Blue

# x1 x2 x4 x3
#right
lab6_z1 = [(619, 317,),(645, 313),(663, 220), (692, 257), (685, 261)]
lab6_z2 = [(639, 230), (646, 227), (671, 265), (663, 269)]

shomrom_z1 = [(627, 189), (635, 184), (657, 213), (646, 215)]
shomrom_z2 = [(604, 194), (612, 190), (630, 218), (625, 224)]

lab8_z1 = [(608, 147), (614, 147), (620, 158), (613, 159)]
lab8_z2 = [(588, 150), (594, 150), (600, 161), (593, 162)]

#left
# x2 x4 x3 x1
smo2_z1 = [(390, 333), (391, 209), (415, 184), (407, 180)]
smo2_z2 = [(408, 212), (421, 220), (442, 194), (431, 190)]

smo1_z1 = [(395, 346), (417, 354), (433, 321), (414, 314)]
smo1_z2 = [(429, 359), (443, 358), (452, 328), (440, 324)]

lab7_z1 = [(429, 157), (446, 134), (451, 138), (436, 161)]
lab7_z2 = [(450, 160), (462, 138), (471, 140), (457, 164)]


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', RGB)

cap = cv2.VideoCapture('rtsp://admin:admin1234@192.168.1.108')

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
