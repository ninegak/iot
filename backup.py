import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('yolo11n.pt')

# area1 = Red
# area2 = Blue
#right
lab6_z1 = [(639,317), (650,344), (666,335), (650,311)]
lab6_z2 = [(610,324), (618,354), (640,347), (632,318)]

shomrom_z1 = [(625,299), (636,311), (643,305), (631,289)]
shomrom_z2 = [(604,300), (612,320), (630,313), (618,295)]

lab8_z1 = [(599,261), (607,273), (615,267), (606,254)]
lab8_z2 = [(574,259), (592,277), (606,279), (595,258)]

#left
# x2 x4 x3 x1
smo2_z1 = [(427,291), (411,307), (422,315), (432,300)]
smo2_z2 = [(439,295), (428,320), (437,322), (451,301)]

smo1_z1 = [(399,319), (378,336), (396,345), (412,325)]
smo1_z2 = [(419,321), (399,351), (413,353), (433,331)]

lab7_z1 = [(463,253), (440,279), (445,283), (469,259)]
lab7_z2 = [(475,258), (453,290), (463,296), (483,264)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', RGB)

cap = cv2.VideoCapture('rtsp://admin:admin1234@192.168.1.108')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

tracker = Tracker()

lab6_people_entering = {}
lab6_entering = set()
lab6_people_exiting = {}
lab6_exiting = set()

shomrom_people_entering = {}
shomrom_entering = set()
shomrom_people_exiting = {}
shomrom_exiting = set()

smo2_people_entering = {}
smo2_entering = set()
smo2_people_exiting = {}
smo2_exiting = set()

lab7_people_entering = {}
lab7_entering = set()
lab7_people_exiting = {}
lab7_exiting = set()

lab8_people_entering = {}
lab8_entering = set()
lab8_people_exiting = {}
lab8_exiting = set()

smo1_people_entering = {}
smo1_entering = set()
smo1_people_exiting = {}
smo1_exiting = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)

    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []

    person_count = 0

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])
            person_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f'Person Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        result = cv2.pointPolygonTest(np.array(lab6_z2, np.int32), (x4, y4), False)
        if result >= 0:
            lab6_people_entering[id] = (x4, y4)
            # pass zone 1
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        if id in lab6_people_entering:
            result1 = cv2.pointPolygonTest(np.array(lab6_z1, np.int32), (x4, y4), False)
            if result1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                lab6_entering.add(id)

        result2 = cv2.pointPolygonTest(np.array(lab6_z1, np.int32), (x4, y4), False)
        if result2 >= 0:
            lab6_people_exiting[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if id in lab6_people_exiting:
            result3 = cv2.pointPolygonTest(np.array(lab6_z2, np.int32), (x4, y4), False)
            if result3 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                lab6_exiting.add(id)
        
        # ShomROm        
        result4 = cv2.pointPolygonTest(np.array(shomrom_z2, np.int32), (x4, y4), False)
        if result4 >= 0:
            shomrom_people_entering[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        if id in shomrom_people_entering:
            result5 = cv2.pointPolygonTest(np.array(shomrom_z1, np.int32), (x4, y4), False)
            if result5 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                shomrom_entering.add(id)

        result6 = cv2.pointPolygonTest(np.array(shomrom_z1, np.int32), (x4, y4), False)
        if result6 >= 0:
            shomrom_people_exiting[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if id in shomrom_people_exiting:
            result7 = cv2.pointPolygonTest(np.array(shomrom_z2, np.int32), (x4, y4), False)
            if result7 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                shomrom_exiting.add(id)
                
        # SMO2
        result8 = cv2.pointPolygonTest(np.array(smo2_z2, np.int32), (x4, y4), False)
        if result8 >= 0:
            smo2_people_entering[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        if id in smo2_people_entering:
            result9 = cv2.pointPolygonTest(np.array(smo2_z1, np.int32), (x4, y4), False)
            if result9 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                smo2_entering.add(id)

        result10 = cv2.pointPolygonTest(np.array(smo2_z1, np.int32), (x4, y4), False)
        if result10 >= 0:
            smo2_people_exiting[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if id in smo2_people_exiting:
            result11 = cv2.pointPolygonTest(np.array(smo2_z2, np.int32), (x4, y4), False)
            if result11 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                smo2_exiting.add(id)

        # LAB7
        result12 = cv2.pointPolygonTest(np.array(lab7_z2, np.int32), (x4, y4), False)
        if result12 >= 0:
            lab7_people_entering[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        if id in lab7_people_entering:
            result13 = cv2.pointPolygonTest(np.array(lab7_z1, np.int32), (x4, y4), False)
            if result13 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                lab7_entering.add(id)

        result14 = cv2.pointPolygonTest(np.array(lab7_z1, np.int32), (x4, y4), False)
        if result14 >= 0:
            lab7_people_exiting[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if id in lab7_people_exiting:
            result15 = cv2.pointPolygonTest(np.array(lab7_z2, np.int32), (x4, y4), False)
            if result15 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                lab7_exiting.add(id)

        # LAB8
        result16 = cv2.pointPolygonTest(np.array(lab8_z2, np.int32), (x4, y4), False)
        if result16 >= 0:
            lab8_people_entering[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        if id in lab8_people_entering:
            result17 = cv2.pointPolygonTest(np.array(lab8_z1, np.int32), (x4, y4), False)
            if result17 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                lab8_entering.add(id)

        result18 = cv2.pointPolygonTest(np.array(lab8_z1, np.int32), (x4, y4), False)
        if result18 >= 0:
            lab8_people_exiting[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if id in lab8_people_exiting:
            result19 = cv2.pointPolygonTest(np.array(lab8_z2, np.int32), (x4, y4), False)
            if result19 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                lab8_exiting.add(id)

        # SMO1
        result20 = cv2.pointPolygonTest(np.array(smo1_z2, np.int32), (x4, y4), False)
        if result20 >= 0:
            lab8_people_entering[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        if id in lab8_people_entering:
            result21 = cv2.pointPolygonTest(np.array(smo1_z1, np.int32), (x4, y4), False)
            if result21 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                lab8_entering.add(id)

        result22 = cv2.pointPolygonTest(np.array(smo1_z1, np.int32), (x4, y4), False)
        if result22 >= 0:
            lab8_people_exiting[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if id in lab8_people_exiting:
            result23 = cv2.pointPolygonTest(np.array(smo1_z2, np.int32), (x4, y4), False)
            if result23 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                lab8_exiting.add(id)
                
    cv2.polylines(frame, [np.array(lab6_z1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(lab6_z2, np.int32)], True, (255, 0, 0), 2)
    
    cv2.polylines(frame, [np.array(shomrom_z1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(shomrom_z2, np.int32)], True, (255, 0, 0), 2)
    
    cv2.polylines(frame, [np.array(smo2_z1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(smo2_z2, np.int32)], True, (255, 0, 0), 2)
    
    cv2.polylines(frame, [np.array(lab7_z1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(lab7_z2, np.int32)], True, (255, 0, 0), 2)
    
    cv2.polylines(frame, [np.array(lab8_z1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(lab8_z2, np.int32)], True, (255, 0, 0), 2)
    
    cv2.polylines(frame, [np.array(smo1_z1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(smo1_z2, np.int32)], True, (255, 0, 0), 2)

    i = len(lab6_entering)
    o = len(lab6_exiting)
    cv2.putText(frame, "Lab6 Entering: " + str(i), (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Lab6 Exiting: " + str(o), (60, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    i = len(shomrom_entering)
    o = len(shomrom_exiting)
    cv2.putText(frame, "Shomrom Entering: " + str(i), (60, 120), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Shomrom Exiting: " + str(o), (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    i = len(smo2_entering)
    o = len(smo2_exiting)
    cv2.putText(frame, "Smo2 Entering: " + str(i), (60, 160), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Smo2 Exiting: " + str(o), (60, 180), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    i = len(lab7_entering)
    o = len(lab7_exiting)
    cv2.putText(frame, "Lab7 Entering: " + str(i), (60, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Lab7 Exiting: " + str(o), (60, 220), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    i = len(lab8_entering)
    o = len(lab8_exiting)
    cv2.putText(frame, "Lab8 Entering: " + str(i), (60, 240), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Lab8 Exiting: " + str(o), (60, 260), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    i = len(smo1_entering)
    o = len(smo1_exiting)
    cv2.putText(frame, "Smo1 Entering: " + str(i), (60, 280), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Smo1 Exiting: " + str(o), (60, 300), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)


    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()