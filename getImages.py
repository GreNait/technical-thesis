import cv2, time
from datetime import datetime
import os
import subprocess
import random

def setCameraSetting(exposure, focus):
    subprocess.run("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=1",shell=True)
    subprocess.run(f"v4l2-ctl -d /dev/video0 --set-ctrl=exposure_absolute={exposure}",shell=True)
    subprocess.run("v4l2-ctl -d /dev/video0 --set-ctrl=focus_auto=0",shell=True)
    subprocess.run(f"v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute={focus}",shell=True)

video = cv2.VideoCapture(0)
setCameraSetting(70,60)

labelName = input("Label name?")
saveingDir = './'+str(labelName)

try:
    os.mkdir(saveingDir)
    print("Directory " + saveingDir+ " created")
except FileExistsError:
    print("Directory already created")

maxAmountImages = int(input("How many images should be saved?"))
timing = int(input("How many seconds between two images?"))

calculateTime = maxAmountImages * timing

try:
    for _ in range(maxAmountImages):
        check, frame = video.read()

        cv2.imshow("Capturing", frame)
        #print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        label = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"_"+str(labelName)
        cv2.imwrite(saveingDir+'/'+label+'.png', frame)

        if(cv2.waitKey(1) == ord('q')):
            break

        print(f"Progress[{(_+1)}] {_ * timing} -> {calculateTime} ")


        setCameraSetting(random.randint(30,110), random.randint(40,80))

        time.sleep(timing)
except Exception as e:
    print(e)
finally:
    video.release()
    cv2.destroyAllWindows()