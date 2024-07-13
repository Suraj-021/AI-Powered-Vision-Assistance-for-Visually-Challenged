import cv2
import pygame
from gtts import gTTS
import os
import math
import tkinter as tk
from PIL import Image, ImageTk  # Importing Image and ImageTk from PIL

def pmusic(file):
    pygame.init()
    pygame.mixer.init()
    clock = pygame.time.Clock()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        print("Playing...")
        clock.tick(1000)

def stopmusic():
    pygame.mixer.music.stop()

def getmixerargs():
    pygame.mixer.init()
    freq, size, chan = pygame.mixer.get_init()
    return freq, size, chan

def initMixer():
    BUFFER = 4096  # audio buffer size, number of samples since pygame 1.8.
    FREQ, SIZE, CHAN = getmixerargs()
    pygame.mixer.init(FREQ, SIZE, CHAN, BUFFER)

thres = 0.5  # threshold to detect object

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, draw=True, objects=[]):
    height, width, _ = img.shape
    center_x = width // 2
    center_y = height // 2

    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=0.2)
    print(classIds, bbox)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])

                obj_center_x = (box[0] + box[2]) // 2
                obj_center_y = (box[1] + box[3]) // 2
                dx = obj_center_x - center_x
                dy = obj_center_y - center_y
                distance = math.sqrt(dx ** 2 + dy ** 2)
                angle = math.atan2(dy, dx) * 180 / math.pi

                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 300, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    info_text = f"{className} is at a distance of {distance} pixels, "
                    if angle < -90:
                        info_text += "to your top left."
                    elif -90 <= angle < -45:
                        info_text += "to your left."
                    elif -45 <= angle < 45:
                        info_text += "in front of you."
                    elif 45 <= angle < 90:
                        info_text += "to your right."
                    else:
                        info_text += "to your top right."

                    #myobj = gTTS(text=info_text, lang='en', slow=False)
                    # myobj = gTTS(text=className, lang='en', slow=False)
                    # myobj.save("1.mp3")
                    # initMixer()
                    
                    # myobj = gTTS(text=className, lang='en', slow=False)
                    # myobj.save("1.mp3")
                    # #initMixer()
                    # pmusic("1.mp3")
                    a = className + info_text
                    TTS = gTTS(text=str(a), lang='en', slow=False)
                    TTS.save('1.mp3')
                    os.system('1.mp3')
                    #audio_file = "audio.mp3"
                    #myobj.save(audio_file)
                    #initMixer()
                    #pmusic(audio_file)

                    # Display direction in GUI
                    direction_label.config(text=f"Direction: {info_text}")

    return img, objectInfo

def update_video_feed():
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame, _ = getObjects(frame)
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to ImageTk format
    photo = ImageTk.PhotoImage(image=Image.fromarray(img))
    display_label.imgtk = photo
    display_label.configure(image=photo)
    display_label.after(10, update_video_feed)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    root = tk.Tk()
    root.title("Object Detection")
    
    display_label = tk.Label(root)
    display_label.pack()

    direction_label = tk.Label(root, text="Direction: ")
    direction_label.pack()

    update_video_feed()
    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()
