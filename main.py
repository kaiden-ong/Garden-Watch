import cv2
import numpy as np
import pyaudio
import wave
import time
import threading

# Play audio file function
def play_audio(file):
    global audio_playing
    def audio_thread(file):
        global audio_playing
        chunk = 1024
        wf = wave.open(file, 'rb')
        device_name = "Speakers (JBL Go 3 Stereo)"
        index = None

        p = pyaudio.PyAudio()

        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info.get('name') == device_name and device_info.get('maxOutputChannels') >= 2:
                index = i
                break

        if index is None:
            print(f"Motion Detected... Device {device_name} not found.")
            return
        else:
            print("Human Motion Detected")

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        output_device_index=index)

        data = wf.readframes(chunk)
        while data != b'':
            stream.write(data)
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()
        p.terminate()

        with audio_lock:
            audio_playing = False

    # Check if audio is currently playing
    with audio_lock:
        if audio_playing:
            return
        audio_playing = True

    threading.Thread(target=audio_thread, args=(file,)).start()

# Selecting video area
def select_corners(event, x, y, flags, param):
    global corners

    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append((x,y))

# Automatically create rectangle w/o overlaps
def order_corners():
    global corners
    x1 = corners[0][0]
    y1 = corners[0][1]
    x2 = corners[1][0]
    y2 = corners[1][1]
    x3 = corners[2][0]
    y3 = corners[2][1]
    x4 = corners[3][0]
    y4 = corners[3][1]

    # line1 = (x2-x1,y2-y1)
    # line3 = (x4-x3,y4-y3)

    # line2 = (x3-x2,y3-y2)
    # line4 = (x1-x4,y1-y4)

    # Check 1->3 2->4
    # A + alpha(AB) = C + beta(CD)
        # x1 + alpha(x2-x1) = x3 + beta(x4-x3)
        # y1 + alpha(y2-y1) = y3 + beta(y4-y3)
        # alpha = a/b, beta = c/b
    a = (x4-x3)*(y3-y1)-(y4-y3)*(x3-x1)
    b = (x4-x3)*(y2-y1)-(y4-y3)*(x2-x1)
    c = (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1)

    d = (x1-x4)*(y4-y2)-(y1-y4)*(x4-x2)
    e = (x1-x4)*(y3-y2)-(y1-y4)*(x3-x2)
    f = (x3-x2)*(y4-y2)-(y3-y2)*(x4-x2)

    if a/b > 0 and a/b < 1 and c/b > 0 and c/b < 1:
        temp = corners[1]
        corners[1] = corners[2]
        corners[2] = temp
    elif d/e > 0 and d/e < 1 and f/e > 0 and f/e < 1:
        temp = corners[0]
        corners[0] = corners[1]
        corners[1] = temp
    return

# Display video fram
def display_frame(cap, window_name):
    corners_collected = False

    button_position = (10, 10)
    button_size = (230, 50)
    button_color = (0, 0, 255)
    button_text = "Press space to reset area"
    
    # Some cv2 algs
    bgSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.rectangle(frame, button_position, 
                      (button_position[0] + button_size[0], button_position[1] + button_size[1]),
                      button_color, -1)
        cv2.putText(frame, button_text, (button_position[0] + 10, button_position[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)

        for corner in corners:
            cv2.circle(frame, corner, 3, (0, 255, 0), -1)            

        if len(corners) == 4 and not corners_collected:
            order_corners()
            corners_collected = True

        for i in range(len(corners)):
            cv2.line(frame, corners[i], corners[(i + 1) % len(corners)], (0, 255, 0), 3)
        
        # frame = detectMotionAndHuman(frame, bgSubtractor, hog)
        detectHumanOnly(frame,hog)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 32:  # Space bar
            corners.clear()
            corners_collected = False

# Not in use right now           
def detectMotion(frame, bgSubtractor):
    fgmask = bgSubtractor.apply(frame)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None)
    thresh = cv2.dilate(thresh, None)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motionBoxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            motionBoxes.append((x, y, w, h))
    return motionBoxes

def detectHuman(frame, hog):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    ret,binaryFrame = cv2.threshold(frame,10,255,cv2.THRESH_BINARY)
    boxes, weights = hog.detectMultiScale(binaryFrame, winStride=(8, 8))
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    return boxes

def detectMotionAndHuman(frame, bgSubtractor, hog):
    global first_detection_time
    motionBoxes = detectMotion(frame, bgSubtractor)
    humanBoxes = detectHuman(frame, hog)

    current_time = time.time()
    human_detected = False

    for (xA, yA, xB, yB) in humanBoxes:
        for (mx, my, mw, mh) in motionBoxes:
            if (xA >= mx and yA >= my and xB <= mx + mw and yB <= my + mh):
                human_detected = True
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                break
        if human_detected:
            break

    if human_detected:
        if first_detection_time is None:
            first_detection_time = current_time
        elif current_time - first_detection_time >= 1:
            play_audio('./gardenwatch.wav')
    else:
        first_detection_time = None

    return frame

# Detects humans in the shot
def detectHumanOnly(frame, hog):
    global first_detection_time
    current_time = time.time()
    human_detected = False
    if len(corners) != 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        boxes, weights = hog.detectMultiScale(gray, winStride=(8,8), padding=(4,4), scale=1.05)
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        for (xA, yA, xB, yB) in boxes:
            cv2.rectangle(frame, (xA, yA), (xB, yB),(0, 255, 0), 2)
            human_detected = True
    else:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        polygon = np.array(corners, np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon], 255)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        gray = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2GRAY)
        boxes, weights = hog.detectMultiScale(gray, winStride=(8,8))
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        for (xA, yA, xB, yB) in boxes:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            human_detected = True

    if human_detected:
        if first_detection_time is None:
            first_detection_time = current_time
        elif current_time - first_detection_time >= 1:
            play_audio('./gardenwatch.wav')
            first_detection_time = current_time
    else:
        first_detection_time = None



def main():
    global corners, first_detection_time, audio_playing, audio_lock
    corners = []
    first_detection_time = None
    audio_playing = False
    audio_lock = threading.Lock()

    cap = cv2.VideoCapture(0)
    window_name = 'Garden Watch'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_corners)

    display_frame(cap, window_name)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()