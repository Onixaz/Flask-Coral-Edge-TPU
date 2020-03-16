

from imutils.video import VideoStream
from flask import Flask, render_template, Response
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
import threading
import imutils
import time
import cv2
from PIL import Image


box_color = (0,0,255)
label_text_color = (255, 255, 255)

outputFrame = None
lock = threading.Lock()
engine = DetectionEngine(
    '/home/pi/Fstreamer/tpu/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
labels = dataset_utils.read_label_file(
    '/home/pi/Fstreamer/tpu/coco_labels.txt')

app = Flask(__name__)


vs = VideoStream(usePiCamera=True,  resolution=(320,240)).start()
#vs = VideoStream(src=0).start()
time.sleep(2.0)


@app.route("/")
def index():
    return render_template("index.html")


def detect_objects():

    global cap, outputFrame, lock

    while True:
        
        start_time = time.time()
        frame = vs.read()
        frame = imutils.rotate(frame, angle=180)

        prepimg = Image.fromarray(frame.copy())

        ans = engine.detect_with_image(
            prepimg, threshold=0.4, keep_aspect_ratio=True, relative_coord=False, top_k=10)
        
    

        if ans:
            for obj in ans:
                box = obj.bounding_box.flatten().tolist()

                box_left = int(box[0])
                box_top = int(box[1])
                box_right = int(box[2])
                box_bottom = int(box[3])
                cv2.rectangle(frame, (box_left, box_top), (box_right,
                                                           box_bottom), box_color, 1)
                
                percentage = int(obj.score * 100)
                label_text = labels[obj.label_id] + " (" + str(percentage) + "%)" 
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = box_left
                label_top = box_top - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]
                cv2.rectangle(frame, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1), box_color, -1)
                cv2.putText(frame, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

        fps = round(1.0 / (time.time() - start_time), 0)

        cv2.putText(frame, f"FPS: {fps}", (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_color, 1)
        
        with lock:
            outputFrame = cv2.resize(frame, (640,480))
            
def generate():
    
    global outputFrame, lock

    while True:
        with lock:
            if outputFrame is None:
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    
    t = threading.Thread(target=detect_objects)
    t.daemon = True
    t.start()

    app.run(host='0.0.0.0', port=5000, debug=True,
            threaded=True, use_reloader=False)

vs.stop()
