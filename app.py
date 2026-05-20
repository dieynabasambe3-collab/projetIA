from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("runs/classify/train-21/weights/best.pt")
cap = cv2.VideoCapture(0)

def generate():
    while True:
        success, frame = cap.read()
        if not success:
            break

        cv2.imwrite("temp.jpg", frame)
        results = model("temp.jpg")

        names = results[0].names
        probs = results[0].probs.data.tolist()

        class_id = probs.index(max(probs))
        label = names[class_id]
        conf = max(probs)

        cv2.putText(frame, f"{label} {conf:.2f}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
    