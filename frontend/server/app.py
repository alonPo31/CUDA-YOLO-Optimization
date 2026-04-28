import json
import os
import time

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE         = 'C:/Alon/CUDA-YOLO-Optimization'
FRAME_PATH   = f'{BASE}/latest_frame.jpg'
STATUS_PATH  = f'{BASE}/status.json'
CONFIG_PATH  = f'{BASE}/frontend_config.json'

# ערך ברירת מחדל לקובץ הקונפיגורציה
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'w') as f:
        json.dump({'nms_enabled': True}, f)


def read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def generate_frames():
    """MJPEG stream - קורא את הפריים האחרון שנכתב על ידי ה-C++ ומשדר לדפדפן."""
    while True:
        try:
            with open(FRAME_PATH, 'rb') as f:
                frame_bytes = f.read()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
        except Exception:
            pass  # הפריים עדיין לא נכתב, ממתינים
        time.sleep(1 / 30)  # ~30 FPS


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/status')
def status():
    return jsonify(read_json(STATUS_PATH, {'fps': 0, 'object_count': 0, 'nms_enabled': True}))


@app.route('/toggle_nms', methods=['POST'])
def toggle_nms():
    config = read_json(CONFIG_PATH, {'nms_enabled': True})
    config['nms_enabled'] = not config['nms_enabled']
    write_json(CONFIG_PATH, config)
    return jsonify(config)


if __name__ == '__main__':
    print('Flask server running on http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, threaded=True)
