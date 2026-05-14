import json
import os
import time

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE        = 'C:/Alon/CUDA-YOLO-Optimization'
FRAME_PATH  = f'{BASE}/latest_frame.jpg'
STATUS_PATH = f'{BASE}/status.json'
CONFIG_PATH = f'{BASE}/frontend_config.json'
UPLOAD_DIR  = f'{BASE}/uploads'

os.makedirs(UPLOAD_DIR, exist_ok=True)

if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'w') as f:
        json.dump({'nms_enabled': True, 'tracking_enabled': True,
                   'video_source': 'camera', 'video_path': ''}, f)


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
    while True:
        try:
            with open(FRAME_PATH, 'rb') as f:
                frame_bytes = f.read()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception:
            pass
        time.sleep(1 / 30)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/frame')
def frame():
    try:
        with open(FRAME_PATH, 'rb') as f:
            data = f.read()
        response = Response(data, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store, no-cache'
        return response
    except Exception:
        return '', 204


@app.route('/status')
def status():
    return jsonify(read_json(STATUS_PATH, {'fps': 0, 'object_count': 0, 'nms_enabled': True}))


@app.route('/toggle_nms', methods=['POST'])
def toggle_nms():
    config = read_json(CONFIG_PATH, {'nms_enabled': True, 'tracking_enabled': True})
    config['nms_enabled'] = not config['nms_enabled']
    write_json(CONFIG_PATH, config)
    return jsonify(config)


@app.route('/toggle_tracking', methods=['POST'])
def toggle_tracking():
    config = read_json(CONFIG_PATH, {'nms_enabled': True, 'tracking_enabled': True})
    config['tracking_enabled'] = not config['tracking_enabled']
    write_json(CONFIG_PATH, config)
    return jsonify(config)


@app.route('/upload_video', methods=['POST'])
def upload_video():
    f = request.files.get('file')
    if not f or not f.filename.lower().endswith('.mp4'):
        return jsonify({'error': 'mp4 file required'}), 400

    save_path = os.path.join(UPLOAD_DIR, f.filename).replace('\\', '/')
    f.save(save_path)

    # Tell the C++ backend to switch from camera to this file.
    # C++ reads frontend_config.json every 15 frames and will restart its VideoCapture.
    config = read_json(CONFIG_PATH, {'nms_enabled': True, 'tracking_enabled': True})
    config['video_source'] = 'file'
    config['video_path']   = save_path
    write_json(CONFIG_PATH, config)

    return jsonify({'status': 'started', 'filename': f.filename})


@app.route('/switch_to_camera', methods=['POST'])
def switch_to_camera():
    config = read_json(CONFIG_PATH, {'nms_enabled': True, 'tracking_enabled': True})
    config['video_source'] = 'camera'
    config['video_path']   = ''
    write_json(CONFIG_PATH, config)
    return jsonify({'status': 'switched'})


if __name__ == '__main__':
    print('Flask server running on http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, threaded=True)
