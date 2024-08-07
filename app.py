from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import mediapipe as mp
import os
import numpy as np
import requests
import uuid

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh

# Define landmarks for facial features
EYEBROW_LANDMARKS = [33, 133, 153, 154]
MOUTH_LANDMARKS = list(range(61, 81))
NOSE_LANDMARKS = [1, 2, 5, 6, 7, 8, 9, 10, 11]
EAR_LANDMARKS = [234, 454]
HAIR_LANDMARKS = [1, 2, 3]
EYE_LANDMARKS = list(range(33, 133))  # Adjust if needed

def get_landmark_positions(face_landmarks, landmark_indices):
    """Extract coordinates from key facial landmarks."""
    positions = {}
    for index in landmark_indices:
        x = face_landmarks.landmark[index].x
        y = face_landmarks.landmark[index].y
        positions[index] = {'x': x, 'y': y}
    return positions

def calculate_distance(p1, p2):
    """Calculate distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def analyze_face_landmarks(face_landmarks):
    """Analyze facial landmarks to determine features."""
    landmark_data = {
        "ears": "attached",
        "eyebrows": "up",
        "facialHair": "scruff",
        "hair": "full",
        "nose": "curve",
        "mouth": "smile",
        "eyes": "eyes"  # Default eye state
    }

    # Analyze eyebrows
    eyebrow_positions = get_landmark_positions(face_landmarks, EYEBROW_LANDMARKS)
    eyebrow_y_values = [pos['y'] for pos in eyebrow_positions.values()]
    if np.mean(eyebrow_y_values) < 0.5:
        landmark_data["eyebrows"] = "up"
    else:
        if any(pos['y'] < 0.5 for pos in eyebrow_positions.values()):
            landmark_data["eyebrows"] = "eyelashesDown"
        else:
            landmark_data["eyebrows"] = "down"

    # Analyze mouth
    mouth_positions = get_landmark_positions(face_landmarks, MOUTH_LANDMARKS)
    mouth_y_values = [pos['y'] for pos in mouth_positions.values()]

    if len(mouth_y_values) < 16:
        landmark_data["mouth"] = "unknown"  # Insufficient data case
    else:
        top_lip_y = mouth_y_values[0:8]
        bottom_lip_y = mouth_y_values[8:16]

        top_lip_height = np.max(top_lip_y) - np.min(top_lip_y)
        bottom_lip_height = np.max(bottom_lip_y) - np.min(bottom_lip_y)
        
        mouth_width = calculate_distance(
            (mouth_positions[MOUTH_LANDMARKS[0]]['x'], mouth_positions[MOUTH_LANDMARKS[0]]['y']),
            (mouth_positions[MOUTH_LANDMARKS[10]]['x'], mouth_positions[MOUTH_LANDMARKS[10]]['y'])
        )

        if top_lip_height > 0.02 and bottom_lip_height > 0.02 and mouth_width > 0.05:
            landmark_data["mouth"] = "smile"
        elif top_lip_height < 0.015 and bottom_lip_height < 0.015 and mouth_width < 0.05:
            landmark_data["mouth"] = "frown"
        elif top_lip_height < 0.02 and bottom_lip_height > 0.02:
            landmark_data["mouth"] = "pucker"
        elif top_lip_height > 0.02 and bottom_lip_height < 0.02:
            landmark_data["mouth"] = "smirk"
        elif top_lip_height < 0.015 and bottom_lip_height > 0.03:
            landmark_data["mouth"] = "sad"
        elif top_lip_height > 0.03 and bottom_lip_height < 0.015:
            landmark_data["mouth"] = "laughing"
        elif top_lip_height > 0.015 and bottom_lip_height < 0.015:
            landmark_data["mouth"] = "nervous"
        elif top_lip_height > 0.025 and bottom_lip_height > 0.025:
            landmark_data["mouth"] = "surprised"

    # Analyze nose
    nose_positions = get_landmark_positions(face_landmarks, NOSE_LANDMARKS)
    nose_width = calculate_distance(
        (nose_positions[NOSE_LANDMARKS[0]]['x'], nose_positions[NOSE_LANDMARKS[0]]['y']),
        (nose_positions[NOSE_LANDMARKS[1]]['x'], nose_positions[NOSE_LANDMARKS[1]]['y'])
    )
    if nose_width > 0.05:
        landmark_data["nose"] = "pointed"
    elif nose_width < 0.03:
        landmark_data["nose"] = "round"
    else:
        landmark_data["nose"] = "curve"

    # Analyze hair
    hair_positions = get_landmark_positions(face_landmarks, HAIR_LANDMARKS)
    hair_width = calculate_distance(
        (hair_positions[HAIR_LANDMARKS[0]]['x'], hair_positions[HAIR_LANDMARKS[0]]['y']),
        (hair_positions[HAIR_LANDMARKS[1]]['x'], hair_positions[HAIR_LANDMARKS[1]]['y'])
    )
    if hair_width < 0.05:
        landmark_data["hair"] = "dannyPhantom"
    elif hair_width < 0.1:
        landmark_data["hair"] = "dougFunny"
    elif hair_width < 0.15:
        landmark_data["hair"] = "fonze"
    else:
        landmark_data["hair"] = "full"

    # Analyze ears
    ear_positions = get_landmark_positions(face_landmarks, EAR_LANDMARKS)
    if len(ear_positions) == 2:
        ear_distance = calculate_distance(
            (ear_positions[EAR_LANDMARKS[0]]['x'], ear_positions[EAR_LANDMARKS[0]]['y']),
            (ear_positions[EAR_LANDMARKS[1]]['x'], ear_positions[EAR_LANDMARKS[1]]['y'])
        )
        if ear_distance > 0.2:
            landmark_data["ears"] = "detached"
        else:
            landmark_data["ears"] = "attached"

    # Analyze eyes
    eye_positions = get_landmark_positions(face_landmarks, EYE_LANDMARKS)
    if len(eye_positions) >= 2:
        # Calculate average eye width
        left_eye_width = calculate_distance(
            (eye_positions[EYE_LANDMARKS[0]]['x'], eye_positions[EYE_LANDMARKS[0]]['y']),
            (eye_positions[EYE_LANDMARKS[16]]['x'], eye_positions[EYE_LANDMARKS[16]]['y'])
        )
        right_eye_width = calculate_distance(
            (eye_positions[EYE_LANDMARKS[17]]['x'], eye_positions[EYE_LANDMARKS[17]]['y']),
            (eye_positions[EYE_LANDMARKS[33]]['x'], eye_positions[EYE_LANDMARKS[33]]['y'])
        )

        if left_eye_width > 0.1 and right_eye_width > 0.1:
            landmark_data["eyes"] = "eyes"
        elif left_eye_width < 0.05 and right_eye_width < 0.05:
            landmark_data["eyes"] = "smilingShadow"
        elif left_eye_width > 0.07 and right_eye_width > 0.07:
            landmark_data["eyes"] = "smiling"
        else:
            landmark_data["eyes"] = "round"  # Default state if not matching other conditions

    return landmark_data

def generate_avatar(landmark_data, avatar_filename):
    """Send request to DiceBear API and return avatar image path."""
    base_url = "https://api.dicebear.com/9.x/micah/svg"
    params = {
        "mouth": landmark_data.get("mouth", "smile"),
        "ears": landmark_data.get("ears", "attached"),
        "eyebrows": landmark_data.get("eyebrows", "up"),
        "facialHair": landmark_data.get("facialHair", "scruff"),
        "hair": landmark_data.get("hair", "full"),
        "eyes": landmark_data.get("eyes", "eyes"),  # Added eyes parameter
        "size": 200,
        "baseColor": "f9c9b6"
    }
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        avatar_path = os.path.join('static', avatar_filename)
        with open(avatar_path, "wb") as f:
            f.write(response.content)
        return f"/static/{avatar_filename}"
    else:
        return None

def process_image(image_path):
    """Process image and return JSON data with facial features and avatar."""
    image = cv2.imread(image_path)
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmark_data = analyze_face_landmarks(face_landmarks)
            avatar_filename = f"avatar_{uuid.uuid4().hex}.svg"
            avatar_url = generate_avatar(landmark_data, avatar_filename)
            return {
                'feature_data': landmark_data,
                'avatar_url': avatar_url,
                'avatar_filename': avatar_filename  # Return filename for download link
            }
        else:
            return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = os.path.join('uploads', file.filename)
        file.save(filename)
        result = process_image(filename)
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Failed to process image'})
    return jsonify({'error': 'Invalid file'})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)


