# edueye


> A real-time classroom attendance system using face recognition and optional emotion analysis, with a PyQt6 desktop GUI and a Node.js analytics dashboard.

---

## ✨ Features

- **Automatic Attendance** — Recognises registered students via webcam or phone camera and marks them present for the active lecture slot
- **Multi-Lecture Support** — Track attendance across up to 8 lecture periods per day with one click
- **Emotion Analysis** *(optional)* — Powered by [DeepFace](https://github.com/serengil/deepface); detects 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral) per student in real time
- **Live Tracker Overlay** — Colour-coded bounding boxes and name labels drawn on the video feed
- **Daily Excel Reports** — One-click export to `.xlsx` with per-lecture attendance timestamps
- **Phone Camera Support** — Stream from an Android/iPhone via DroidCam or IP Webcam
- **Vision Analytics Dashboard** *(optional)* — A lightweight Node.js/Express web dashboard backed by SQLite

---

## 🗂️ Project Structure

```
EDUEYE/
├── main.py                    # Desktop application — run this
├── encode_face_v2.py          # One-time setup: encodes known faces
├── requirements.txt           # Python dependencies
│
├── known_faces/               # Add student photos here (one folder per person)
│   └── Sample Person/
│       └── README.txt
│
├── attendance/                # Reserved for future use
├── attendance_day/            # Daily Excel reports auto-saved here
│
└── vision_analytics/          # Optional Node.js analytics dashboard
    ├── server.js
    ├── package.json
    └── public/
        ├── index.html
        ├── student.html
        ├── css/
        └── js/
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.9 – 3.11 | Python 3.12 may have issues with some dependencies |
| CMake | Required to build `dlib` |
| C++ compiler | Visual Studio Build Tools (Windows) or `build-essential` (Linux/macOS) |
| Node.js 18+ | Only needed for the optional web dashboard |

### 1. Clone the repository

```bash
git clone https://github.com/Sameer_010406/EDUEYE.git
cd EDUEYE
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Windows users:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with the *Desktop development with C++* workload before running `pip install`.

> **macOS users:** Run `xcode-select --install` and `brew install cmake` first.

### 3. Add face photos

Create one sub-folder per person inside `known_faces/` and place clear face photos inside:

```
known_faces/
├── Alice Johnson/
│   ├── alice_front.jpg
│   └── alice_side.jpg
└── Bob Smith/
    └── bob.jpg
```

- Use **clear, well-lit** photos with a single face per image.
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- The folder name becomes the person's display name in the app.

### 4. Generate face encodings *(run once, or after adding new people)*

```bash
python encode_face_v2.py
```

This creates `known_face_encodings.pkl` in the project root. Re-run this whenever you add or remove people from `known_faces/`.

### 5. Launch the application

```bash
python main.py
```

---

## ⚙️ Configuration

All settings are at the top of `main.py`:

| Variable | Default | Description |
|---|---|---|
| `LAPTOP_CAM_SRC` | `0` | Webcam device index |
| `IPHONE_CAM_SRC_URL` | `http://192.168.1.100:4747/video` | Phone stream URL — change to your device's IP |
| `RECOGNITION_THRESHOLD` | `0.55` | Face match distance — lower = stricter |
| `NUM_LECTURES` | `8` | Number of lecture slots tracked per day |
| `REQUESTED_FACE_DETECT_MODEL` | `"hog"` | `"hog"` (CPU-fast) · `"cnn"` (GPU-accurate) · `"blazeface"` (MediaPipe) |
| `FACE_DETECT_FRAME_SKIP` | `10` | Full detection every N frames; trackers update every frame |
| `EMOTION_ANALYZE_FRAME_SKIP` | `10` | Emotion inference cadence |

---

## 📱 Phone Camera Setup

1. Install **DroidCam** (Android/iOS) or **IP Webcam** (Android) on your phone.
2. Connect your phone and PC to the **same Wi-Fi network**.
3. Open the app and note the video stream URL (e.g. `http://192.168.1.100:4747/video`).
4. Update `IPHONE_CAM_SRC_URL` in `main.py` with your phone's IP.
5. Click **Phone Cam** in the app to switch sources.

---

## 🌐 Vision Analytics Dashboard *(optional)*

A web dashboard for viewing attendance and emotion history.

```bash
cd vision_analytics
npm install
npm start
```

Open `http://localhost:3000` in your browser.

> **Note:** The dashboard uses a separate SQLite database. Integration with the Python backend requires additional setup (REST calls from `main.py` are not wired by default).

---

## 🖥️ Usage

| UI Element | Action |
|---|---|
| **Lecture buttons (1–8)** | Select the active lecture period |
| **Laptop Cam / Phone Cam** | Switch camera source |
| **Start Emo** | Enable real-time emotion detection (requires DeepFace) |
| **Show Details** | Open floating emotion breakdown windows per student |
| **Save Daily Report** | Export today's attendance to `.xlsx` in `attendance_day/` |
| **Reset Today's Log** | Clear all attendance records for the current day |

The system auto-resets attendance data at midnight when the date changes.

---

## 🔒 Privacy & Ethics

- **Face photos and encodings are personal biometric data.** The `known_faces/` directory and `known_face_encodings.pkl` file are listed in `.gitignore` and must **never** be committed to a public repository.
- Obtain informed consent from all individuals before enrolling their faces.
- Comply with applicable data protection regulations (GDPR, PDPA, IT Act, etc.) in your jurisdiction.
- This project is intended for educational and research purposes.

---

## 🐛 Troubleshooting

**`dlib` fails to install**
Ensure CMake and a C++ compiler are installed and on your PATH. Try `pip install cmake` first.

**`known_face_encodings.pkl` not found on startup**
Run `python encode_face_v2.py` first.

**No faces detected during encoding**
Check that photos are clear and well-lit with one face per image. Lower `DETECTION_CONFIDENCE` in `encode_face_v2.py` for lower-quality photos.

**Camera not opening**
Check the device index (`LAPTOP_CAM_SRC`) or stream URL. On Windows, the `CAP_DSHOW` backend is tried automatically as a fallback.

**DeepFace / emotion detection unavailable**
Install TensorFlow: `pip install tensorflow` (or `tensorflow-cpu`). Set the environment variable `TF_USE_LEGACY_KERAS=1`.

---

## 📦 Dependencies

### Python
| Package | Purpose |
|---|---|
| `opencv-python` | Camera capture and image processing |
| `face-recognition` | Face encoding and distance comparison |
| `dlib` | Correlation tracker and HOG/CNN face detection |
| `deepface` | Emotion analysis |
| `mediapipe` | BlazeFace fast face detection |
| `PyQt6` | Desktop GUI |
| `openpyxl` | Excel report generation |
| `numpy` | Numerical operations |
| `Pillow` | Image I/O |

### Node.js (dashboard only)
| Package | Purpose |
|---|---|
| `express` | Web server |
| `sqlite3` | Attendance/emotion database |
| `cors` | Cross-origin resource sharing |
| `morgan` | HTTP request logging |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please do **not** include any real face photos or biometric data in pull requests.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [DeepFace](https://github.com/serengil/deepface) by Sefik Ilkin Serengil
- [MediaPipe](https://github.com/google/mediapipe) by Google
- [dlib](http://dlib.net/) by Davis King
