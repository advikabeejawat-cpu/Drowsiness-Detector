# Face Mesh Detection

A Python computer vision project that uses **OpenCV** and **MediaPipe Face Mesh** to detect facial landmarks in real time through a webcam.

## ✨ Features

* 📷 Real-time webcam video
* 🙂 Face landmark detection
* 🕸️ Face mesh visualization
* 🎯 Detects up to one face at a time
* 🔄 Real-time face tracking
* ⌨️ Press **Esc** to exit

## 🛠️ Technologies Used

* Python
* OpenCV
* MediaPipe

## 📁 Project Structure

```text
Face-Mesh-Detection/
│
├── face.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Requirements

Before running the project, make sure you have:

* Python 3.9+
* A working webcam
* OpenCV
* MediaPipe

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Face-Mesh-Detection.git
cd Face-Mesh-Detection
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

Run the following command:

```bash
python face.py
```

The program will access the default webcam and display the live camera feed with the detected face mesh.

## 🔄 How It Works

1. OpenCV accesses the computer's webcam.
2. The camera captures frames continuously.
3. Each frame is converted from BGR to RGB.
4. MediaPipe Face Mesh processes the image.
5. Facial landmarks are detected.
6. The detected landmarks are connected to create a face mesh.
7. The processed frame is displayed in a window.

The project uses the webcam through:

```python
cap = cv2.VideoCapture(0)
```

The camera index can be changed if the default camera does not work.

## 🎯 Face Detection Settings

The project is configured to detect a maximum of one face:

```python
max_num_faces=1
```

It also uses:

```python
min_detection_confidence=0.5
min_tracking_confidence=0.5
```

These settings control the minimum confidence required for face detection and tracking.

## 🕸️ Face Mesh

When facial landmarks are detected, the program draws the MediaPipe face mesh over the detected face.

The project uses:

```python
mp_face_mesh.FACEMESH_TESSELATION
```

to create the mesh connections between facial landmarks.

## ⌨️ Controls

Press:

```text
Esc
```

to stop the program and close the webcam window.

## ⚠️ Troubleshooting

### Camera is not opening

The project uses camera index `0` by default:

```python
cap = cv2.VideoCapture(0)
```

If your camera does not open, try another camera index such as:

```python
cap = cv2.VideoCapture(1)
```

or:

```python
cap = cv2.VideoCapture(2)
```

The available camera index depends on your system.

### Webcam shows a blank screen

Make sure:

* Your webcam is connected.
* Camera permissions are enabled.
* No other application is currently using the webcam.
* You are using the correct camera index.

## 🚀 Future Improvements

* 👥 Support multiple faces
* 👁️ Eye tracking
* 😊 Facial expression detection
* 😷 Face mask detection
* 📏 Face distance estimation
* 🎨 Custom face landmark styles
* 🖥️ Graphical user interface
* 📊 Real-time facial landmark statistics
