# Introduction
This is a simple code to detect faces, emotions and age in a video using webcam, it uses the [DeepFace](https://github.com/serengil/deepface) library to analyze the faces and get the emotions.
# Introduction
This is a simple code to detect faces, emotions and age in a video using webcam, it uses the [DeepFace](https://github.com/serengil/deepface) library to analyze the faces and get the emotions.

# How it works
The code uses the [Haar Cascade](https://www.geeksforgeeks.org/python/opencv-python-program-face-detection/) to detect faces in the video. Then it uses the [DeepFace](https://github.com/serengil/deepface) library to analyze the faces and get the emotions.


# How to run 
You need to install the following packages run this:

```bash
python -m venv .venv
source .venv/bin/activate # or .venv/Scripts/activate on Windows
pip install -r requirements.txt
```

Or if you don't want to use virtual environment you can use a conda environment:
```bash
conda create -n deepface python=3.13.7

conda activate deepface
pip install -r requirements.txt
```

In many way just make sure you have the right packages installed all packages are listed in the `requirements.txt` file.


Before running this code you need to check you webcam is working and you have the right permissions to use it.
You can change the webcam number in the code to use the external webcam in file `main.py` line `cam = cv2.VideoCapture(2)` if you want to use your webcam on laptop or `cam = cv2.VideoCapture(0)` if you want to use your webcam on external device.

When you done you can run the code with `python main.py` and you will see the output in the terminal.


# How it works
The code uses the [Haar Cascade](https://www.geeksforgeeks.org/python/opencv-python-program-face-detection/) to detect faces in the video. Then it uses the [DeepFace](https://github.com/serengil/deepface) library to analyze the faces and get the emotions.

Here is some inpomations about the project [Web Hackers Realm](https://www.hackersrealm.net/post/gender-and-age-prediction-using-python) 


# How to run 
You need to install the following packages run this:

```bash
python -m venv .venv
source .venv/bin/activate # or .venv/Scripts/activate on Windows
pip install -r requirements.txt
```

Or if you don't want to use virtual environment you can use a conda environment:
```bash
conda create -n deepface python=3.13.7
conda activate deepface
pip install -r requirements.txt
```

In many way just make sure you have the right packages installed all packages are listed in the `requirements.txt` file.


Before running this code you need to check you webcam is working and you have the right permissions to use it.
You can change the webcam number in the code to use the external webcam in file `main.py` line `cam = cv2.VideoCapture(2)` if you want to use your webcam on laptop or `cam = cv2.VideoCapture(0)` if you want to use your webcam on external device.

When you done you can run the code with `python main.py` and you will see the output in the terminal.

