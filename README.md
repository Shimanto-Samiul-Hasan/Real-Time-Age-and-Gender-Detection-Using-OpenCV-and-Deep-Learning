# Real-Time Age and Gender Prediction System Using OpenCV & Deep Learning

This project is a real-time age and gender prediction system that uses deep learning models to detect faces, predict age ranges, and classify gender. Built with OpenCV and pre-trained convolutional neural networks, this application can analyze both static images and live webcam feeds, providing a user-friendly interface with real-time feedback.

## Features

- **Real-Time Age and Gender Prediction**: Detects faces and predicts age range and gender in both live webcam feeds and static images.
- **Multi-Face Detection**: Supports multi-face detection, processing each detected face for age and gender independently.
- **User-Friendly Interface**: Uses Tkinter to provide a simple GUI where users can select between webcam and image upload.
- **Error Handling**: Provides real-time notifications and error handling for scenarios like low lighting or undetected faces.
- **Comparative Performance**: Performs competitively with popular tools like Dlib, DeepFace, and Face++ in age and gender classification.

## System Architecture

The application is organized into three main layers:
1. **User Interface Layer**: Handles user interaction and displays results.
2. **Processing Layer**: Manages face detection, feature extraction, and prediction processing.
3. **Error Handling Layer**: Manages real-time feedback, including error notifications and loop control for continuous webcam processing.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/age-gender-prediction.git
   cd age-gender-prediction
   ```

2. **Install Required Packages**:
   - OpenCV
   - Numpy
   - Tkinter (usually pre-installed with Python)
   - Matplotlib (optional, for plotting graphs)
   - Graphviz (for generating architecture diagrams, if needed)

3. **Download Pre-Trained Models**:
   - Download the required pre-trained models for face detection, age, and gender prediction:
     - **Face Detection Model**: `opencv_face_detector.pbtxt` and `opencv_face_detector_uint8.pb`
     - **Age Prediction Model**: `age_net.caffemodel` and `age_deploy.prototxt`
     - **Gender Prediction Model**: `gender_net.caffemodel` and `gender_deploy.prototxt`
   
## Usage

1. **Run the Application**:
   Run the main script to start the application:
   ```bash
   python app.py
   ```

2. **Using the Interface**:
   - **Select Webcam or Image**: At the start, choose between live webcam input or uploading a static image.
   - **Live Feed**: If using the webcam, press **'q'** to quit the feed.
   - **Error Feedback**: The application will notify you if no face is detected or if there are lighting issues affecting accuracy.

3. **Results**:
   - Predictions for age range and gender will appear in real-time on the detected faces.
   - For multi-face detection, each face in the frame or image will have separate predictions.

## Project Structure

```plaintext
├── app.py                   # Main application file
├── age_net.caffemodel
├── age_deploy.prototxt
├── gender_net.caffemodel
├── gender_deploy.prototxt
├── opencv_face_detector_uint8.pb
├── opencv_face_detector.pbtxt
├── README.md                # Project documentation
└── images/                  # Optional folder for storing test images
```

## Example Results

![Webcam Test](images/test2.png)


![Loaded Image Test](images/test1.png)

## Future Improvements

- **Enhanced Lighting Robustness**: Add pre-processing for improved accuracy under varied lighting conditions.
- **Expanded Demographics**: Include additional features like emotion detection for broader applicability.
- **Custom Model Training**: Train custom models on diverse datasets for improved accuracy in age classification.


## References

1. **OpenCV Documentation**:[https://opencv.org/](https://opencv.org/)

2. **Pre-trained Models for Age and Gender Prediction**:
   - Levi, G., & Hassner, T. (2015). Age and Gender Classification Using Convolutional Neural Networks. In *IEEE Workshop on Analysis and Modeling of Faces and Gestures* (AMFG).
   - Adience Benchmark for Age and Gender Classification, [https://talhassner.github.io/home/publication/2015_CVPR](https://talhassner.github.io/home/publication/2015_CVPR)

3. **Graphviz for System Architecture Diagram**: [https://graphviz.org/](https://graphviz.org/)

4. **Dlib Documentation**:[http://dlib.net/](http://dlib.net/)

5. **DeepFace Library**:[https://github.com/serengil/deepface](https://github.com/serengil/deepface)

6. **Face++ API Documentation**:[https://www.faceplusplus.com/](https://www.faceplusplus.com/)

7. **Tkinter Documentation**:[https://docs.python.org/3/library/tkinter.html](https://docs.python.org/3/library/tkinter.html)


---

### License

This project is licensed under the MIT License - see the LICENSE file for details.

