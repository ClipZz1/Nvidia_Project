# Nvidia_Project

This project is an AI-based Gesture-Based Survey/Feedback System that detects hand gestures—specifically, thumbs up, thumbs down, or neutral—to gather feedback. It helps people take surveys with just putting their thumb up or down, running the simple interface. It provides a simple and intuitive interface where users can give feedback by using hand gestures. This project leverages NVIDIA’s Jetson Nano for real-time gesture detection using a trained AI model.

![what it should look like](direct image link here)

## The Algorithm

The core of this project involves a machine learning model trained to classify hand gestures: thumbs up, thumbs down, and neutral. The model is built using Teachable Machine and then exported for use on the Jetson Nano. We then retrained the model from the pictures within it and had a functioning model to use

- Gesture Detection: The model processes input from a camera, analyzing each frame to determine the presence of the specified gestures.
- Classification: Depending on the detected gesture, the system records feedback. A thumbs-up indicates positive feedback, thumbs-down indicates negative feedback, and neutral implies no response to the feedback request.
- Interaction: The system uses the detected gesture to record feedback for various questions or prompts, providing a simple way for users to respond.

## Running this project

# Dependencies include:
- Python 3.6 or higher
- TensorFlow or TensorFlow Lite
- OpenCV for image processing
- PyTorch
- Jetson Nano-specific libraries for handling hardware acceleration
- Webcam
- Model

# Steps

1. Install the model from this google drive link or the one above "model.txt"
   - 
2. - Clone the repository: git clone https://github.com/yourusername/Nvidia_Project.git
   - cd Nvidia_Project
   - Add 'resnet18.onnx' to the folder with it
3. Install required libraries:
- pip install tensorflow opencv-python
- pip install torch torchvision torchaudio
4. Set the model directory
5. - Run the Python script: python3 project.py
6. Interaction: Ensure your camera is set up correctly and interact with the bot using hand gestures.
7. Deploy on Jetson Nano: Follow Jetson Nano-specific setup for optimal performance.

[View a video explanation here](https://youtu.be/qsvBHe-yud4)
