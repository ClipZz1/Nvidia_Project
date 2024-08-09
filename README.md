# Nvidia_Project

 This project is an AI-based chatbot that detects hand gestures—specifically, thumbs up, thumbs down, or neutral—to assess your emotional state. It aims to provide a simple and intuitive interface where the bot engages with users by asking how they are feeling and responding based on the detected gesture. This project leverages NVIDIA’s Jetson Nano for real-time gesture detection using a trained AI model. > 

![add image descrition here](direct image link here)

## The Algorithm

The core of this project involves a machine learning model trained to classify hand gestures: thumbs up, thumbs down, and neutral. The model is built using Teachable Machine and then exported for use on the Jetson Nano.

- Gesture Detection: The model processes input from a camera, analyzing each frame to determine the presence of the specified gestures.
- Classification: Depending on the detected gesture, the AI bot interprets the user’s emotional state. A thumbs-up indicates positivity, 
  thumbs-down indicates negativity, and neutral is neither, and implies no response to the question.
 - Interaction: The chatbot uses the detected gesture to ask questions, offer support, or provide responses tailored to the user’s 
   emotional state based on how they answered the question with their thumb.

## Running this project

# Dependencies include:
- Python 3.0
- TensorFlow or TensorFlow Lite
- OpenCV for image processing
- Jetson Nano-specific libraries for handling hardware acceleration
- PyTorch
- Webcam

# Steps

1. - Clone the repository: git clone https://github.com/yourusername/Nvidia_Project.git
   - cd Nvidia_Project
2. Install required libraries:
   - pip install tensorflow opencv-python
   - pip install torch torchvision torchaudio
3. - Run the Python script: python3 gesture_chatbot.py
4. Interaction: Ensure your camera is set up correctly and interact with the bot using hand gestures.
5. Deploy on Jetson Nano: Follow Jetson Nano-specific setup for optimal performance.

[View a video explanation here](video link)
