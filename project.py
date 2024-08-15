import cv2
import numpy as np
import onnxruntime as ort
import time

model_path = "python/training/classification/models/saved_model/resnet18.onnx"
session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) 
    img = np.transpose(img, (2, 0, 1))  
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  
    return img

def get_prediction(image):
    processed_image = preprocess(image)
    outputs = session.run([output_name], {input_name: processed_image})
    prediction = np.argmax(outputs[0])
    return prediction

def wait_for_valid_response(camera):
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture image.")
            continue
        prediction = get_prediction(frame)
        
        if prediction == 2: 
            return "Yes"
        elif prediction == 1: 
            return "No"
        else:
            print("Neutral detected. Waiting for Thumbs Up or Thumbs Down...")
            time.sleep(1)

def main():
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Failed to open camera.")
            return
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return
    
    questions = [
        "Are you satisfied with our service?",
        "Did you find what you were looking for?",
        "Would you recommend us to others?"
    ]

    for question in questions:
        print(f"{question} (Thumbs Up for Yes, Thumbs Down for No)")
        print("Waiting for your response...")
        time.sleep(3) 
        response = wait_for_valid_response(camera)
        print(f"You responded: {response}")
        print("Do you want to continue? (Thumbs Up for Yes, Thumbs Down for No)")
        time.sleep(3)  
        continue_response = wait_for_valid_response(camera)

        if continue_response == "No": 
            print("Thank you for your feedback!")
            break

    print("Survey completed. Thank you!")
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
