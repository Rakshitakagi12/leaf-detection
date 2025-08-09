from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW to avoid MSMF issues

# Check if camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera")
    exit()

# Set camera properties for better compatibility
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Check if frame was captured successfully
    if not ret or image is None:
        print("Error: Failed to capture frame")
        continue

    # Check if image is valid before resizing
    if image.size == 0:
        print("Error: Empty image received")
        continue

    # Resize the raw image into (224-height,224-width) pixels
    try:
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print(f"Error resizing image: {e}")
        continue

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
