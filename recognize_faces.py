import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from numpy.__config__ import show
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from train_model import load_data, TEST_SIZE


def main():
    
    filepath = './saved_models/model'
    model = load_model(filepath)

    truth_table = {
        0: "Atom",
        1: "Sanay",
        2: "Aarav"
    }

    cap = cv2.VideoCapture(0)

    truth_table = {
        0: "Atom",
        1: "Sanay",
        2: "Aarav"
    }

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (400, 192)
    fontScale = 1
    fontColor = (255,255,255)
    lineType = 2

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Our operations on the frame come here
        show_img = cv2.resize(frame, (505, 383))
        pred_img = np.append(np.array([]), show_img).reshape((1, 383, 505, 3))

        prediction = model.predict(pred_img)
        person = truth_table[int(np.argmax(prediction, -1)[0])]
        print(person)

        cv2.putText(show_img, person, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)


        # Display the resulting frame
        cv2.imshow('frame', show_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()





if __name__ == "__main__":
    main()
