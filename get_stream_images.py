#/usr/bin/python3

import cv2
from time import ctime
import os


video = cv2.VideoCapture(r"rtsp://Atom:Atom1234@192.168.86.28/live", cv2.CAP_FFMPEG)
person = 'atom'
                

def main():
    while True:
        ret,img = video.read()
        time = ctime()
        count = len(os.listdir("data_set/atom/"))
        cv2.imshow('frame', img)
        if count == 10000:
            print(f"\n\nGOAL REACHED\n\n")
            break

        try:
            if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
                break

            if ret:
                cv2.imshow('FRAME', img)
                print(f"writing {time}.png to data_set/{person}/ COUNT: {count}")
                cv2.imwrite(f'data_set/{person}/{time}.png', img)

            else:
                with open('log.txt', 'a') as log:
                    t = ctime()
                    log.write(f"Stream froze at {t}\n")
                break


        except KeyboardInterrupt:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    