"""
Testing how to face align a simple image:
source: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
"""

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

import imutils
import dlib
import cv2

predictor_path = "/home/uc3m1/PycharmProjects/siameseFaceNet/data/align/shape_predictor_68_face_landmarks.dat"
image_path = "/home/uc3m1/PycharmProjects/siameseFaceNet/data/WhatsApp Image 2018-06-28 at 12.12.18.jpeg"


def main():
    # Initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    fa = FaceAligner(predictor, desiredFaceWidth=182)

    # Load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Show the original input image and detect faces in the grayscale image
    cv2.imshow("Input", image)
    rects = detector(gray, 2)

    # Loop over the face detections
    for rect in rects:
        # Extract the ROI of the *original* face, then align the face using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        face_orig = imutils.resize(image[y:y + h, x:x + w], width=256)
        face_aligned = fa.align(image, gray, rect)

        # Display the output images
        cv2.imshow("Original", face_orig)
        cv2.imshow("Aligned", face_aligned)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
