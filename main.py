import cv2 as cv
import numpy as np

print('xxxxxxxx')
if __name__ == "__main__":
    print("aaaaaa")
    o = cv.imread("test.png")
    gray = cv.cvtColor(o, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    binary = cv.bitwise_not(binary)
    contours, hiberachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print("contours", len(contours))
    for contour in contours:
        pass #print(contour)
    o = cv.drawContours(o, contours, -1, (0, 0, 255), 1)

    #cv.imshow("t", binary)
    cv.imshow("t", o)
    print("xxxxxxx")
    cv.waitKey(0)
    print("yyyyyyyy")
    #cv.destroyAllWindows()
    #cv.waitKey(1)

    exit()
