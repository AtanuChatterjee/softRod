import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


def get_xy_list_from_contour(contours):
    full_dastaset = []
    for contour in contours:
        xy_list = []
        for position in contour:
            [[x, y]] = position
            xy_list.append([x, y])
        full_dastaset.append(xy_list)
    return full_dastaset

def obtain_dilated_contour(frame):
    # Convert the input image to the HSV color space
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the green color range
    green_lower = np.array([45, 100, 50])
    green_upper = np.array([75, 255, 255])

    # Create a mask for the green color range
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # Remove the green background from the image
    removed_green = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(green_mask))

    # Apply a Gaussian blur filter with a kernel size of (3, 3)
    blurred = cv2.GaussianBlur(removed_green, (3, 3), 0)

    # Define the lower and upper bounds of the white color range
    white_lower = np.array([0, 0, 10])
    white_upper = np.array([179, 50, 255])

    # Convert the blurred image to the HSV color space
    imgHSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Apply a color threshold to the HSV image using the lower and upper bounds
    mask = cv2.inRange(imgHSV, white_lower, white_upper)

    # Apply a dilation operation with a 3x3 kernel and nine iterations to fill in gaps and holes in the mask and obtain a more complete binary image.
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(mask, kernel, iterations=9)

    # Find the contours in the dilated image
    contours, _ = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return contours



def obtain_skeleton(frame, i):
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    rgb_lower = np.array([0, 0, 0])
    rgb_upper = np.array([0, 255, 0])
    mask = cv2.inRange(blurred, rgb_lower, rgb_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)
    skel = (closing != 0) * 1
    skeleton_image = skeletonize(skel)
    return skeleton_image


def get_data(skeleton_image):
    yx_coords = np.column_stack(np.where(skeleton_image > 0))
    xy_coords = np.flip(np.column_stack(np.where(skeleton_image > 0)), axis=1)
    x, y = np.array(xy_coords[:, 0]), np.array(xy_coords[:, 1])
    data = x, y
    return data


def main(cap, left, top, right, bottom):
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    count = 0
    currFrame = 0

    contourDict = {}
    skeletonDict = {}

    while True:

        currFrame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        ret, frame = cap.read()

        if frame is None:
            break

        frame = frame[top:bottom, left:right]

        contours = obtain_dilated_contour(frame)
        contour_data = get_xy_list_from_contour(contours)

        try:
            contour = contours[-1]
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
        except IndexError:
            pass
        skeleton = obtain_skeleton(frame, currFrame)
        skeleton_data = get_data(skeleton)

        if contour_data is not None:
            contourDict[currFrame] = contour_data
        else:
            contourDict[currFrame] = None

        if skeleton_data is not None:
            skeletonDict[currFrame] = skeleton_data
        else:
            skeletonDict[currFrame] = None

        if ret == True:
            print("Frame: {}".format(currFrame))
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # if (cv2.waitKey(1) & 0xFF == ord('q')) or (1000 <= currFrame):
                break
        else:
            print("No stream ...")
            break

    cap.release()
    cv2.destroyAllWindows()

    print('saving data ...')

    with open(vidpath + color + type + data + video + '_' + 'contourDict.pkl', 'wb') as f:
        pickle.dump(contourDict, f)

    with open(vidpath + color + type + data + video + '_' + 'skeletonDict.pkl', 'wb') as f:
        pickle.dump(skeletonDict, f)

    print('done!')


if __name__ == '__main__':
    # path = '/Users/atanu/Desktop/soft_pendulum/videos/'
    vidpath = 'Z:/Atanu/exp_2021_fluid_ants/soft_rods'
    color = '/white_rod/'  # 'white'
    type = 'white_5cm_hinged'
    data = '/gif/'
    video = 'S5120007'

    left = 50
    top = 100
    right = 800
    bottom = 850

    cap = cv2.VideoCapture(vidpath + color + type + '/' + video + '.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('FPS', fps)
    main(cap, left, top, right, bottom)
