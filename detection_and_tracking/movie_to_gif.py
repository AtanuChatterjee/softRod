import imageio
import cv2

# open video file
cap = cv2.VideoCapture('Z:/Atanu/exp_2021_fluid_ants/soft_rods/white_rod/white_5cm_hinged/S5120006.MP4')

# set start and end frames
start_frame = 32500
end_frame = 32800

# set current frame to start frame
currFrame = start_frame

# set crop parameters
left = 50
top = 100
right = 800
bottom = 850
# create output file name
output_filename = 'Z:/Atanu/exp_2021_fluid_ants/soft_rods/white_rod/white_5cm_hinged/S5120006.gif'

# create list of frames to convert
frames = []
for currFrame in range(start_frame, end_frame + 1):
    ret, frame = cap.read()
    if not ret:
        break
    # crop frame
    cropped_frame = frame[top:bottom, left:right]
    frames.append(cropped_frame)
    currFrame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

# convert frames to GIF
imageio.mimsave(output_filename, frames, fps=30)

# release video capture and close windows
cap.release()
cv2.destroyAllWindows()



