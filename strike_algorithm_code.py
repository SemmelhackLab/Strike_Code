import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import seaborn as sns
from scipy import stats

# specify the video path
import os
import re
from filepicker import *
import pickle
import imutils

def init_condition(avis):
    capture = cv.VideoCapture(avis)
    _, frame1 = capture.read()
    num_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    # Create mask
    hsv = np.zeros_like(frame1)
    # Make image saturation to a maximum value
    hsv[..., 1] = 255
    return capture, num_frames,hsv


def calculate_movement(capture, num_frames, hsv):
    mag1 = []
    ang1 = []
    strike_frame = []
    mag_final = []
    arg_final = []
    mseval=[]


    for i in range(1, num_frames):

      if i < (num_frames) -1 :

        pre = capture.set(1, i - 1)
        _,pre = capture.read()
        pre_final = cv.cvtColor(pre, cv.COLOR_BGR2GRAY)

        cur = capture.set(1, i)
        _,cur=capture.read()
        cur_final = cv.cvtColor(cur, cv.COLOR_BGR2GRAY)

        h, w = cur_final.shape
        diff = cv.subtract(cur_final, pre_final)
        err = np.sum(diff)
        mse = err / (float(h * w))
        theta = mse
        thresh1=theta
        mseval.append(mse)

    res=[]
    print(mseval)
    stdtest= np.mean(mseval) + 3 * np.std(mseval)
    print(stdtest)

    for a in range(0,len(mseval)):
        if mseval[a] > stdtest:
            res.append(a)


    sns.distplot(mseval)
    print(res)
    print(len(res))
    plt.show()
    plt.close()

    for k in range(0, len(res)):
        param = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 50,
            'iterations': 1,
            'poly_n': 5,
            'poly_sigma': 1.5,
            'flags': cv.OPTFLOW_FARNEBACK_GAUSSIAN
            # 'flags': cv.OPTFLOW_LK_GET_MIN_EIGENVALS
        }
        if res[k] < (num_frames) - 1:
            pre = capture.set(1, res[k] - 1)
            _, pre = capture.read()
            pre_final = cv.cvtColor(pre, cv.COLOR_BGR2GRAY)
            pre_final = cv.GaussianBlur(pre_final, (21, 21), 0)

            cur = capture.set(1, res[k])
            _, cur = capture.read()

            cur_final = cv.cvtColor(cur, cv.COLOR_BGR2GRAY)
            cur_final = cv.GaussianBlur(cur_final, (21, 21), 0)


            flow = cv.calcOpticalFlowFarneback(pre_final,cur_final, None, **param)
            mag1, ang1 = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)


            print(res[k])
            move_sense = ang1[mag1 > stdtest]

            move_mode = (stats.mode(move_sense)[0])
            if move_mode != 0:

                    move_mode = float(move_mode)
            else:
                    move_mode = 0.0

            print(move_mode)
            loc=4

            if 60 <= move_mode <= 120:
                    loc=0


            elif 150 <= move_mode <= 210:
                    loc=1

            elif 240 <= move_mode <= 300:
                    loc =2


            elif 330 <= move_mode or move_mode <= 30:
                   loc=3

            else:
                    loc=4

            ang_180 = ang1 / 2


            if loc == 0 :
                    text = 'Moving down'
            elif loc == 1:
                    text = 'Moving to the right'
            elif loc == 2:
                     text = 'Moving up'
            elif loc == 3:
                    text = 'Moving to the left'
            else:
                    text = 'No Movement'

            if loc == 2 :
                    print(res[k])
                    print(loc)
                    print(thresh1)

                    strike_frame.append(res[k])
                    mag_final.append(np.mean(mag1))
                    arg_final.append(move_mode)

            else:
                    print(res[k])
                    arg_final.append(0)
                    mag_final.append(0)

        hsv[:, :, 0] = ang_180
        hsv[:, :, 2] = cv.normalize(mag1, None, 0, 255, cv.NORM_MINMAX)
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        if text == 'Moving up':

            colr = (0, 0, 255)
        else:
            colr = (0, 0, 0)

        cv.putText(cur, 'Frame Number ' + ': ' + str(res[k]), (30, 50), cv.FONT_HERSHEY_COMPLEX, cur.shape[1] / 500,
                   colr, 2)
        cv.putText(cur, text, (30, 80), cv.FONT_HERSHEY_COMPLEX, cur.shape[1] / 500, colr, 2)

        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break
        cv.imshow('Frame', cur)


    cv.destroyAllWindows()
    return mag_final, arg_final, strike_frame

def plot(mag_final, arg_final, strike_frame, filename):
    print(strike_frame)


def main():
    main_dir = pickdir()
    directory_contents = os.listdir(main_dir)

    list_folder = []
    for i in range(0, len(directory_contents)):
        list_folder.append(main_dir + '/' + directory_contents[i])

    for a in range(0, len(list_folder)):
        # # READ ALL THE AVIS FILES WITHIN THAT FOLDER
        folder = list_folder[a]
        filenames = os.listdir(folder)
        print(filenames)

        avis = [filename for filename in filenames if re.search(".avi",os.path.splitext(filename)[1])]
        for k in range(0,len(avis)):
         if len(avis) != 0:

            capture, num_frames, hsv = init_condition(folder + '\\' + avis[k])
            filename1 = folder + '\\' + avis[k]
            filename = filename1.replace('.avi', "")

            print(filename1)
            mag_final, arg_final, strike_frame = calculate_movement(capture, num_frames, hsv)
            plot(mag_final, arg_final, strike_frame, filename)


if __name__ == "__main__":
    main()


