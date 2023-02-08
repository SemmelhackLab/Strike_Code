from filepicker import * #SQ*
from eye_tracker_helpers import *
import pandas as pd
import os
from matplotlib import pyplot as plt
import csv
import copy
import autodetection_tools


def analyseVideo(video, thresh, roi, plot, mouth_roi=None, mouth_thresh=20, frame_range=(0, 100), first_cali_vector=None):
    """
    Main analysis function
    :param video: Video class object (video_handling)
    :param thresh: threshold used to find eyes and swimbladder
    :param roi: crop each frame to ROI (if None then video is not cropped)
    :return: pandas DataFrame (frame number and vergence angles)
    """
    left = []
    right = []
    convergence = []
    bladder_mid = []
    size_sb_list = []
    mouth_area = []

    # if there is no input of frame_range then just use the overall range
    if frame_range == (0, 100):
        frame_range = (0, video.framecount)

    count = 0

    for frame in range(frame_range[0], frame_range[1]):
        count += 1

        img = video.grabFrameN(frame)
        if roi is not None:
            img = cropImage(img, roi)

        """
        frameData() returns centres and angles for the fish axis and both eyes.
        To calculate vergence angles:
            - Angles increase CCW and a converged eye points towards the midline (CCW is counter clockwise)
            - A converged left eye has a greater CCW angle than the body axis
            - CW and CCW are reversed
            - Angles greater than 180 degrees mist be divergent
        """

        c, th, l_c, l_th, r_c, r_th, size_sb = frameData(img, thresh)  # tag
        # Note. th is the orientation/body axis

        if frame == 0:
            first_l_c = l_c
            first_r_c = r_c
            first_mid_eyes = findMidpoint(first_l_c, first_r_c)
            if mouth_roi != None:
                first_c_roi = findMidpoint(mouth_roi[0], mouth_roi[1])
                first_cali_vector = vector(first_mid_eyes, first_c_roi)  # dx = float(b[0]) - float(a[0])

        l_verg = findClockwiseAngle(th, l_th)
        r_verg = findClockwiseAngle(r_th, th)

        if l_verg > math.pi:
            l_verg -= 2 * math.pi
        if r_verg > math.pi:
            r_verg -= 2 * math.pi

        bladder_mid_distance = distance(c, findMidpoint(r_c, l_c))

        # mouth analysis section
        if mouth_roi != None:
            mouth_thresh = mouth_thresh  ########30 is roughly a good one
            p1 = (int(mouth_roi[0][0]), int(mouth_roi[0][1]))
            p2 = (int(mouth_roi[1][0]), int(mouth_roi[1][1]))
            # task. confirm this p1 and p2 is dynamically right
            # Answer. confirmed!

            mouth = img[p1[1]:p2[1] + 1, p1[0]:p2[0] + 1]
            internals = findAllContours(mouth, thresh=mouth_thresh)
            area = cv2.contourArea(internals[0])

        left.append(math.degrees(l_verg))
        right.append(math.degrees(r_verg))
        convergence.append(math.degrees(l_verg) + math.degrees(r_verg))

        bladder_mid.append(bladder_mid_distance)
        size_sb_list.append(size_sb)

        if mouth_roi != None:
            mouth_area.append(area)
        # above are all the list

    # calculate the cr for swim bladder
    list = bladder_mid
    cr_list = []
    list_len = len(list)
    interval = 5
    for i in range(len(list) - interval + 1):
        cr_list.append((list[i + interval - 1] - list[i]) / interval)
    for i in range(len(list) - interval + 1, len(list)):
        cr_list.append(cr_list[len(list) - interval])
        
    sb_cr = cr_list


    df = pd.DataFrame(dict(left=left, right=right, convergence=convergence, bladder_mid=bladder_mid, sb_cr=sb_cr,size_sb=size_sb_list),
                      index=range(frame_range[0], frame_range[1]),
                      columns=['left', 'right', 'convergence', 'bladder_mid', 'sb_cr', 'size_sb'])

    # PLOT THE BLADDER_MIDPOINT_DISTANCE HERE!

    n_size_sb_list = []  # normalized the size of swimbladder before plotting
    for item in size_sb_list:
        n_size_sb_list.append(item / max(size_sb_list) * min(bladder_mid))



    # plt.plot(left)
    # plt.plot(right)
    # plt.show()

    return df


def checkTracking(video, thresh=200, roi=None):
    """
    Check that tracking is working
    :param video: Video class object (video_handling)
    :param thresh: threshold used to find eyes and swimbladder
    :param roi: crop each frame to ROI (if None then video is not cropped)
    :return: None
    """
    winname = video.name
    video.addDisplay(winname, displayFunction=showEyes, displayKwargs={'thresh': thresh, 'roi': roi})  # tag
    cv2.waitKey(0)
    video.removeDisplay(winname)


def dt_csv_to_dict(filename):
    # Note. output is a dictionary with key as the name of videos and corresponding value is a list consists of all striking frames
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        list = [row for row in reader]

    d = dict()
    for i in range(len(list)):
        key = list[i][0]
        value_list = []
        for j in range(1, len(list[i])):
            if list[i][j] != '':
                value_list.append(list[i][j])
        d[key] = value_list


    return d


if __name__ == "__main__":

    ### USER INPUT HERE ###
    ### SELECT A FOLDER TO ANALYZE, FIND ALL AVI FILES IN THAT FOLDER ###
    folder = pickdir()
    filenames = os.listdir(folder)
    avis = [filename for filename in filenames if os.path.splitext(filename)[1] == '.avi']
    print(avis)
    ### CREATE A FOLDER WHERE RESULTS WILL BE SAVED ###
    output_folder = os.path.join(folder, 'results')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ### PROMPTS FOR USING ROI, CHANGING DEFAULT THRESHOLD, CHECK TRACKING FOR EACH VIDEO, PLOTTING DATA ###
    use_roi = 'y'  # raw_input('draw ROI? y/n ')
    # set_threshold = raw_input('set new threshold? y/n ')
    set_threshold = 'y'
    check_tracking = 'y'  # raw_input('check eye tracking for each video? y/n ')
    plot_data = 'n'  # raw_input('plot each trial once analysed? y/n ')

    ### ANALYSIS STARTS HERE (LOOP THROUGH EACH AVI FILE IN SELECTED FOLDER) ###
    counter=0
    thresh1=0
    for avi in avis:
        print(avi)
        Roi=()

        print('current processing is: ', avi)  # tell the user which avi is processing

        ### ROI SELECTION ###
        if use_roi == 'y'and counter ==0:
            print('choose general ROI')
            video = Video(os.path.join(folder, avi))
            # Note. the read shape is right
            ROI = selectROI(video)
            Roi = ROI# tag
            # ROI = ((36, 152), (595, 593))
            print('general ROI is: ', ROI)
        else:
            ROI = ROI

        ### NEW SET MOUTH ROI
        video = Video(os.path.join(folder, avi))
        print('choose the ROI for mouth analysis')


        ### SETTING GENERAL THRESHOLD FOR EYE IDENTIFICATION
        if set_threshold == 'y' and counter == 0:

                video = Video(os.path.join(folder, avi))
                thresh = setThreshold(video, 100, ROI)  # tag
                thresh1=thresh

                tem_data = analyseVideo(video, thresh, ROI, plot='bladder', mouth_roi=None)
                counter = counter + 1
                # # Note. I choose to plot mouth_area in analyseVideo, so plot='no' here because the threshold for mouth is not setted
                # user_input = int(input('do you think the threshold ' + str(thresh) + ' is ok?'))
                # if user_input == 1:  # press 1 to exit the loop, otherwise the program would pick the same avi files and process it
                #     break
        else:
            video = Video(os.path.join(folder, avi))
            thresh = thresh1

            tem_data = analyseVideo(video, thresh, ROI, plot='bladder', mouth_roi= None)  # tag
        # Note. I choose to plot mouth_area in analyseVideo, so plot='no' here because the threshold for mouth is not setted

        ### DISPLAY THE MOUTH ROI (WITH BINARIZATION?)
        ### Task. I should made change here to be able to set the threshold for mice

        # while (1):
        #    print 'Now you are setting the threshold for mouth protrustion analysis'
        #    mouth_thresh = setmouthThreshold(video,33,ROI,mouth_roi,thresh)  #DT-tag
        #    #mouth_thresh = 30
        #    # the last argument 'thresh' here is actually the thresh for eye recognization
        #    data = analyseVideo(video, thresh, ROI, plot='mouth',mouth_roi = mouth_roi, mouth_thresh = mouth_thresh)    #DT-tag
        #    #user_input = int(input('do you think the mouth_threshold ' + str(mouth_thresh) + ' is ok?'))
        #    #if user_input == 1:  # press 1 to exit the loop, otherwise the program would pick the same avi files and process it
        #    break

        ### CREATE FILEPATH TO VIDEO FILE ###
        file_path = os.path.join(folder, avi)
        name = os.path.splitext(avi)[0]
        print(name)
        print(ROI)

        ### IMPROT VIDEO ###
        video = Video(file_path)
        video.name = avi

        ### MAIN ANALYSIS FUNCTION ###
        # data = analyseVideo(video, thresh, ROI, plot='', mouth_roi=mouth_roi, mouth_thresh=mouth_thresh)  # tag


        ### CREATE AN OUTPUT PATH TO SAVE RESULTS ###
        if ROI:
            output_path = os.path.join(output_folder, name + '_realtime_version.csv')
        else:
            output_path = os.path.join(output_folder, name + '_realtime_version.csv')
        ### SAVE RESULTS ###

        print(output_path)
        print(tem_data)
        tem_data.to_csv(output_path)