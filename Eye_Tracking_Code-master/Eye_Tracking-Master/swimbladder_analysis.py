from filepicker import *  # SQ*
from Bimodal_fitting import *
from eye_tracker_helpers import *
import pandas as pd
import os
from matplotlib import pyplot as plt
import csv
import copy
from autodetection_tools import *
from tailfit9 import *
from freq_ampli_tail import *
import os.path
import pickle
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter
import shutil
import glob

tail_amplitude_list=[]
def analyseVideo(video, thresh, roi, plot, mouth_ana = 'n', mouth_roi=None, mouth_thresh=20, frame_range=(0, 100), first_cali_vector=None, one_mm = None):
    """
    Main analysis function
    :param video: Video class object (video_handling)
    :param thresh: threshold used to find eyes and swimbladder
    :param roi: crop each frame to ROI (if None then video is not cropped)
    :return: pandas DataFrame (frame number and vergence angles)
    """

    ### INITIATE VARIABLE ###
    left = []
    right = []
    convergence = []
    bladder_mid = []
    size_sb_list = []

    if mouth_ana == 'y':
       mouth_area = []

    # if there is no input of frame_range then just use the overall range
    if frame_range == (0, 100):
        frame_range = (0, video.framecount)

    ### ANALYSIS FRAMES BY FRAMES ###
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
        if mouth_ana == 'y' and mouth_roi != None:
            mouth_thresh = mouth_thresh  ########30 is roughly a good one
            p1 = (int(mouth_roi[0][0]), int(mouth_roi[0][1]))
            p2 = (int(mouth_roi[1][0]), int(mouth_roi[1][1]))
            # task. confirm this p1 and p2 is dynamically right
            # Answer. confirmed!
            mouth = img[p1[1]:p2[1] + 1, p1[0]:p2[0] + 1]
            internals = findAllContours(mouth, thresh=mouth_thresh)
            area = cv2.contourArea(internals[0])
            mouth_area.append(area)

        left.append(math.degrees(l_verg))
        right.append(math.degrees(r_verg))
        convergence.append(math.degrees(l_verg) + math.degrees(r_verg))
        bladder_mid.append(bladder_mid_distance)
        size_sb_list.append(size_sb)
        # above are all the list

    # calculate the cr for swim bladder
    list = bladder_mid
    cr_list = []
    list_len = len(list)
    interval = 5   #[PARAMETER]
    for i in range(len(list) - interval + 1):
        cr_list.append((list[i + interval - 1] - list[i]) / interval)
    for i in range(len(list) - interval + 1, len(list)):
        cr_list.append(cr_list[len(list) - interval])
    sb_cr = cr_list

    # calculate the cr for mouth protrusion
    if mouth_ana == 'y':
        list = mouth_area
        cr_list = []
        list_len = len(list)
        interval = 5
        for i in range(len(list) - interval + 1):
            cr_list.append((list[i + interval - 1] - list[i]) / interval)
        for i in range(len(list) - interval + 1, len(list)):
            cr_list.append(cr_list[len(list) - interval])
        mouth_cr = cr_list

    # calculate the real_distance unit for bladder movement and bladder cr
    if one_mm != None:
        bladder_mid_um = []
        sb_cr_um_ms = []
        for item in bladder_mid:
            bladder_mid_um.append(item*1000/one_mm)
        for item in sb_cr:
            coefficient = float(1000.0/float(one_mm))*float(300.0/1000.0)
            item_new = item*coefficient
            sb_cr_um_ms.append(item_new)
    else:
        bladder_mid_um = left
        sb_cr_um_ms = right

    ### PLOT SESSION ###

    # bladder_mid_um = gaussian_filter(bladder_mid_um, sigma=3, mode='nearest')
    # sb_cr_um_ms = gaussian_filter(sb_cr_um_ms, sigma=3, mode='nearest')

    if plot == 'eye':
        plt.plot(left, 'g')
        plt.plot(right, 'b')
        plt.plot(convergence, 'r')
        plt.show()

    if plot == 'bladder':
        plt.subplot(211)
        plt.plot(bladder_mid_um, 'b')
        plt.subplot(212)
        plt.plot(sb_cr_um_ms, 'g')
        plt.show()

    if mouth_ana == 'y' and plot == 'mouth':
        plt.plot(mouth_area, 'b')  # mouth_area is plotted here
        # plt.plot(n_size_sb_list, 'g')
        if ROI:
            plt.savefig(video.name + "_t_" + str(thresh) + '_ROI_' + str(ROI) + '_mthresh_' + str(
                mouth_thresh) + '_mroi_' + str(mouth_roi) +
                        '_range_' + str(frame_range) + '_static_mouth.png')
        else:
            plt.savefig(video.name + "_t_" + str(thresh) + '_mthresh_' + str(mouth_thresh) + '_mroi_' + str(
                mouth_roi) + '_range_' + str(frame_range) + '_static_mouth.png')

        plt.show()


    ### FINAL OUTPUT HERE ###
    if mouth_ana == 'y':
        df = pd.DataFrame(dict(left=left, right=right, convergence=convergence, bladder_mid_um=bladder_mid_um, bladder_mid=bladder_mid, sb_cr_um_ms=sb_cr_um_ms, sb_cr=sb_cr,size_sb=size_sb_list, mouth_area=mouth_area, mouth_cr=mouth_cr),
                      index=range(frame_range[0], frame_range[1]),columns=['left', 'right', 'convergence', 'bladder_mid_um','bladder_mid', 'sb_cr_um_ms', 'sb_cr', 'size_sb', 'mouth_area','mouth_cr'])
    else:
        df = pd.DataFrame(dict(left=left, right=right, convergence=convergence, bladder_mid_um=bladder_mid_um, bladder_mid=bladder_mid, sb_cr_um_ms=sb_cr_um_ms, sb_cr=sb_cr,size_sb=size_sb_list),
                          index=range(frame_range[0], frame_range[1]),columns=['left', 'right', 'convergence', 'bladder_mid_um', 'bladder_mid', 'sb_cr_um_ms', 'sb_cr', 'size_sb'])
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

if __name__ == "__main__":

    ### PART-1. USER INPUT HERE ###

    ### 1-1. INPUT BASIC PARAMETERS FOR PLOT ###

 main_dir = pickdir()
 directory_contents = os.listdir(main_dir)

 list_folder=[]
 for i in range(0,len(directory_contents)):
           list_folder.append(main_dir  + '/' + directory_contents[i])

 ROI = None # set the general ROI for eye and swim bladder analysis, setted like ((36, 152), (595, 593)) or None
 eye_thresh = None # setted like 200 or None
 bladder_thresh = 50 # setted like 200 or None
 tail_fitting = 'n' # y for doing the tailfitting
 tail_startpoint = None
 one_mm = 113.0 # 1mm = x pixel, should be a float!
 mouth_ana = 'n'
 scale = 0.75 # scaling for tail-fitting
 count=0
    #
    # ### 1-2. INPUT THRESH PARAMETERS FOR ANALYSIS ###
    #
 print(list_folder)
 list_convergence_threshhold= {}
 for a in range(0,len(list_folder)):
    # # READ ALL THE AVIS FILES WITHIN THAT FOLDER
    folder=list_folder[a]
    print(folder)
    filenames = os.listdir(folder)


    avis = [filename for filename in filenames if os.path.splitext(filename)[1] == '.avi']
    listpkl = [filename for filename in filenames if os.path.splitext(filename)[1] == '.pkl']
    print(listpkl)

    if len(listpkl) != 0:
        flag = 1
    else:
        flag =0

    if flag == 1:

        folder_pickle = os.path.join(folder, 'Pickle_File')
        if not os.path.exists(folder_pickle):
            os.makedirs(folder_pickle)

        for a in range(0,len(listpkl)):
            shutil.copy(folder + "\\" + listpkl[a], folder_pickle + "\\" + listpkl[a])



        match = (re.search("([0-9]{8})", folder))
        date= match.group()
        print(folder)
        fish_number = folder.split("\\")[len(folder.split("\\")) - 1]

        fish_data= fish_number + '_' + date

        folder_bimodal = os.path.join(folder, 'BiModal_Dist')
        if not os.path.exists(folder_bimodal):
             os.makedirs(folder_bimodal)

        result = (pd.concat([pd.read_pickle(folder_pickle + '\\' + x) for x in os.listdir(folder_pickle)]))

        df_data = result


        left = df_data['LeftEye']
        left_x = df_data['LeftEye_x']
        left_y = df_data['LeftEye_y']

        right_x = df_data['RightEye_x']
        right_y = df_data['RightEye_y']

        right = df_data['RightEye']
        convergence = df_data['Convergence']

        ob1 = EyeConvergenceAnalyser(convergence, folder_bimodal)
        ob1.kernel_density_estimation()
        ob1.find_convergence_threshold()
        ob1.calculate_convergence_score()
        ob1.plot_histogram()
        ob1.plot_kernel_density_estimation()
        ob1.plot_threshold()
        print(ob1.threshold)
        plt.close('all')

        xval = ob1.threshold

        list_convergence_threshhold[fish_data] = xval

        count = 0
        # if xval < 30:
        #     xval = 40
        # print(xval)

    #CREATE A FOLDER WHERE RESULTS WILL BE SAVED


    output_folder = os.path.join(folder, 'original_plot')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder=folder

     ### PART-2. SET THRESHOLD, PLOT DATA AND SAVE ###
    ### ANALYSIS STARTS HERE (LOOP THROUGH EACH AVI FILE IN SELECTED FOLDER) ###
    for avi in avis:

        count=count+1
        print '*****************************************************************************************************************************************************************************************************************************************'
        print 'current processing is: ', avi  # tell the user which avi is processing

        ### 2-1. GENERAL ROI SELECTION ###

        if flag == 0:
         if ROI == None:
            print 'choose general ROI'
            video = Video(os.path.join(folder, avi))
            # Note. the read shape is right
            ROI = selectROI(video)  # tag
            print 'general ROI is: ', ROI

        ### 2-3. SETTING BLADDER_THRESHOLD, PLOT BLADDER MOVEMENT IN BOTH PIXEL/FRAME AND UM/MS###

         # if bladder_thresh == None:
         #    while (1):
         #        print 'setting threshold for swim bladder'
         #        video = Video(os.path.join(folder, avi))
         #        bladder_thresh = setThreshold(video, 180, ROI)  # tag
         #        tem_data = analyseVideo(video, bladder_thresh, ROI, plot='bladder', one_mm=one_mm)
         #
         #        user_input = int(input('do you think the swim bladder threshold ' + str(bladder_thresh) + ' is ok?'))
         #        if user_input == 1:  # press 1 to exit the loop, otherwise the program would pick the same avi files and process it
         #            break
         # else:
         #    video = Video(os.path.join(folder, avi))
         #    tem_data = analyseVideo(video, bladder_thresh, ROI, plot='no', one_mm=one_mm)
         #
         # bladder_mid_um = tem_data['bladder_mid_um']
         # bladder_mid = tem_data['bladder_mid']
         # sb_cr_um_ms = tem_data['sb_cr_um_ms']
         # sb_cr = tem_data['sb_cr']
         # size_sb_list = tem_data['size_sb']

        ### 2-2-1. SETTING EYE_THRESHOLD, PLOT EYE MOVEMENT ###

        if flag == 0:
         if eye_thresh == None:
            while (1):
                print 'setting threshold for left eye, left eye are green'
                video = Video(os.path.join(folder, avi))
                eye_thresh = setThreshold(video, 180, ROI)
                tem_data = analyseVideo(video, eye_thresh, ROI, plot='eye')
                user_input = int(input('do you think the eye threshold ' + str(eye_thresh) + ' is ok?'))
                if user_input == 1:  # press 1 to exit the loop, otherwise the program would pick the same avi files and process it
                    break
         # else:
         #    video = Video(os.path.join(folder, avi))
         #    tem_data = analyseVideo(video, eye_thresh, ROI, plot='no')

         left = tem_data['left']
         right = tem_data['right']
         convergence = tem_data['convergence']

        if flag == 1:

            result_pickle = os.path.join(folder,listpkl[count -1])
            #result_final_pickle =str(result_pickle).replace(".avi.h5.pkl","")
            df_data = pickle.load(open(result_pickle, 'rb'))

            tail_amplitude_list = df_data['TailTip']
            swimbladder_left=df_data['swimbladder_left']
            swimbladder_right=df_data['swimbladder_right']


            left = df_data['LeftEye']
            left_x = df_data['LeftEye_x']
            left_y = df_data['LeftEye_y']

            right_x = df_data['RightEye_x']
            right_y = df_data['RightEye_y']

            right = df_data['RightEye']
            convergence = df_data['Convergence']

            lefteyecord =(zip(left_x,left_y))
            righeeyecord=(zip(right_x,right_y))
            swimbladdercord=(zip(swimbladder_left,swimbladder_right))
            eyemidx=[]
            eyemidy=[]
            for i in range(0,len(lefteyecord)):
                eyemidx.append((left_x[i] + right_x[i])/2)
                eyemidy.append((right_y[i] + left_y[i])/2)


            eyemidcord=zip(eyemidy,eyemidx)

            bladder_mid=[]
            for i in range(0, len(swimbladdercord)):
                bladder_mid.append(euclidean(eyemidcord[i],swimbladdercord[i]))

            size_sb_list=[]
            size_sb=[]
            size_sb_list.append(size_sb)
            list = bladder_mid
            cr_list = []
            list_len = len(list)
            interval = 5  # [PARAMETER]
            for i in range(len(list) - interval + 1):
                cr_list.append((list[i + interval - 1] - list[i]) / interval)
            for i in range(len(list) - interval + 1, len(list)):
                cr_list.append(cr_list[len(list) - interval])
            sb_cr = cr_list


            # calculate the real_distance unit for bladder movement and bladder cr
            if one_mm != None:
                bladder_mid_um = []
                sb_cr_um_ms = []
                for item in bladder_mid:
                    bladder_mid_um.append(item * 1000 / one_mm)
                for item in sb_cr:
                    coefficient = float(1000.0 / float(one_mm)) * float(300.0 / 1000.0)
                    item_new = item * coefficient
                    sb_cr_um_ms.append(item_new)
            else:
                bladder_mid_um = left
                sb_cr_um_ms = right


            # bladder_mid_um = gaussian_filter(bladder_mid_um, sigma=3, mode='nearest')
            # sb_cr_um_ms = gaussian_filter(sb_cr_um_ms, sigma=3, mode='nearest')
            plt.style.use('seaborn-white')
            plt.plot(left, 'g')
            plt.plot(right, 'b')
            plt.plot(convergence, 'r')
            #plt.show()


            plt.subplot(211)
            plt.plot(bladder_mid_um, 'b')
            plt.subplot(212)
            plt.plot(sb_cr_um_ms, 'g')
            #plt.show()




        ### 2-4. SETTING TAIL STARTING POINT, SAVE TO SHV ###
        if tail_fitting == 'y':
            # display = askyesno(text='Display frames?')
            display = True
            displayonlyfirst = True
            'TAIL FITTTING'
            video_path = str(folder + '\\' + avi)
            if str(type(tail_startpoint)) == "<type 'NoneType'>":
                # you can either set the startpoint or process the same batch of videos with the startpoint setted in the first videos
                tail_startpoint, tail_amplitude_list, boutedges_thresh = tailfit_batch([video_path], display,
                                                                                       displayonlyfirst,
                                                                                       shelve_path=folder + '\\original_plot\\' + avi + '_tail.shv',
                                                                                       reuse_startpoint=True,
                                                                                       scale=scale)
                # tail_ampitude_list actually stores the actual value for tail movement
            else:
                display = False
                tail_startpoint, tail_amplitude_list, boutedges_thresh = tailfit_batch([video_path], display,
                                                                                       displayonlyfirst,
                                                                                       shelve_path=folder + '\\original_plot\\' + avi + '_tail.shv',
                                                                                       reuse_startpoint=True,
                                                                                       tail_startpoint=tail_startpoint,
                                                                                       scale=scale)
                # the corresponding shv files will be saved to the same folder as the tailfit
                # tail_ampitude_list actually stores the actual value for tail movement



        ### 2-5. SAVE EYE/BLADDER/TAIL ORIGINAL DATA ###


        ### add the tail_angle  data here

        data = pd.DataFrame(dict(left=left, right=right, convergence=convergence, bladder_mid_um=bladder_mid_um, bladder_mid=bladder_mid, sb_cr_um_ms=sb_cr_um_ms,
                               sb_cr=sb_cr,size_sb=size_sb_list, tail_amplitude_list=tail_amplitude_list), index=range(0, len(left)), columns=['left', 'right', 'convergence', 'bladder_mid_um', 'bladder_mid','sb_cr_um_ms', 'sb_cr', 'size_sb', 'tail_amplitude_list'])

        csv_name = avi + '_plot.csv'
        output_path = os.path.join(output_folder, csv_name)
        data.to_csv(output_path)
        print '+++++++++++++++++++++++++++++++++FINISH PLOTTING, START ANALYSIS+++++++++++++++++++++++++++++++++++++++++++'


        ### PART-3. ANALYSIS ###

        ### 3-1. DETERMINE THE TAIL-BOUTS BASED ON TAIL-RELATED THRESHOLD ###

        ### 3-2. DETERMINE STRIKE CANDIDATES IN TAIL BOUTS ###

        ### 3-3. DETERMINE EYE MOVEMENT ###

        ### 3-4. FURTHER EXCLUDE NON-STRIKE CANDIDATES ###

        threshval = .30  # this threshold is used to detect the bout, the higher the threshold, the less or shorter the bout would be
        Fs = 300
        peakthres = 4
        coefficient = float(1000.0/float(one_mm))*float(300.0/1000.0)
        strike_thresh_02 = 0.2*coefficient #for previous way to calculate the threshold
        strike_thresh = 0.65 # unit um/ms
        tail_thresh = 70 #if tail_amplitude bigger than this then exclude it
        print '[STRIKE THRESH]', strike_thresh, 'um/ms'

        max_amplitude_list = []
        #freq = tailbeatfreq(output_folder, threshval, Fs, peakthres, shv_file=avi + '_tail.shv')
        # # freq[0][n] contains the frequencies of the nth bout
        # # freq[1][n] contains the angles of the nth bout
        # # freq[2][n] contains the x,y coordinates of each peak of the nth bout
        # # freq[3][n] contains the framerange of each bouts
        # # freq[4][n] contains the name of the nth video within that shvs (shvname_list)

        # for i, frequency in enumerate(freq[0]):
        #     print 'Mean tail bend frequency of ', i, 'th bout are: ', frequency, 'Hz'
        #     max_amplitude = max([abs(x) for x in freq[1][i]])
        #     max_amplitude_list.append(max_amplitude)
        #     print 'Max tail bend angle(abs value) within the ', i, 'th bout are: ', max_amplitude, 'degree'
        #     framerange = range(freq[3][i][0], freq[3][i][1])  # construct the framerange of each bouts and use it as x for plotting

        ### CREATE FILEPATH TO VIDEO FILE ###
        file_path = os.path.join(folder, avi)
        name = os.path.splitext(avi)[0]

        ### IMPROT VIDEO ###
        video = Video(file_path)

        ### MAIN ANALYSIS FUNCTION ###
        striking_candidates, original_striking_candidates = striking_detection(sb_cr_um_ms, convergence, amplitude_thresh=strike_thresh,binocular_thresh=xval)  # set the threshold here
        print 'striking candidates before exclusion are: ', original_striking_candidates
        print 'striking candidates before tail analysis are: ', striking_candidates
        # striking_candidates should be a list!

        # to store them into para_data.csv
        para_original_striking_candidates = [original_striking_candidates]
        para_striking_candidates = [striking_candidates]

        #striking_candidates = tail_analysis(striking_candidates, freq, tail_thresh = tail_thresh)   #############key!!!
        #print 'final strike candidates are: ', striking_candidates

        # to store them into para_data.csv
        para_tail_striking_candidates = [striking_candidates]

        if mouth_ana == 'y':
           plot_strike_candidates(convergence, bladder_mid_um, sb_cr_um_ms,tail_amplitude_list, striking_candidates, mouth_area = data['mouth_area'], video=video, mouth_ana=mouth_ana)
        else:
            plot_strike_candidates(convergence, bladder_mid_um, sb_cr_um_ms, tail_amplitude_list, striking_candidates, binocular_thresh=xval,video=video)
            # def plot_strike_candidates(binocular, sb_mid, sb_cr, tail_amplitude_list, striking_candidates, mouth_area = None, video = None, mouth_ana = '', annotation_list = [], general_thresh = 0):

        para_data =  pd.DataFrame({'final_striking':para_tail_striking_candidates,'striking':para_striking_candidates,'original_striking':para_original_striking_candidates,
                                   'bout_max_amplitude':[max_amplitude_list],'ROI':[ROI],'eye_thresh':[eye_thresh],
                                  'bladder_thresh':[bladder_thresh],'tail_startpoint':[tail_startpoint],'scale':scale,'one_mm':one_mm,'strike_thresh':strike_thresh, 'strike_thresh_02':strike_thresh_02,
                                  'tail_thresh':tail_thresh,'threshval':threshval,'Fs':Fs,'peakthres':peakthres,'coefficient':coefficient})
        if striking_candidates != []:
            para_data.to_csv(output_folder + '\\' + avi + '_para_SD.csv')
        else:
            para_data.to_csv(output_folder+'\\'+avi+'_para.csv')

 pickle.dump(list_convergence_threshhold, open(main_dir + 'list_convergence_threshhold.pkl', 'wb'),protocol=2)