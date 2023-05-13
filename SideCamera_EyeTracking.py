import numpy as np
import cv2
import pandas as pd
import os

def createCoordinatesResultsCSV(lst, path):
    # Create data frame from list and save as CSV
    df = pd.DataFrame(lst, columns=['x','y'])
    df.index.name = 'Frames'
    df.to_csv(path, index=True)
    print("Successfully saved CSV to " + path)


def eyecenter(in_path):
    print("Video running " + in_path)
    cap = cv2.VideoCapture(in_path)
    results = []

    while True:
        # Loop through every frame in video

        ret, frame = cap.read()

        if ret == True:
            # Operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),0)
            ret, thresh_img = cv2.threshold(blur,27,255,cv2.THRESH_BINARY)

            contours =  cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
            contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse= True)
            c = contours_sorted[1]

            # MINIMUM ENCLOSING CIRCLE
            (x,y),radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            temp = [int(x),int(y)]
            cv2.circle(frame, center, 2, (0,255,0),2)

            results.append(temp)
            cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Eye midpoint results completed")
    return results

in_path = "E:\\FIGURES_DISTANCEESTPAPER\\SECOND_VERSION\Videos_methods\\to run\\"

out_path = "E:\\FIGURES_DISTANCEESTPAPER\\SECOND_VERSION\Videos_methods\\to run\\"

if not os.path.exists(out_path):
    # If output path doesn't exist create one
    os.makedirs(out_path)

def main():
 for file in os.listdir(in_path):
    # Loop through every file in input path
    file_name = file.split("\\")[-1]
    
    # Result file name is original_name_INTRESULTS.csv
    result_file_name = os.path.splitext(file_name)[0] + "_INTRESULTS.csv"
    result_path = out_path + result_file_name

    # Call eye center function
    results = eyecenter(in_path + file)

    # Call save to CSV function
    createCoordinatesResultsCSV(results, result_path)

if __name__ == "__main__":
    main()