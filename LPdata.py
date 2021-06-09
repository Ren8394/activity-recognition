import numpy as np
import pandas as pd

def read_label_data(dataFilePath, segFilePath):

    NUM_SENSOR = 6
    data = pd.read_csv(dataFilePath, header=[0,1,2]) # header -> sensor location, sensor type, which axis
    seg = pd.read_csv(segFilePath, header=[0]) 

    """columns: 
    1,2,3 accelerometer x,y,z   - 15,16,17  -(+14) > 14n+2 | 0~6
    4,5,6 gyroscope x,y,z       - 18,19,20  -(+14) > 14n+5 | 0~6
    """
    # Select accerlerometer and gyroscope data
    dataCols = sum([[(14*n+1), (14*n+2), (14*n+3), (14*n+4), (14*n+5), (14*n+6)] for n in range(NUM_SENSOR)], [])
    selectData = data.iloc[:, dataCols]
    # Rename column name
    selectData.columns = [
                    'LT_a_x', 'LT_a_y', 'LT_a_z', 'LT_g_x', 'LT_g_y', 'LT_g_z',
                    'W_a_x', 'W_a_y', 'W_a_z', 'W_g_x', 'W_g_y', 'W_g_z',
                    'LS_a_x', 'LS_a_y', 'LS_a_z', 'LS_g_x', 'LS_g_y', 'LS_g_z',
                    'RS_a_x', 'RS_a_y', 'RS_a_z', 'RS_g_x', 'RS_g_y', 'RS_g_z',
                    'RT_a_x', 'RT_a_y', 'RT_a_z', 'RT_g_x', 'RT_g_y', 'RT_g_z',
                    'C_a_x', 'C_a_y', 'C_a_z', 'C_g_x', 'C_g_y', 'C_g_z']

    # Extract Start point[0] and End Point[1] for all activities
    segSet = tuple([int(seg['start'][row]), int(seg['end'][row])] for row in range(6))
    # Label all points by reference segSet
    # SLR(R) - 1, SLR(L) - 2, SAE(R) - 3, SAE(L) - 4, KE(R) - 5, KE(L) - 6, None of above - 0
    labels = []
    for i in range(len(selectData)):
        if i >= segSet[0][0] and i <= segSet[0][1]:
            labels.append(1)
        elif i >= segSet[1][0] and i <= segSet[1][1]:
            labels.append(2)
        elif i >= segSet[2][0] and i <= segSet[2][1]:
            labels.append(3)
        elif i >= segSet[3][0] and i <= segSet[3][1]:
            labels.append(4)
        elif i >= segSet[4][0] and i <= segSet[4][1]:
            labels.append(5)
        elif i >= segSet[5][0] and i <= segSet[5][1]:
            labels.append(6)
        else:
            labels.append(0)

    # Add LABEL column into selectData
    selectData['LABEL'] = labels

    return selectData

def sliding(df, ls, timeStep=128, overlap=0.5):

    # How many points should be slided
    steps = int(timeStep * overlap)
    windows = []
    windowLabels = []

    for row in range(0, int(df.shape[0] - timeStep), steps):
        window = df.iloc[row : row+timeStep,].values.tolist() # [timeStep, 6*6]
        windowLabel = ls[row : row+timeStep] # [timeStep, 1]

        windows.append(window)
        windowLabels.append(windowLabel)
        
    windows = np.asarray(windows, dtype=np.float32).reshape(-1, 36)
    windowLabels = np.asarray(windowLabels).reshape(-1, 1)

    return windows, windowLabels

if __name__ == '__main__':
    df = pd.DataFrame([
        [1,2,3,4,5],
        [2,3,4,5,6],
        [3,4,5,6,7],
        [4,5,6,7,8],
        [5,6,7,8,9],
        [6,7,8,9,0],
    ])

    ls = [0,10,20,30,40,50,60]