import cv2
import pandas as pd
import numpy as np

def padWithZeros(X, margin):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def PDoF(X, y, windowSize,target_window=1,gamma=1, removeZeroLabels=True,PDoF = True):
    # If use PDoF，set PDoF = True
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))

    x1, x2, x3 = X.shape
    Y = np.zeros([2, x1, x2, x3])
    Y[1] = X
    if PDoF:
        Y[0] = cv2.blur(X, (gamma,gamma))
    else:
        Y[0] = X
    zeroPaddedX = []
    for layer in range(2):
        pad = [int((windowSize - 1) / 2),int((target_window - 1)/2)]
        margin = pad[layer]
        zeroPaddedX.append(padWithZeros(Y[layer], margin=margin))
        patchIndex = 0
        
        if layer ==0:
            for r in range(margin, zeroPaddedX[layer].shape[0] - margin):
                for c in range(margin, zeroPaddedX[layer].shape[1] - margin):
                    patch = zeroPaddedX[layer][r - margin:r + margin + 1, c - margin:c + margin + 1]
                    patchesData[patchIndex, :,:, :] = patch
                    patchesLabels[patchIndex] = y[r - margin, c - margin]
                    patchIndex = patchIndex + 1
        else:
            for r in range(margin, zeroPaddedX[layer].shape[0] - margin):
                for c in range(margin, zeroPaddedX[layer].shape[1] - margin):
                    patch = zeroPaddedX[layer][r - margin:r + margin + 1, c - margin:c + margin + 1]
                    patchesData[patchIndex, int((windowSize - 1) / 2)-int((target_window - 1)/2):int((windowSize - 1) / 2)+int((target_window - 1)/2)+1, int((windowSize - 1) / 2)-int((target_window - 1)/2):int((windowSize - 1) / 2)+int((target_window - 1)/2)+1, :] = patch
                    patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    return patchesData, patchesLabels