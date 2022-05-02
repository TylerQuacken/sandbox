from aifc import Aifc_read
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from IPython import embed

from pathlib import Path

A1_DEPTH = 2.704
A5_DEPTH = -7.746
pointList = ["A3", "A5"]
thresholds = [[100, 1000], [500, 1500]]

thetas = np.arange(0, 2*np.pi, np.pi/4)
measurements = np.zeros([2,8])

for pointNum in range(len(pointList)):
    point =  pointList[pointNum] 

    DATA_PATH = Path('../BoreSensorData/').glob(f'{point}*.txt')

    fig, axs = plt.subplots(8,1)
    fig.suptitle(point)

    i = 0

    for dataFile in DATA_PATH:
        rawData = np.genfromtxt(dataFile, delimiter=';', skip_header=1)
        print(dataFile, "\t", rawData.shape)
        t = rawData[:, 0]
        peaks = rawData[:, 1:4]
        lowVal = thresholds[pointNum][0]
        highVal = thresholds[pointNum][1]
        measPeak = np.where(peaks[:, 0] > lowVal, peaks[:,0], peaks[:, 1])

        measurements[pointNum, i] = np.nanmean(measPeak)

        axs[i].plot(t, measPeak)

        i += 1

diameters = (measurements[:, :4] + measurements[:, 4:])/2
avgDiam = np.mean(diameters, axis=1)
trueDiams = np.array([29000, 57000])

absMeasurements = measurements + np.reshape(trueDiams, [2,1])/2 - np.reshape(avgDiam, [2,1])

xyPoints = np.zeros([2,8,2])
xyPoints[:, :, 0] = np.cos(thetas)*absMeasurements/1000
xyPoints[:, :, 1] = np.sin(thetas)*absMeasurements/1000

def calc_best_circle(xPoints, yPoints):
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((xPoints-xc)**2 + (yPoints-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    x_m = np.average(xPoints)
    y_m = np.average(yPoints)
    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)
    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)
    print(f'center xy: {xc_2}, {yc_2}')
    print(f'radius: {R_2}')
    print(f'residuals: {Ri_2 - R_2}')

    center = np.array([xc_2, yc_2])
    radius = R_2
    return center, radius

centerA1, rA1 = calc_best_circle(xyPoints[0,:,0], xyPoints[0,:,1])
centerA5, rA5 = calc_best_circle(xyPoints[1,:,0], xyPoints[1,:,1])

xyPoints[0,:,:] -= centerA1
xyPoints[1,:,:] -= centerA5
# xyPoints[1,:,:] -= centerA1

adjustedMeasurements = np.linalg.norm(xyPoints, axis=2)
measuredDeviation = np.array([14.5, 28.5]).reshape([2,1]) - adjustedMeasurements

A1devScale = 200
fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
# ax2.scatter(xyPoints[0,:,0], xyPoints[0,:,1])
A1PlotData = adjustedMeasurements[0] + measuredDeviation[0]*A1devScale
A1PlotData = np.append(A1PlotData, A1PlotData[0])
thetasPlot = np.append(thetas, thetas[0])
ax2.plot(thetasPlot, A1PlotData)
ax2.set_aspect('equal')
ax2.set_rmax(27)
ax2.set_rticks([14.5])  # Less radial ticks
# ax2.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax2.set_yticklabels([])
ax2.grid(True)
fig2.suptitle('A3 Measured Points')

textOffsets = [[25,0],[25,10],[0,10],[-25,10],[-25,0],[-25,-10],[0,-10],[25,-10]]
for i in range(8):
    # x,y = xyPoints[0,i,0], xyPoints[0,i,1]
    x,y = thetas[i], A1PlotData[i]
    label = f"{measuredDeviation[0,i]*1000:.1f} \u03bcm"
    ax2.annotate(label, 
                 (x,y),
                 textcoords="offset points", # how to position the text
                 xytext=textOffsets[i], # distance from text to points (x,y)
                 ha='center')

A5devScale = 150
fig3, ax3 = plt.subplots(subplot_kw={'projection': 'polar'})
# ax3.scatter(xyPoints[1,:,0], xyPoints[1,:,1])
A5PlotData = adjustedMeasurements[1] + measuredDeviation[1]*A5devScale
A5PlotData = np.append(A5PlotData, A5PlotData[0])
ax3.plot(thetasPlot, A5PlotData)
ax3.set_rmax(50)
ax3.set_rticks([28.5])  # Less radial ticks
# ax3.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax3.set_yticklabels([])
ax3.grid(True)
fig3.suptitle('A5 Measured Points')

for i in range(8):
    # x,y = xyPoints[1,i,0], xyPoints[1,i,1]
    x,y = thetas[i], A5PlotData[i]
    label = f"{measuredDeviation[1,i]*1000:.1f} \u03bcm"
    ax3.annotate(label, 
                 (x,y),
                 textcoords="offset points", # how to position the text
                 xytext=textOffsets[i], # distance from text to points (x,y)
                 ha='center')

plt.show()