from email.errors import MessageParseError
from json import load
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from pathlib import Path
from IPython import embed

def load_data_from_file(filepath, threshold):
    rawData = np.genfromtxt(filepath, delimiter=';', skip_header=43)
    print(filepath, "\t", rawData.shape)
    t = rawData[:, 0]
    peaks = rawData[:, 1:4]
    lowVal = threshold
    measPeak = np.where(peaks[:, 0] > lowVal, peaks[:,0], peaks[:, 1])

    return t, measPeak


def get_measurement_from_peak(measPeak):
    measurement = np.mean(measPeak)

    return measurement

threshold = 100

thetas = np.arange(0, 2*np.pi, np.pi/4)
rawData = []

dataPath = Path('../A3_repeatability_test/').glob(f'*.txt')

fig, axs = plt.subplots(8,1)
fig.suptitle('A3 Repeatibility Test Raw Data')

i = 0

for dataFile in dataPath:
    # thetas[i] = float(dataFile.stem) * np.pi / 180
    t, measPeak = load_data_from_file(dataFile, threshold)

    if i < 8:
        rawData.append(measPeak)
    else:
        rawData[i%8] = np.append(rawData[i%8], measPeak)
    # measurements[i] = get_measurement_from_peak(measPeak)

    axs[i%8].scatter(t, measPeak, s = 2)
    if i < 8:
        axs[i].set_ylabel(f'{45*i} Degrees')

    i += 1

axs[0].legend(['First Scan', 'Second Scan', 'Third Scan'])
axs[7].set_xlabel('Time (s)')

measurements = np.zeros([8])
stDev = np.zeros([8])
for i in range(8):
    stDev[i] = np.std(rawData[i])
    measurements[i] = np.mean(rawData[i])

diameters = (measurements[:4] + measurements[4:])/2
avgDiam = np.mean(diameters)
trueDiam = 29000

absMeasurements = measurements + trueDiam/2 - avgDiam

xyPoints = np.zeros([8,2])
xyPoints[:, 0] = np.cos(thetas)*absMeasurements/1000
xyPoints[:, 1] = np.sin(thetas)*absMeasurements/1000

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

centerA3, rA3 = calc_best_circle(xyPoints[:,0], xyPoints[:,1])

xyPoints -= centerA3

adjustedMeasurements = np.linalg.norm(xyPoints, axis=1)
measuredDeviation = 14.5 - adjustedMeasurements

A3devScale = 200
fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
# ax2.scatter(xyPoints[0,:,0], xyPoints[0,:,1])
A3PlotData = adjustedMeasurements + measuredDeviation*A3devScale
A3PlotData = np.append(A3PlotData, A3PlotData[0])
stDev = np.append(stDev, stDev[0])
upperBound = A3PlotData + 3*stDev
lowerBound = A3PlotData - 3*stDev
thetasPlot = np.append(thetas, thetas[0])
ax2.plot(thetasPlot, A3PlotData)
ax2.plot(thetasPlot, upperBound, 'k--', linewidth=1)
ax2.plot(thetasPlot, lowerBound, 'k--', linewidth=1)
ax2.set_aspect('equal')
ax2.set_rmax(27)
ax2.set_rticks([14.5])  # Less radial ticks
ax2.set_yticklabels([])
ax2.grid(True)
ax2.legend([r"Measurement ($\mu$m)", r"$3\sigma$ Error"]).set_draggable(True)
fig2.suptitle('A3 Measured Points')

textOffsets = [[25,0],[25,10],[0,10],[-15,5],[-25,0],[-25,-10],[0,-10],[25,-10]]
for i in range(8):
    # x,y = xyPoints[0,i,0], xyPoints[0,i,1]
    x,y = thetas[i], upperBound[i]
    label = f"{measuredDeviation[i]*1000:.1f} \u03bcm"
    ax2.annotate(label, 
                 (x,y),
                 textcoords="offset points", # how to position the text
                 xytext=textOffsets[i], # distance from text to points (x,y)
                 ha='center')

print(3*stDev)

plt.show()