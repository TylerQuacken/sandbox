import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

# https://lucidar.me/en/mathematics/calculating-the-transformation-between-two-set-of-points/
# https://stats.stackexchange.com/questions/186111/find-the-rotation-between-set-of-points

# Generate a grid of points (a)
BLOB_SIZE = np.array([5, 3])
A_TRANSLATION = np.array([7, 5])
B_NOISE = .01
B_SHIFT = np.array([-1,3])
N_POINTS = 2

xRange = A_TRANSLATION[0] + np.array([-BLOB_SIZE[0]/2, BLOB_SIZE[0]/2]) 
yRange = A_TRANSLATION[1] + np.array([-BLOB_SIZE[1]/2, BLOB_SIZE[1]/2]) 
pointsAx = np.random.uniform(xRange[0], xRange[1], N_POINTS)
pointsAy = np.random.uniform(yRange[0], yRange[1], N_POINTS)
aPoints = np.zeros([N_POINTS, 2])
aPoints[:,0] = pointsAx
aPoints[:,1] = pointsAy

# rotate points to get destination (b)
# rotAngle = np.deg2rad(30)
# bTrans = np.array([1,3]) + A_TRANSLATION
# R = np.eye(3,3)
# c = np.cos(rotAngle)
# s = np.sin(rotAngle)
# R[:2,:2] = [[c, -s],
#             [s, c]]
# R[:2, 2] = trans

# aExpanded = np.ones([N_POINTS, 3, 1])    # size == (batch, rows, col) for matrix mult
# aExpanded[:,:2,0] = aPoints - A_TRANSLATION
# bExpanded = R@aExpanded
# bPoints = bExpanded[:,:2,0] + A_TRANSLATION

rotAngle = np.deg2rad(np.random.uniform(0,180))
bTrans = B_SHIFT + A_TRANSLATION
c = np.cos(rotAngle)
s = np.sin(rotAngle)
R = np.array([[c, -s],
              [s, c]])

aExpanded = np.ones([N_POINTS, 2, 1])    # size == (batch, rows, col) for matrix mult
aExpanded[:,:,0] = aPoints - A_TRANSLATION
bExpanded = R@aExpanded
bPoints = bExpanded[:,:,0] + bTrans


bNoise = np.random.uniform(-B_NOISE, B_NOISE, bPoints.shape)
bPoints += bNoise

# find COG of each cloud, move to origin
aCOG = np.mean(aPoints, 0)
bCOG = np.mean(bPoints, 0)

print("aCOG: {} \taTrans: {}".format(aCOG, A_TRANSLATION))
print("bCOG: {} \tbTrans: {}".format(bCOG, bTrans))

aPrime = aPoints - aCOG
bPrime = bPoints - bCOG

# compute N

C = bPrime.T@aPrime   # matrices should be 2 rows, n cols (mine are opposite)

# do SVD and get R
U, s, Vt = np.linalg.svd(C, full_matrices=True)
V = Vt.T
d = np.sign(np.linalg.det(U))
D = np.array([[1, 0], [0, d]])

# R2 = V@D@U.T
R2 = V@U.T
theta = np.arccos(R2[0,0])

# print(R)
print(np.rad2deg(rotAngle))
# print(R2)
print(np.rad2deg(theta))



# Rotate A


# Plot A and B

fig, ax = plt.subplots()
ax.scatter(aPoints[:,0], aPoints[:,1])
ax.scatter(bPoints[:,0], bPoints[:,1])
ax.scatter(aCOG[0], aCOG[1])
ax.scatter(bCOG[0], bCOG[1])
ax.set_xlim([0,15])
ax.set_ylim([0,15])
plt.show()
