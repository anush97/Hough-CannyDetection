# import other necessary libaries
import numpy as np
from skimage import io
from skimage import feature
from utils import create_line, create_mask
import matplotlib.pyplot as plt

# load the input image
image = io.imread("road.jpg",as_gray= True)


# run Canny edge detector to find edge points
edges = feature.canny(image)
plt.imshow(edges, cmap='gray')
plt.title('Canny edges')
plt.show()
# create a mask for ROI by calling create_mask
X, Y = image.shape[:2]
mask = create_mask(X, Y)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.show()
# extract edge points in ROI by multipling edge map with the mask
roi_edges = edges * mask

plt.imshow(roi_edges, cmap='gray')
plt.title('Edges within ROI')
plt.show()



def hough_transform(image, angles, distances):
    # Initialize accumulator
    hough_space = np.zeros((len(distances), len(angles)))

    # Loop over all edge points
    y_idxs, x_idxs = np.nonzero(image)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        # Loop over all angles
        for j in range(len(angles)):
            theta = angles[j]

            # Calculate distance from origin
            rho = x*np.cos(theta) + y*np.sin(theta)

            # Find index of distance in accumulator array
            dist_idx = np.argmin(np.abs(distances - rho))

            # Increment accumulator
            hough_space[dist_idx, j] += 1

    return hough_space
angles = np.deg2rad(np.arange(-90, 90))

# Define range of distances to search over
distances = np.arange(0, np.sqrt(np.square(image.shape[0]) + np.square(image.shape[1])), 1)

# Perform Hough transform
hough_space = hough_transform(roi_edges, angles, distances)

# Find the two peaks in Hough space
peaks = feature.peak_local_max(hough_space, min_distance=10, num_peaks=2)

# Find the parameters of the two lines
lines = []
for peak in peaks:
    rho = distances[peak[0]]
    theta = angles[peak[1]]
    line = create_line(rho, theta, image)
    lines.append(line)

# Plot the two lines on the original image
plt.imshow(image)

 
plt.plot(lines[0][0], lines[0][1] ,color ='b',linewidth=2)
plt.plot(lines[1][0], lines[1][1] ,color ='r',linewidth=2)

plt.title("Detected Edges")
plt.show()



# zero out the values in accumulator around the neighborhood of the peak

# find the left lane by finding the peak in hough space

# plot the results
