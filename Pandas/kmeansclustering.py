# Import the OpenCV and NumPy modules.
import cv2
import numpy as np

# Read an image file.
img = cv2.imread("tiger.jpg")

# Convert the image to vector form.
vectorized = img.reshape((-1,3)) # Convert a 2D image array to a 1D vector.
vectorized = np.float32(vectorized) # Convert the type to float32 for input to the K-means algorithm.

# Run the k-means algorithm.
K = [2, 3, 5] # Set the k value.

for k in K:
    # Set the parameters of the k-means algorithm.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # cv2.TERM_CRITERIA_EPS : Accuracy standard
    # cv2.TERM_CRITERIA_MAX_ITER : max number of iterations
    # 10 : The number of iterations of the k-means algorithm
    # 1.0 : The algorithm terminates when both the accuracy criterion and the number of iterations are met.

    # Run the k-means algorithm.
    ret, label, center = cv2.kmeans(vectorized, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # vectorized : input data
    # k : number of clusters
    # None : Do not set initial cluster centroids
    # criteria : Algorithm End Criterion
    # 10 : max number of iterations
    # cv2.KMEANS_RANDOM_CENTERS : Initialize the cluster centers randomly.

    # Convert the center value to 8 bits.
    center = np.uint8(center)

    # Convert the label to image form.
    res = center[label.flatten()] # Convert the labels to a one-dimensional vector and find the cluster centroid corresponding to each label.
    result_image = res.reshape((img.shape)) # Convert vector to image array form.

    # Output the resulting image.
    cv2.imshow('k = ' + str(k), result_image)
    cv2.waitKey(0) # Waits for a keystroke.

cv2.destroyAllWindows() # Close all windows.
