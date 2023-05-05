import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Read the image.
img = Image.open("tiger.jpg")

# Convert the image to the RGB color model.
img = img.convert("RGB")

# Convert the image to a NumPy array.
img_arr = np.array(img)

# Change the shape of the image array.
img_arr = img_arr.reshape((-1, 3))

# Specifies the number to cluster.
k = 2 # 3,5

# Randomly select an initial centroid.
centers = img_arr[np.random.choice(img_arr.shape[0], k, replace=False)]

# Specifies the number of iterations for clustering.
max_iter = 10

# Apply the k-means algorithm.
for i in range(max_iter):
    # Calculates the distance between all data points and the centroid.
    distances = np.linalg.norm(img_arr[:, np.newaxis, :] - centers, axis=2)

    # Find the index of the nearest centroid.
    labels = np.argmin(distances, axis=1)

    # A new centroid is obtained by averaging the data points belonging to each cluster.
    new_centers = np.array([img_arr[labels == j].mean(axis=0) for j in range(k)])

    # Calculates the change in the central value.
    diff = np.linalg.norm(new_centers - centers)

    # The algorithm terminates when the centroid does not change.
    if diff == 0:
        break

    # Save the new centroid.
    centers = new_centers

# Change the label of the cluster to which each data point belongs to the shape of the image array.
labels = labels.reshape(img.size[1], img.size[0])

# Visualize the clustering results.
plt.imshow(labels)
plt.axis("off")
plt.show()
