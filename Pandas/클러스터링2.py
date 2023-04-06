import cv2
import numpy as np

# 타이거 이미지 로드
tiger_img = cv2.imread('lion.jpg')
tiger_img = cv2.cvtColor(tiger_img, cv2.COLOR_BGR2RGB)

# 비행기 이미지 로드
airplane_img = cv2.imread('plane.jpg')
airplane_img = cv2.cvtColor(airplane_img, cv2.COLOR_BGR2RGB)

# k-means 알고리즘 실행
k_values = [2, 3, 5]
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

# 타이거 이미지에 대한 클러스터링
for k in k_values:
    tiger_img_flat = tiger_img.reshape((-1, 3))
    tiger_centroids, tiger_labels, _ = cv2.kmeans(tiger_img_flat, k, None, criteria, 10, flags)
    tiger_res = tiger_centroids[tiger_labels.flatten()].reshape(tiger_img.shape)

    # 결과 출력
    import matplotlib.pyplot as plt
    plt.imshow(tiger_res)
    plt.title(f'Clustered Tiger Image (k={k})')
    plt.show()

# 비행기 이미지에 대한 클러스터링
for k in k_values:
    airplane_img_flat = airplane_img.reshape((-1, 3))
    airplane_centroids, airplane_labels, _ = cv2.kmeans(airplane_img_flat, k, None, criteria, 10, flags)
    airplane_res = airplane_centroids[airplane_labels.flatten()].reshape(airplane_img.shape)

    # 결과 출력
    import matplotlib.pyplot as plt
    plt.imshow(airplane_res)
    plt.title(f'Clustered Airplane Image (k={k})')
    plt.show()
