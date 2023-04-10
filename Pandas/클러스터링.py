import cv2
import numpy as np

# 이미지 파일을 읽어들입니다.
img = cv2.imread("tiger.jpg")

# 이미지를 벡터 형태로 변환합니다.
vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)

# k-means 알고리즘을 실행합니다.
K = [2, 3, 5] # k 값을 설정합니다.

for k in K:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(vectorized, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 센터 값을 8비트로 변환합니다.
    center = np.uint8(center)

    # 레이블을 이미지 형태로 변환합니다.
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))

    # 결과 이미지를 출력합니다.
    cv2.imshow('k = ' + str(k), result_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
