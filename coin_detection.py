import cv2
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Step 1: Read Image (FIXED PATH)
# =========================
image = cv2.imread(r"C:\Users\admin\Downloads\coins.jpg")

if image is None:
    print("Image not found. Check coins.jpg path.")
    exit()

imageCopy = image.copy()

plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.show()


# =========================
# Step 2: Grayscale
# =========================
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(imageGray, cmap="gray")
plt.title("Grayscale Image")
plt.show()


# =========================
# Step 3: Thresholding
# =========================
_, imageThresh = cv2.threshold(imageGray, 120, 255, cv2.THRESH_BINARY_INV)

plt.imshow(imageThresh, cmap="gray")
plt.title("Threshold Image")
plt.show()


# =========================
# Step 4: Morphological Operations
# =========================
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

imageDilated = cv2.dilate(imageThresh, kernel, iterations=2)
imageEroded = cv2.erode(imageDilated, kernel, iterations=1)

plt.imshow(imageEroded, cmap="gray")
plt.title("Morphology Result")
plt.show()


# =========================
# Step 5: Blob Detection
# =========================
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = False

params.filterByCircularity = True
params.minCircularity = 0.8

params.filterByConvexity = True
params.minConvexity = 0.8

params.filterByInertia = True
params.minInertiaRatio = 0.8

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(imageEroded)


# =========================
# Step 6: Result
# =========================
print("Number of coins detected:", len(keypoints))

imageWithBlobs = cv2.drawKeypoints(
    imageCopy,
    keypoints,
    None,
    (0, 255, 0),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.imshow(imageWithBlobs[:, :, ::-1])
plt.title("Detected Coins")
plt.show()