# Assignment 1: Coin Detection, Segmentation, and Counting

## Objective
This assignment focuses on detecting, segmenting, and counting coins in an image using image processing techniques in OpenCV.
Apart from this we also explore image stitching algorithms.

## Tasks
### 1. Detect All Coins in the Image (2 Marks)
- Apply **edge detection** to identify all coins in the image.
- Visualize the detected coins by drawing outlines around them.

### 2. Segmentation of Each Coin (3 Marks)
- Use **region-based segmentation techniques** to isolate individual coins.
- Provide segmented outputs for each detected coin.

### 3. Count the Total Number of Coins (2 Marks)
- Implement a function to count the total number of detected coins.
- Display the final count as output.

## Approach
1. **Preprocessing:**
   - Convert image to grayscale.
   - Apply Gaussian blur to reduce noise.

2. **Edge Detection:**
   - Use Canny edge detection.
   - Alternatively, apply Laplacian or Hough Circle Transform for improved accuracy.

3. **Contour Detection & Segmentation:**
   - Find contours using `cv2.findContours()`.
   - Extract individual coins based on contour properties.

4. **Counting Coins:**
   - Count the number of detected contours.
   - Display the image with detected coins outlined.

## Dependencies
Ensure you have the following installed:
```bash
conda create --name Assignment1 --file environment.txt
conda activate Assignment1
```

## Viewing the actual results
The actual submission can be viewed in the following files:
1. CoinDetection.ipynb
2. ImageStichingSIFT.ipynb

## Output
- An image with detected coins outlined.
- Segmented images of individual coins.
- Printed count of detected coins.

---
### Notes
- Experiment with different edge detection and thresholding techniques for better results.
- Tune parameters for Gaussian blur and Canny edge detection to optimize performance.

