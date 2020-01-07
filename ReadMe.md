# SELF-DRIVING CAR: Advance Lane Lines Detection
In the previous project, we talked about implementing Lane Line Detection and we observed there are some limitations in adverse light conditions. Also, we know that the algorithm does not work properly when the road would turn left or right. In this project, we are going to eliminate those restrictions by using new algorithms. So let's talk about some essential concepts and instruments.

### STEPS TO FIND LANE LINES
We are supposed to talk about the following steps:
* Camera Calibration
* What is Warping Perspective?
* Creating Binary Image
* Defining Masking
* Warping Perspective of Lane
* Improving

### Camera Calibration
At first, for making our implementation more realistic, let's talk about a technique which is called Camera Calibration. As you know, there are some types of lens which are used for any situation. One of them is called Fish-Eye. If you get an image by using that lens you will see a distorted image like below:

![Distorted Image](https://media-cdn.tripadvisor.com/media/photo-w/12/dc/90/3e/fish-eye-marine-park.jpg)

so it seems the first step might be calibrating the camera for undistorting the gotten images. Like the previous project, we are going to use the favorite library which is named OpenCV. There is a method in this library that is named 'calibrateCamera'. The purpose of this method is transforming an image that is gotten by the Fish-Eye lens to a normal image without distorting. This act is called calibration.
For calibration, we must use a known picture. Why? Because based on the real data and known picture, we can find differences between them and then we can model a suitable transform matrix for converting a distorted image to normal and undistorted.
One of the famous and suitable images for calibrating the camera is the Chessboard.

<img src="https://render.fineartamerica.com/images/rendered/default/print/8.000/7.875/break/images-medium-5/blank-chess-board-in-black-and-white-aarrows.jpg" alt="Chessboard Image" width="375" height="375" border="10" />

The method is simple. Just find the chessboard in the image by using 'findChessboardCorners' method then put the output of this method as an input of calibrateCamera. That is all. See the below code for more precise.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
undist = cv2.undistort(img, mtx, dist, None, mtx)
```

You can see the output of this code as below:

<img src="https://github.com/PooyaAlamirpour/AdvanceLaneLineDetection/blob/master/camera_cal/calibration1_undist.png" width="540" border="10" />

There are some small tips which we have to know. What is the 'objpoints' and 'imgpoints'?
The 'imgpoints' is each corner of the chessboard cells that are detected by the OpenCV. When the 'findChessboardCorners' can find corners of the chessboard, it sets 'true' for the 'ret' parameter. So we can be noticed there are some corners then we can pick corners point like below:

```python
imgpoints = []
if ret == True:
    imgpoints.append(corners)
```

Firstly it seems that the concept of 'objpoints' is not intelligible but it is completely easy. Let me make an instance. Assume you are playing the puzzle game. At the beginning of the game, all pieces of the puzzle are disordered and you have to put each piece to the appropriate location. The location of each piece in the disordered state equivalent to the 'imgpoints' parameter and the location of each piece in the right position on the puzzle board is equivalent to the 'objpoints'. So we should create 'objpoints' by using below code:

```python
objpoints = []
objp = np.zeros((nx * ny, 3), np.float32)
objp[:,:2] =  np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
if ret == True:
    objpoints.append(objp) 
```

The 'nx' and 'ny' are respectively equal to number of cells in each row and number of cells in each column in the chessboard. 	

<img src="https://github.com/PooyaAlamirpour/AdvanceLaneLineDetection/blob/master/camera_cal/calibration2_undist.png" width="540" border="10" />

<img src="https://github.com/PooyaAlamirpour/AdvanceLaneLineDetection/blob/master/camera_cal/calibration3_undist.png" width="540" border="10" />

### Warping Perspective
Assume you are on a road. Everything you see which are far from you, are in perspective. If we could see all the objects from the front-view instead of perspective, it would be great. It is amazing if you know that there is a method in the OpenCV library exactly for that purpose and it is called 'warpPerspective'. 
The method is easy, set source and destination points and then use 'warpPerspective' for getting the warped image like below:

M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

So, question is, what are the source and destination points? The source points are defined as all coordinates of the corners which were detected on the chessboard and the destination points are defined as all optimal points which we want. It is mean, we want to see an object from the front-view, so we have to presume destination points as same as when we are seeing that object from the front-view. So both of them are defined as below:

```python
src = np.float32(
[
	corners[0], 
	corners[n-1], 
	corners[-1], 
	corners[-n]
])

dst = np.float32(
[
	[offsetx, offsety], 
	[img_size[0]-offsetx, offsety], 
	[img_size[0]-offsetx, img_size[1]-offsety], 
	[offsetx, img_size[1]-offsety]
])
```

I used 'offset' for putting a margin around the output image. You can see the result here:

چندتا تصویر که از نظر پرسپکتیو اصلاح شده اند

### Binary Image
In this section, we want to extract suitable information from an image. In the previous project, we observed some important data that were missed. Because we used Color Channel for finding lane and under adverse light condition some part of the lane line would be missed. In this project, we want to combine two algorithms for solving that issue. At first, Let me introduce HSL Channel. 
HSL (hue, saturation, lightness) and HSV (hue, saturation, value) are alternative representations of the RGB color model. HSL and HSV are both cylindrical geometries, with hue, their angular dimension, starting at the red primary at 0°, passing through the green primary at 120° and the blue primary at 240°, and then wrapping back to red at 360°. In each geometry, the central vertical axis comprises the neutral, achromatic, or gray colors, ranging from black at lightness 0 or value 0, the bottom, to white at lightness 1 or value 1, the top.

نمودار اچ اس ال

So briefly if you see two boxes that have same color, it means both of them to have almost the same H value and if one box has a lighter or darker color it means their 'I' value is different.
For more precise, let see the difference between each value practically. On the 'test_images' folder, there is an image which is called 'straight_lines1'. 

تصویر اصلی straight_lines1

Let see this image in tree H, S and L channel.

تصویر تبدیل شده تصویر اصلی سه تا کانال

Each image shows some features based on the origin image. As it is obvious, the 'S' channel shows the lane line better that other channel. This test gives us an idea for detecting lane lines in different light conditions.
For completing our project, let introduce the 'Soble Algorithm' too. The Sobel Algorithm is used in image processing and computer vision, particularly within edge detection algorithms where it creates an image emphasizing edges. It seems to look like the 'Canny Algorithm' which was used in the previous project. Sobel detection refers to computing the gradient magnitude of an image using 3x3 filters. Where "gradient magnitude" is, for each a pixel, a number giving the absolute value of the rate of change in light intensity in the direction that maximizes this number.
Canny edge detection goes a bit further by removing speckle noise with a low pass filter first, then applying a Sobel filter, and then doing non-maximum suppression to pick out the best pixel for edges when there are multiple possibilities in a local neighborhood. That's a simplification, but basically it’s smarter than just applying a threshold to a Sobel filter. So briefly we know that there is a Sobel engine inner of the Canny Algorithm. In this project, we want to use the Sobel Algorithm by putting some thresholds. You can see my implementation here:

```python
img = np.copy(img)
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
s_channel = hls[:, :, 2]
sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
abs_sobelx = np.absolute(sobelx)
scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= hsv_thresh[0]) & (s_channel <= hsv_thresh[1])] = 1
color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
```

You can use the Sobel Algorithm in any direction you want. I used it in 'X' axis. I used two kinds of thresholds here. One for 'S' channel and another for the 'Sobel Algorithm'. So based on the defined thresholds I can change the efficacy of my algorithm. You can see result below:

تصویر باینری شده straight_lines1

There are two colors in the pictures. One is green, and another is blue. Both of them are the output of HSL and Sobel, which are combined. So obviously you can see all the critical information remained, and other un-useful data was removed. It is amazing. This image is called 'Binary Image'.

### Defining Masking
We have discussed the benefit of using a mask on an image. In the below image you can a mask which bounds a main part of the road. 

تصویر با ماسک straight_lines1

As you see that the put mask is not exactly in the center in comparison with the lane line. First, watch this video:

https://youtu.be/1kwTAMwuYc4

When a car moves and turns, because of lateral motion, it might be you can not put a fixed mask so you can not expect that it is bounded the lane line completely. Of course, you can say we may make a large mask area. But this idea has a bug. When you choose a significant area for masking, it is mean you are selecting more detail of an image, and it might be added some deceptive data. So we have a restriction here. For solving this issue, I have implemented a dynamic masking algorithm. The idea of this algorithm is simple. There is a limited area in the center of the image that we expect there would be lane. So I start to search the inside of a rectangle that I have put to the center of the image. Here is the related part of creating the rectangle in the center of the image:

```python
y1 = int(rec_bottom_left_y)
y2 = int(rec_top_left_y)
x1 = int(rec_top_left_x)
x2 = int(rec_top_right_x)
crop_image = cpy_src[y1:y2, x1:x2]
crop_binary_image = make_binary(crop_image)
```

یک تصویر با یه مستطیل قرمز رنگ روی تصویر باینری برای نمایش مستطیل انتخاب شروع ماسک کردن

For finding lanes inside of this rectangle I used Histogram Plot. 

تصویر هیستوگرام با دو تا قله

This plot shows there are 2 lines because of 2 peaks. So based on the location of them we can realize approximately the start position of our mask. But if there would be 3 peaks, what is the solution? We are always supposed to select 2 peaks that are near together. Now we can set the start points of our mask. In the same way, we can define the endpoints of the mask. So we have dynamic masking. 

### Warping Perspective of Lane
In the same way that we used for the warping perspective chessboard, we can use it for seeing lane from the bird-view. In the future, you can perceive we are interested in changing the view from perspective to bird-view. 

```python
src = np.float32(
[
  [top_left_x, top_left_y],           # top left
  [top_right_x, top_right_y],         # top right
  [bottom_right_x, bottom_right_y],   # bottom ritgh
  [bottom_left_x, bottom_left_y]      # bottom left
])

img_size = (input_image.shape[1], input_image.shape[0])
offset = 100
dst = np.float32(
[
  [offset, offset], 
  [img_size[0]-offset, offset], 
  [img_size[0]-offset, img_size[1]-offset], 
  [offset, img_size[1]-offset]
])
```

colored_binary_warped, Minv = warp(result_make_gradient_transform, src, dst)

نمای خط از بالا

Let's again look at the histogram plot of the bird-view image. As you can see, there are two peaks; it is mean almost there are lanes there. Now, we will define a small window and try to slide it over the expected location. You can see the result here:

تصویر هیستوگرام با دوتا قله

تصویر اسلاید شده

Based on the found points for lane, we can define an equation, and based on that, we can draw a curved line on each frame of the video when we could detect it. In the previous project, we were not able to detect the curved lane. But by having a suitable equation we can do it now.

```python
leftx_find_lane_pixels, lefty_find_lane_pixels, rightx_find_lane_pixels, righty_find_lane_pixels, out_img_find_lane_pixels = find_lane_pixels(binary_warped)
left_fit = np.polyfit(lefty_find_lane_pixels, leftx_find_lane_pixels, 2)
right_fit = np.polyfit(righty_find_lane_pixels, rightx_find_lane_pixels, 2)

left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]
left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
```

So based on the found equation, we can draw a green rectangle inside of the left and right lane. We have a fitted, curved path for the left and right of the road. Now it is time for plotted back down the created green rectangle onto the road. It is means we want to transform the green rectangle from bird-view to perspective view like below:

تصویر مستطیل سبز

We know how we can transform an image from perspective to bird-view. But what about the opposite of it? Do you remember that we have defined source and destination points for transferring? Now we can reverse them for finding a suitable matrix.

newwarp = cv2.warpPerspective(color_warp, Minv, (input_image.shape[1], input_image.shape[0])) 
result = cv2.addWeighted(input_image, 1, newwarp, 0.3, 0)

The 'Minv' is the inverse matrix. Despite we complete our algorithm for finding lane in the road but our story is not ended. 

### Improving
The 'project_video.mp4' has tow main parts. The first and last seconds of the video are in suitable lighting conditions. In the middle of the video, there are 2 scenes that the lighting condition is adverse. Our algorithm can not work properly in that situation. For improving our work there are two approaches. First is tuning the parameters of the Sobel and HSL channel and second is a little bit more complex.
Let's talk about the second approach because the first approach will be achieved by try and error. The second approach talks about combining histogram and HSL channels. We were supposed to use the 'S' channel. But sometimes the 'L' channel has useful information too. The main challenge in the second approach is detecting when we can use 'S' or 'L.' For better selecting, I used histogram information. After converting the input image to binary, I got a histogram from the bounded area of the picture. If there would not be a peak or if there would be lots of unusual peaks, I use 'L' channel instead of 'S', in otherwise, I use the 'S' channel. The final video is downloadable here:

https://www.youtube.com/watch?v=wKwqKegFnK8&list=PLChwywmfd8lqhyap8yrjOeALFLkJ5nRTv&index=5&t=0s

In this project, we used an equation for calculating lane curvature. This value can be used for real measurement. There is a ratio for converting pixel length or curvature to the meter. Let's say that our camera image has 1280 relevant pixels in the y-dimension (remember, our image is perspective-transformed!). Therefore, to convert from pixels to real-world meter measurements, we can use:

```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/1200 # meters per pixel in x dimension
```

So the lane curvature can be calculated as below:

```python
ploty, left_fit_cr, right_fit_cr = generate_data2(ym_per_pix, xm_per_pix)
y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
```

### Refrence
* [HSL and HSV](https://en.wikipedia.org/wiki/HSL_and_HSV)
* [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator)