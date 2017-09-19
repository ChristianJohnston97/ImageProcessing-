#####################################################################

# Image Segmentation Summative
# Source Code

# Author : Christian Johnston, christian.johnston@durham.ac.uk

# version 0.1

#####################################################################

import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt
import math

#####################################################################

                    #Read in images

#####################################################################



#####################################################################

     # TO RUN PLEASE PASTE IN BOTH CHANNEL IMAGES AND GROUND TRUTH FOR CORRESPONDING IMAGE 

channel1 = '1649_1109_0003_Amp5-1_B_20070424_A01_w1_9E84F49F-1B25-4E7E-8040-D1BB2D7E73EA.tif'
channel2 = '1649_1109_0003_Amp5-1_B_20070424_A01_w2_15ADF48D-C09E-47DE-B763-5BC479534681.tif'
ground_truth = 'A01_binary.png'


#####################################################################

#Read in both image channels as grayscale
#Normalises so images in range 0-255 instead of 0-1 as input image is.

#read in channel 1 image
img = cv2.imread( channel1 , 0);
cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

#read in channel 2 image
img2 = cv2.imread(channel2, 0);
cv2.normalize(img2, img2, 0, 255, cv2.NORM_MINMAX)

#Read in ground truth image for later
im3 = cv2.imread(ground_truth,0)


#For later
img3 = cv2.imread(channel2, cv2.IMREAD_UNCHANGED);



#Difference between 2 images
def difference (img):
    "This takes an image and compares it with ground truth data"
    diff = cv2.absdiff(img,im3)
    return diff


#Add them together with the same weighting
dst = cv2.addWeighted(img,0.5,img2,0.5,0)



#This function will be used a lot as currently worms are black with white image
#easier to work with black background and white image
#especially for morphological transforms

def invert (img):
    "This takes an image and inverts it"
    "White worms, black background."
    inv = cv2.bitwise_not(img, img)
    return inv

#This power law transform is a different way to brighten the image so is usable
#I will use this later.

# power law transform
# I - colour image I
# gamma - "gradient" co-efficient of gamma function
def powerlaw_transform(I, gamma):
    for i in xrange(0, I.shape[1]): # image width
        for j in xrange(0, I.shape[0]): # image height
                # compute power-law transform
                # remembering not defined for pixel = 0 (!)

                if (I[j,i] > 0):
                    I[j,i] = int(math.pow(I[j,i], gamma));
    return I;

#####################################################################

                     #REGULAR THRESHOLDING

#####################################################################


#If pixel value is greater than a threshold valye, it is assigned
#one value, otherwise assigned another

def binaryThresh (img, val):
    "This takes an image and applies binary thresholding"
    "Threshold value of user input"
    ret,thresh = cv2.threshold(img,val,255,cv2.THRESH_BINARY)
    return thresh

def truncThresh (img, val):
    "This takes an image and applies trunc thresholding"
    "Threshold value of user input"
    ret,thresh = cv2.threshold(img,val,255,cv2.THRESH_TRUNC)
    return thresh

def toZeroThresh (img, val):
    "This takes an image and applies toZero thresholding"
    "Threshold value of user input"
    ret,thresh = cv2.threshold(img,val,255,cv2.THRESH_TOZERO)
    return thresh

#####################################################################

                     #ADAPTIVE THRESHOLDING

#####################################################################

#Normal Thresholding not a good idea often because the image has different lighting conditions in different areas.
#Therefore we go for adaptive thresholding
#calculate threshold for a small region of the image

#the threshold value is the mean of the neighbourhood area
def adaptiveThreshMean (img, blockSize, C):
    "This takes an image and applies mean adaptive thresholding"
    "block size and C of user input"
    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                            cv2.THRESH_BINARY,blockSize,C)
    return thresh



#threshold value is the weighted sum of neighbourhood values when weights
#are a gaussian window
def adaptiveThreshGaussian (im, blockSize, Cg):
    "This takes an image and applies gaussian adaptive thresholding"
    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                            cv2.THRESH_BINARY,blockSize,C)
    return thresh

#####################################################################

                        #OTSU'S BINARIZATON

#####################################################################

#automatically calculates a threshold value from image histogram for an image
#the algorithm finds the optimal threshold value and returns it.

def otsu (img):
    "This takes an image and applies Otsu's binarization thresholding"
    ret2, otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return otsu


#####################################################################

                     #MORPHOLOGICAL TRANSFORMATIONS

#####################################################################
def kernelMaker(size):
    kernel = np.ones((size,size),np.uint8)
    return kernel

#Erosion erodes away boundaries of forground object.
def erosion (img,kernel):
    "This takes an image and applies Erosion"
    erosion = cv2.erode(img,kernelMaker(kernel),iterations=1)
    return erosion

#Dilation is the opposite of erosion
#Joins broken parts of an object
def dilation (img,kernel):
    "This takes an image and applies dilation"
    dilation = cv2.dilate(img,kernelMaker(kernel),iterations=1)
    return dilation


#Opening removes noise in image
def opening (img,kernel):
    "This takes an image and applies opening"
    opening = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernelMaker(kernel))
    return opening


#Closing closes small holes inside forground object
def closing (img, kernel):
    "This takes an image and applies closing"
    closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernelMaker(kernel))
    return closing


#####################################################################

                     #IMAGE SMOOTHING

#####################################################################

#Gaussian Filtering to remove Gaussian Noise
def GaussianBlur (img):
    "This takes an image and applies Gaussian Blur"
    gBlur = cv2.GaussianBlur(img, (5,5), 0)
    return gBlur

#Median Filtering Removes salt and pepper noise
def medianFilter (img):
    "This takes an image and applies Median Filtering"
    median = cv2.medianBlur(img,5)
    return median

#Bilateral Filtering removes noise while mainting edges
def bilateralFilter (img):
    "This takes an image and applies Bilteral Filtering"
    bilateral = cv2.bilateralFilter(img,9,75,75)
    return bilateral

#####################################################################

                     #IMAGE SEGMENTATION
#Applying a Gaussian blur smoothing technique before applying an
#adaptive threshold
#Then inverting the image before applying both a morphological
#opening and closing.
#This removes noise in image and closes holes in worms.
#Segments the worms from the background

#####################################################################

def segmentation (img):
    "This takes an image and appliesnumerous image processing"
    "techniques to segment an image from its background"
    #gBlur = cv2.GaussianBlur(dst, (5,5), 0)
    #Gaussian Blur to remove noise in the image
    img = GaussianBlur(img)
  
    
    #thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 33,10)
    #Adaptive thresholding
    #I found these values gave me the best results
    img = adaptiveThreshMean(img,33,10)

    
    #thresh = cv2.bitwise_not(thresh, thresh)
    #Invert the image
    img = invert(img)

    
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #Morphological opening
    img = opening(img,3)
    
    #closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #Morphological closing
    img = closing(img,3)
   
    return img





#####################################################################

#Or segmentation using regular thresholding
#Same as above process however uses binary global thresholding instead of adaptive

def segmentation2 (img):
    img = GaussianBlur(img)
    #I found this egmentation threshold value to give the best result
    #I found it gave a better thresholding than Otsu
    img = binaryThresh(img,54)
    img = invert(img)
    img = opening(img,3)
    img = closing(img,3)
    
    return img



#####################################################################

                     #WATERSHED ALGORITHM METHOD 1

#A tool for foreground/background seperation and extraction
#as well as for general image segmentation

#####################################################################

def watershed1 (img):    
    #start with finding an approximate estimate of the coins using thresholding

    #thresholding, i found this gave a better output than using Otsu's binarization
    img = binaryThresh(img, 54) 
    
    img = invert(img)

    closed = closing(img,5)
    #Have to create a marker, a marker is the image with the same size as that of
    #original image.

    #Some regions in image where definitely forground- mark with 255 in marker image
    #Region where sure background marked with 128
    #Region not sure mark with 0

    #Erode worms so that we are suyre remaining image belongs to background
    fg = cv2.erode(closed,None, iterations=1)

    #Dilate threshold region so background region is reduced
    bgt = cv2.dilate(closed,None,iterations=3)
    ret,bg = cv2.threshold(bgt,1,128,1)

    #Add foreground and background
    marker = cv2.add(fg,bg)

    #COnvert into 32SC1
    marker32 = np.int32(marker)

    #Apply watershed, needs colour image
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    cv2.watershed(img,marker32)

    #Convert back to uint8 image
    m = cv2.convertScaleAbs(marker32)

    #Threshold
    thresh = otsu(m)
    res = cv2.bitwise_and(dst,dst, mask = thresh)

    res = cv2.equalizeHist(res,res)
  
    return res

#####################################################################

                     #WATERSHED ALGORITHM METHOD 2

#####################################################################

def watershed2(img):

    img = GaussianBlur(img)
    img = adaptiveThreshMean(img,33,10)

    img = cv2.bitwise_not(img, img)
    kernel = np.ones((3,3), np.uint8)                      
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((1,1), np.uint8)                                              
    closing = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)

    #Region near to center of objects are foreground and region much away from the object are background.
    #Only region we are not sure is the boundary region of worms.
    #So we need to extract the area which we are sure they are worms.
    sure_bg = cv2.dilate(closing,kernel,iterations=3)
    
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    #Erosion removes the boundary pixels. So whatever remaining, we can be sure it is worm.
    #That would work if objects were not touching each other.
    #But since they are touching each other,find the distance transform and apply a proper threshold.
    #Next we need to find the area which we are sure they are not worms.
    #For that, we dilate the result. Dilation increases object boundary to background.
    #Whatever region in background in result is really a background, since boundary region is removed.


    #The remaining regions are those which we don't have any idea, whether it is coins or background.
    #Watershed algorithm should find it.
    #These areas are normally around the boundaries where foreground and background meet or even two worms meet.
    #It can be obtained from subtracting sure_fg area from sure_bg area
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    #Barrier between worms

    #mark unknown region with 0
    #sure background is 1
    #sure worm is 2

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers +1
    markers[unknown==255] = 0

    dst = cv2.cvtColor(opening,cv2.COLOR_GRAY2RGB)
    #Watershed algorithm requires colour image
    markers = cv2.watershed(dst,markers)
    dst[markers == -1] = [255,0,0]

   

    return dst



##########################################################################


                    #Power Law Transform

#This is a different way to brighten and segment the images
#Only using the 'w2' channel
#I found gamma of 1.3 gave the best results



##########################################################################



#Using power law transform on "w2" channel only
def powerLawSegmentation(img):
    
    #Define the gamma constant
    gamma = 1.3;
    img = powerlaw_transform(img, gamma)

    

    #Converting to 8 bit
    img = np.uint8(img/256);

    #Simple median blur to remove noise
    img = cv2.medianBlur(img,3)

    #Making a copy of the image
    plate = img.copy()

    #Here im trying to get a clear white frame with well defined edges
    #and an image with well defines worms
    #Then add them together.


    #Used adaptive histogram to improve contrast of image
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(150,150))
    plate = clahe.apply(plate)

    #Otsu's Binarization Thresholding
    ret2, plate = cv2.threshold(plate,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #Median Blur
    plate = cv2.medianBlur(plate,101)

    #Power law transform once again
    img = powerlaw_transform(img, 1.1)
    #Adaptive thresholding
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_BINARY,11,10)

    #Bilateral filter to smooth image without ruining edges
    img = cv2.bilateralFilter(img,30,15,10)

    

    #Median Blur
    img = cv2.medianBlur(img,5)

    #Add images together
    img = img + plate

    #Invert so can apply morphological tranformations
    img = cv2.bitwise_not(img, img)

    #Define the kernel before applying various morphological tranforms
    kernel = np.ones((2,2), np.uint8)
    img = cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernel, iterations =4)
    img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel, iterations =2)
    #kernel = np.ones((3,3), np.uint8)
    #img = cv2.dilate(img,kernel,iterations = 1)

    
    return img



#####################################################################

                     #CONTOURING

#Curve joining all the continuous points along a boundary with same intensity
#Used for detection of individual worms, object labelling and counting
#For better accuracy, use binary images so applied thresholding and various
#filters.

#This function seperates worms from the background, segments individual worms, labelles them,
#counts them, classify them as dead or alive and segments them individually
#printing them onto a black background

#####################################################################

def contouring (img):
    #Use above thresholding method
    img = powerLawSegmentation(img)
    
    #Find contours of image 
    img, contours, hierachy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    #Variables to keep track of number of contours
    i = 0
    j = 0
    #Loop through all contours
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        #perimeter of each contour, approximately 2x length of worm
        perimeter = cv2.arcLength(cnt,True)
        #Length approximately length of worm
        length = perimeter/2

        #If the area of these rectangles is > 250 and <10000, plot
        if(cv2.contourArea(cnt) > 250 and cv2.contourArea(cnt) < 10000):

            #Create a black imgae
            a,black = cv2.threshold(img,255,0,cv2.THRESH_BINARY_INV)

            #Create a minimum area bounding rectangle
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            #Draw the contours
            cv2.drawContours(black,contours,i, (255,0,255),-1)

            #Print each worm individually onto a black background and write to a file!
            
            cv2.imwrite("worm" + (str(j+1)) + ".jpg", black)

            #Drawing Contours
            cv2.drawContours(img, [box],0, (255,255,255), 2)
            cv2.drawContours(img,contours,i,(255,0,255),-1)
            j = j+1

            #Diagonal is the diagonal of the bounding rectangle
            diagonal = abs(abs(box[1][0] + box[3][0]) - abs(box[1][1] + box[3][1]))
            
            #Labelling each worm depending on whether dead or alive
            if(length > diagonal):
                cv2.putText(img, 'Dead Worm', (x+w,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            else:
                cv2.putText(img, 'Alive Worm', (x+w,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)      
        i = i+1
        
    #Print number of contours and thus the number of rectangles
    #Prints in terminal
    print "Number of worms = " + str(j)
    return img                             

                              
#####################################################################

                     #CLASSIFICATION OF DEAD OR ALIVE
                     #Implemented above^^^ in contouring

#####################################################################

#Alive worms are curved and dead worms are straight
#To detect whether dead or alive. I compared the distance of the rotated
#bounding box diagonal to the length of the contour/2
#The straight worms will have a length closer to this than the curved worms.
#If the length is greater than the diagonal, classified as dead worm

#Loop through all contours and create a bounding rectangle for each
#If the area of these rectangles is > 250 and <10000, plot
#Otherwise ignore. This is to avoid drawing recangles due to noise.




#####################################################################

                     #SCRIPT AND EVALUATION
                     #Comparison with Ground Truth Data

#This runs both of my initial segmentation processes followed by the
#difference between them and the ground truth image
#Then runs both watershed algorithm methods
#Then runs the power law image segmentation function and finally
#runs the contouring function.

# Writes all of these to the working directory.

#####################################################################
#Can then subtract the main ground truth image to find the difference
#Comparison


img = segmentation(dst)
cv2.imwrite("ThresholdMethod1.jpg", img)

diff = difference(img)
cv2.imwrite("Difference of segmentation1.jpg", diff)

img = segmentation2(dst)
cv2.imwrite("ThresholdMethod2.jpg", img)

diff = difference(img)
cv2.imwrite("Difference of segmentation2.jpg", diff)

img = watershed1(dst)
cv2.imwrite("Watershed Method 1.jpg", img)

img = watershed2(dst)
cv2.imwrite("Watershed Method 2.jpg", img)

img = powerLawSegmentation(img3.copy())
cv2.imwrite('Power Law Segmentation.jpg', img)

img = contouring(img3)
cv2.imwrite('Contouring.jpg', img)

cv2.waitKey()
cv2.destroyAllWindows()
