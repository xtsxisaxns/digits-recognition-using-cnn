from __future__ import division
import numpy as np
import cv2
import mahotas


# load csv format MNIST data
def load_digits(datasetPath):
    data = np.genfromtxt(datasetPath,delimiter = ",",dtype = "uint8")
    target = data[:,0]
    data = data[:,1:].reshape(data.shape[0],28,28)
    return (data,target)

# deskew data image 1
def deskew(image,width):
    (h,w) = image.shape[:2]
    moments = cv2.moments(image)
    skew = moments["mu11"]/moments["mu02"]
    M = np.float32([
        [1,skew,-0.5*w*skew],
        [0,1,0]])

    image = cv2.warpAffine(image,M,(w,h),flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return image

# #deskew data image 2
# def deskew(img):
#     m = cv2.moments(img)
#     if abs(m['mu02']) < 1e-2:
#         return img.copy()
#     skew = m['mu11']/m['mu02']
#     M = np.float32([
#         [1, skew, -0.5*SZ*skew],
#         [0, 1, 0]])
#     img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
#     return img

# input deskewed image,centerize the image
def center_extent(image,size):
    (eW,eH) = size

    if image.shape[1] > image.shape[0]:
        ratio = eW/image.shape[1]
        image = cv2.resize(image,(eW,int(ratio*image.shape[0])))
    else:
        ratio = eH/image.shape[0]
        #debug
        #print "ratio is",ratio
        image = cv2.resize(image,(int(ratio*image.shape[1]),eH))

    extent = np.zeros((eH,eW),dtype="uint8")
    offsetX = (eW - image.shape[1]) // 2
    offsetY = (eH - image.shape[0]) // 2

    extent[offsetY:offsetY+image.shape[0],offsetX:offsetX+image.shape[1]] = image

    CM = mahotas.center_of_mass(extent)
    (cY,cX) = np.round(CM).astype("int32")
    (dx,dy) = ((size[0] // 2) - cX,(size[1] // 2) - cY)
    M = np.float32([[1,0,dx],[0,1,dy]])
    extent = cv2.warpAffine(extent,M,size)

    return extent
