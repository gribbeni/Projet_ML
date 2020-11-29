# import the necessary packages
from ImageClassification import imutils 

def pyramid(image, scale=1.5, minSize=(36, 36)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image
        
def pyramid_v2(image, crop=5, minSize=(36, 36),layer=0,scale=1.5):
    # yield the original image
    x=0
    y=0
    img=image.copy()
    w=minSize[1]+10
    
    yield imutils.resize(img, width=w),x,y
    
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        #w = int(image.shape[1] / scale)
        img = img[crop:-crop,crop:-crop]
        x+=crop
        y+=crop
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if img.shape[0] < minSize[1] or img.shape[1] < minSize[0] :
            break

        # yield the next image in the pyramid
        yield (imutils.resize(img, width=w),x,y)
        
        layer+=1

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])