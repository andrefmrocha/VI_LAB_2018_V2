import cv2
import numpy as np
import matplotlib.pyplot as plt


MIN_MATCH_COUNT = 30                                    # Minimum number of matches
CARD_IMG_FN     = "./TrainingData/template_card.jpeg"   # Template card path (query image)
TEST_VIDEO_FN   = "./TestVideo/movie.avi"               # Test video path (train images)
VERBOSE         = True                                                            

# Initialization of sift
sift = cv2.xfeatures2d.SIFT_create()

# FLANN parameters
FLANN_INDEX_KDITREE = 1
flannParam          = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
search_params       = dict(checks=50) # or use {}

# Initialization of FLANN
flann = cv2.FlannBasedMatcher(flannParam, search_params)

# Read the template image
card_img = cv2.imread(CARD_IMG_FN,0)

"""
TODO: Detect key-points on the template image using SIFT algorithm.

You should output the following variables:
  card_kpts - SIFT key-points
  card_desc - SIFT descriptors
  
Hint: check sift.detectAndCompute()
"""
######################
### YOUR CODE HERE ###
######################


# Draw key-points
if VERBOSE:
    kpts_img = cv2.drawKeypoints(card_img, card_kpts, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(kpts_img)
    plt.axis('off')
    plt.show()

# Creation of VideoCapture object to read the video
video  = cv2.VideoCapture(TEST_VIDEO_FN)
current_frame = 0
frame_step    = 20

# Loop over video frames
while True:
    if (current_frame % frame_step) != 0:
        ret, video_img_bgr = video.read()
        current_frame += 1
        continue

    # read the next video frame and convert it to gray-scale
    ret, video_img_bgr = video.read()
    rows, cols, depth  = video_img_bgr.shape
    video_img_gray     = cv2.cvtColor(video_img_bgr, cv2.COLOR_BGR2GRAY)
    
    
    """
    TODO: Detect key-points on the current frame using SIFT algorithm.
    
    You should output the following variables:
      video_kpts - SIFT key-points
      video_desc - SIFT descriptors
    """
    ######################
    ### YOUR CODE HERE ###
    ######################
    
    
    """
    TODO: Match key-point descriptors between the two images, using FLANN algorithm. 
    For each key-point descriptor in the query image, you should find the 2 best 
    matches in the video frame.
    
    You should output the following variables:
      matches - list of DMatch objects containing the matches found.
      
    Hint: check flann.knnMatch()  
    """
    ######################
    ### YOUR CODE HERE ###
    ######################
    
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # Find only the good matches using ratio test as per Lowe's paper
    goodMatch = []
    for i, (m,n) in enumerate(matches):
        """
        TODO: Write the condition for the ratio test. You may try different ratios,
        e.g.: 0.8, 0.7, 0.3, ...
        
        You should output the following variables:
          condition - boolean saying whether the test has passed or failed.
        
        Hint: m.distance contains the distance corresponding to the best match
        (i.e. the lowest distance), n.distance ccontains the distance corresponding 
        to the second best match.
        """
        ######################
        ### YOUR CODE HERE ###
        ######################
        
        
        # Discard matches that failed the ratio test 
        if condition:
            goodMatch.append([m])
            matchesMask[i]=[1,0]

    if VERBOSE and current_frame == 0:
        draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)

        matches_img = cv2.drawMatchesKnn(card_img, card_kpts, video_img_gray, video_kpts, matches, None, **draw_params)
        plt.imshow(matches_img)
        plt.axis('off')
        plt.show()

        goodMatch = sorted(goodMatch, key = lambda x:x[0].distance)
        draw_params = dict(matchColor = (0,255,0))
        matches_img = cv2.drawMatchesKnn(card_img,card_kpts,video_img_gray,video_kpts,goodMatch[:30],None,**draw_params)
        plt.imshow(matches_img)
        plt.axis('off')
        plt.show()

    if(len(goodMatch) > MIN_MATCH_COUNT):
        tp = []
        qp = []
        # Extract the locations of good matched key-points in both images
        for m in goodMatch:
            qp.append(card_kpts[m[0].queryIdx].pt)   # query points
            tp.append(video_kpts[m[0].trainIdx].pt)  # target points

        qp, tp = np.float32((qp, tp))
        
        """
        TODO: Find the Homography matrix between the query points and the target
        points. This matrix defines a mapping between them.
        
        You should output the following variables:
          H - the Homography matrix.
        
        Hint: check cv2.findHomography()
        """
        ######################
        ### YOUR CODE HERE ###
        ######################


        # Height and width of the card in the query image
        h, w = card_img.shape
        """
        TODO: Find the coordinates of the card corners in the video image.
        For this purpose, you should apply a perspective transform to the card 
        corners coordinates in the query image (which you know) using the Homography
        matrix H
        
        You should output the following variables:
          card_corners_video - the coordinates of the card corners in the video image.
        
        Hint: cv2.perspectiveTransform()
        """
        ######################
        ### YOUR CODE HERE ###
        ######################
        
        
        cv2.polylines(video_img_bgr, [np.int32(card_corners_video)], True, (0,255,0), 5)
    else:
        print("Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT))

    cv2.imshow('result',video_img_bgr)
    
    if cv2.waitKey(10)==ord('q'):
        break

    current_frame += 1

video.release()
cv2.destroyAllWindows()
