 
#=============================================================================#
        #
        #    >>>>>>>>>>>>>>> Goals <<<<<<<<<<<<<<<<
        #
        #     	1. Isolate holds of a certain color 
        #		2. Circle isolated holds
        #
        #
#=============================================================================#
# imports
import cv2
import math
import numpy

from matplotlib import pyplot
 

def show(image, color):
    """ Helper method to display a single image 
    with pyplot """
    if (color == "gray"):
        pyplot.imshow(image, cmap="gray")
    else:
        pyplot.imshow(image)
    pyplot.show()


def holds(image, choice):
    high, low = choice
    color_mask = cv2.inRange(image, low, high)
    return cv2.bitwise_and(image, image, mask=color_mask)
    
# read input image
wall = cv2.imread('test2-1.jpg')
## show(wall)
length, width, depth = wall.shape

# Convert color from BGR to RGB
wall = cv2.cvtColor(wall, cv2.COLOR_BGR2RGB)
show(wall, "RGB")

# make black and white version
wall_gray = cv2.cvtColor(wall, cv2.COLOR_RGB2GRAY)
## show(wall_gray, "gray")
 

# use color mask to isolate holds
green_high = (160, 210, 120)
green_low = (80, 130, 30)
yellow_high = (255, 255, 75)
yellow_low = (155, 120, 0)
orange_high = (255, 90, 75)
orange_low = (115, 50, 30)
pink_high = (255, 95, 160)
pink_low = (160, 40, 70)
blue_high = (100, 155, 255)
blue_low = (25, 45, 120)
purple_high = (140, 70, 120)
purple_low = (75, 40, 60)
white_high = (200, 200, 200)
white_low = (150, 150, 150)

color_dictionary = {
    "green"  : (green_high, green_low),
    "yellow" : (yellow_high, yellow_low),
    "orange" : (orange_high, orange_low),
    "pink"   : (pink_high, pink_low),
    "blue"   : (blue_high, blue_low),
    "purple" : (purple_high, purple_low),
}
 

print("What color route would you like to climb?")
print("\tgreen\n\tyellow\n\torange\n\tpink\n\tblue\n\tpurple")

#holds = holds(wall, color_dictionary[input("")])
 
high, low = color_dictionary[input("")]
color_mask = cv2.inRange(wall, low, high)
holds = cv2.bitwise_and(wall, wall, mask=color_mask)

show(wall, "RGB")
show(holds, "RGB")
