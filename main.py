#import matplotlib.image as mpimg
#import matplotlib.pyplot as mplot
from vehicleDetection import *




   
filename = 'or.jpeg'
image = mpimg.imread(filename)
draw_img = vehicleDetect(image)
fig = plt.figure()
plt.imshow(draw_img)
plt.title('svm pipeline', fontsize=30)
plt.show()

    
