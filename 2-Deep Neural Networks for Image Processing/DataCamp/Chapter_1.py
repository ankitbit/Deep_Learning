## Image Processing With Neural Networks

# Import matplotlib
import matplotlib.pyplot as plt

# Load the image
data = plt.imread('bricks.png')

# Display the image
plt.imshow(data)
plt.show()

## Images as data: changing images

'''
To modify an image, you can modify the existing numbers in the array. 
In a color image, you can change the values in one of the color channels without affecting the other colors, 
by indexing on the last dimension of the array.
The image you imported in the previous exercise is available in data.
'''

