import numpy as np
from scipy import signal as sg

# 1-dimensional convolution
x = [3, 4, 5]
h = [2, 1, 0]

# with 0 padding
y = np.convolve(x, h)
print(y)

x1 = [6, 2]
h1 = [1, 2, 5, 4]

y = np.convolve(x1, h1, 'full')
print(y)

# with no padding
y = np.convolve(x1, h1, 'valid')
print(y)

# 2-dimensional convolution
I = [[255, 7, 3],
     [212, 240, 4],
     [218, 216, 230]]
g = [[-1, 1]]

print('without zero padding \n')
print('{0} \n'.format(sg.convolve(I, g, 'valid')))
# the valid argument states that the output consists only of
# those elements that do not rely on the zero-padding

print('with zero padding \n')
print(sg.convolve(I, g, 'full'))