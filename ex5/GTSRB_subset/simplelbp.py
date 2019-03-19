import numpy as np
from math import floor, ceil

def bilinear(image, r, c):
    minr = floor(r)
    minc = floor(c)
    maxr = ceil(r)
    maxc = ceil(c)

    dr = r-minr
    dc = c-minc

    top = (1-dc)*image[minr,minc] + dc*image[minr,maxc]
    bot = (1-dc)*image[maxr,minc] + dc*image[maxr,maxc]

    return (1-dr)*top+dr*bot

def local_binary_pattern(image, P=8, R=1):
    rr = - R * np.sin(2*np.pi*np.arange(P, dtype=np.double) / P)
    cc = R * np.cos(2*np.pi*np.arange(P, dtype=np.double) / P)
    rp = np.round(rr, 5)
    cp = np.round(cc, 5)
    
    rows = image.shape[0]
    cols = image.shape[1]

    output = np.zeros((rows, cols))

    for r in range(R,rows-R):
        for c in range(R,cols-R):
            lbp = 0
            for i in range(P):
                if bilinear(image, r+rp[i], c+cp[i]) - image[r,c] >= 0:
                    lbp += 1<<i
                            
            output[r,c] = lbp

    return output
