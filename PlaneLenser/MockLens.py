import numpy as np


def W(pos, dims, thickness):
    # units len: dims / 3
    units = [(x - 1) / 3 for x in dims]
    normedPos = [(dims[i] - 1 - pos[i]) / units[i] for i in range(len(dims))]

    # line 1: y = 2.5 - 2 x, if x < 1
    # line 2: y = 0.5 + 3 (x - 1), if 1 < x < 3/2
    # line 3: y = 2 - 3 (x - 1.5), if 3/2 < x < 2
    # line 4: y = 0.5 + 2 (x - 2), if 1 < x < 3/2

    if normedPos[1] < 1:
        yCenter = 2.5 - 2 * normedPos[1]
    elif normedPos[1] < 1.5:
        yCenter = 0.5 + 3 * (normedPos[1] - 1)
    elif normedPos[1] < 2:
        yCenter = 2 - 3 * (normedPos[1] - 1.5 )
    else:
        yCenter = 0.5 + 2 * (normedPos[1] - 2)

    return 1 if abs(pow(normedPos[0] - yCenter, 2) * units[0]) < 0.5 * thickness \
           else 0.5 if abs(pow(normedPos[0] - yCenter, 2) * units[0]) < 0.6 * thickness \
           else 0

def I(pos, dims, thickness):
    return abs(pos[1] - dims[1]/2) < thickness

def MockLens(dims, thickness, band, shape = W):
    result = np.zeros(dims)
    bandlessDims = [x - 2 * band for x in dims]
    for x in range(dims[0] - 2 * band):
        for y in range(dims[1] - 2 * band):
            result[x + band, y + band] = shape([x, y], bandlessDims, thickness)
    return result

def ShowMockLens(dims, thickness, band, shape = W):
    lens = MockLens(dims, thickness, band, shape)
    for x in range(dims[0]):
        for y in range(dims[1]):
            print( " " if lens[x, y] == 0 else 1, end=" ")
        print()



if __name__ == "__main__":
    ShowMockLens([30, 30], 2, 4, I)
