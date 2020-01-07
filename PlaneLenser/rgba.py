# A single pixel, rgba, supporting blending.

import numpy as np
import png
from math import floor

class RGBAInitialValueTooShort(Exception):
    pass

class RGBAPixel:
    """Construct with plain values in [0,1] for R,G,B and A. Internally uses pre-multiplied alpha: R*A, G*A, B*A, A."""

    def __init__(self, iterableRGBA, isPremultiplied = False, canStealReference = False):
        if ( len(iterableRGBA) < 4 ):
            raise RGBAInitialValueTooShort("Not enough information to construct an RGBA value, need len 4, got len " + str(len(iterableRGBA)) + ".")

        self.rgba = np.array(iterableRGBA) if not canStealReference else iterableRGBA
        alpha = self.rgba[3]
        alpha = 0 if alpha < 0 else 1 if alpha > 1 else alpha
        colorLimit = alpha if isPremultiplied else 1

        for i in range(3):
            self.rgba[i] = 0 if self.rgba[i] < 0 else colorLimit if self.rgba[i] > colorLimit else self.rgba[i]
            if (not isPremultiplied):
                self.rgba[i] *= alpha

    def SetToRGBA(self, iterableRGBA, isPremultiplied = False):

        alpha = iterableRGBA[3]
        alpha = 0 if alpha < 0 else 1 if alpha > 1 else alpha
        colorLimit = alpha if isPremultiplied else 1

        for i in range(4):
            self.rgba[i] = 0 if iterableRGBA[i] < 0 else colorLimit if iterableRGBA[i] > colorLimit else iterableRGBA[i]
            if (i != 3 and not isPremultiplied):
                self.rgba[i] *= alpha

    def SetTo(self, otherPixel):
        self.SetToRGBA( otherPixel, isPremultiplied = True)

    def __str__(self):
        return "aRaGaBA [" + str(self.rgba[0]) + ", " + str(self.rgba[1]) + ", " + str(self.rgba[2]) + ", " + str(self.rgba[3]) + "]"

    def __getitem__(self, args):
        return self.rgba[args]

    def __iadd__(self, b):
        frontAlpham1 = 1 - self.rgba[3]
        for i in range(4):
            self.rgba[i] += frontAlpham1 * b.rgba[i]
        return self
        
    def __add__(self, b):
        rgba = np.zeros(4);
        frontAlpham1 = 1 - self.rgba[3]
        for i in range(4):
            rgba[i] = self.rgba[i] + frontAlpham1 * b.rgba[i]

        return RGBAPixel(rgba, isPremultiplied = True, canStealReference = True)

    def __radd__(self, b):
        rgba = np.zeros(4);
        frontAlpham1 = 1 - b.rgba[3]
        for i in range(4):
            rgba[i] = b.rgba[i] + frontAlpham1 * self.rgba[i]

        return RGBAPixel(rgba, isPremultiplied = True, canStealReference = True)

    def __mul__(self, b):
        rgba = np.zeros(4);
        newAlpha = self.rgba[3] * b
        newAlpha = 0 if newAlpha < 0 else 1 if newAlpha > 1 else newAlpha
        for i in range(4):
            rgba[i] = b * self.rgba[i]
            rgba[i] = 0 if rgba[i] < 0 else newAlpha if rgba[i] > newAlpha else rgba[i]

        return RGBAPixel(rgba, isPremultiplied = True, canStealReference = True)

    def __rmul__(self, b):
        return self.__mul__(b)


class RGBAImage:
    """A 2D array of RGBAPixels."""

    def __init__(self, layout):
        if ( len(layout) < 2 ):
            raise RGBAInitialValueTooShort("Not enough information to construct an RGBA array, need len 2, got len " + str(len(iterableRGBA)) + ".")

        self.layout = [layout[0], layout[1], 4]
        self.data = np.zeros(self.layout)
        self.image = np.empty(layout, dtype=object)

        for x in range(layout[0]):
            for y in range(layout[1]):
                self.image[x, y] = RGBAPixel(self.data[x, y, :], isPremultiplied = True, canStealReference = True)

        # image and data point to same memory.

    def __getitem__(self, args):
        return self.image[args]

    def __setitem__(self, args, value):
        self.image[args].SetTo(value)

    def blendAtFloat(self, pixel, pos, behind = False):
        """Draw a single anti-aliased pixel at a floating-point position."""
        ipos = [int(floor(pos[0])), int(floor(pos[1]))]
#        self.image[ipos[0], ipos[1]] += pixel
#        return
        for i in range(2):
            pi = ipos[0] + i
            if ( pi < 0 or pi >= self.layout[0] ):
                print(pi, self.layout[0])
                continue
            iw = abs(pi - pos[0])
            for j in range(2):
                pj = ipos[1] + j
                jw = abs(pj - pos[1])
                if ( pj < 0 or pj >= self.layout[1] ):
                    print(pi, pj, iw * jw, pixel, self.image[pi, pj])
                    continue
#                print(ipos, i, j, iw * jw)
                if behind:
                    self.image[pi, pj] += (iw * jw) * pixel
                else:
                    self.image[pi, pj].SetTo((iw * jw) * pixel + self.image[pi, pj])

    def __str__(self):
        result = "RGBAImage(" + str(self.layout[0]) + ", " + str(self.layout[1]) + ")\n"
        for i in range(0, min(5, self.layout[0])):
            for j in range(0, min(5, self.layout[1])):
                result += str(self.image[i, j]) + " "
            result += "...\n"
        result += "...\n"
        return result

    def CopyOfImage(self):
        mydata = self.data.copy()
        myimage = np.empty([self.layout[0], self.layout[1]], dtype=object)

        for x in range(self.layout[0]):
            for y in range(self.layout[1]):
                myimage[x, y] = RGBAPixel(mydata[x, y, :], isPremultiplied = True, canStealReference = True)

        return mydata, myimage

    def WritePNG(self, name, background = None):

        if ( background != None ):
            background = RGBAPixel(background)
            mydata, myimage = self.CopyOfImage();
            for row in myimage:
                for px in row:
                    px += background
        else:
            mydata = self.data
        
        
        # create a uint8 array
        array = np.zeros(self.layout, dtype='uint8')
        for x in range(self.layout[0]):
            for y in range(self.layout[1]):
                alpha = mydata[x, y, 3]
                aragaba = [mydata[x, y, j] for j in range(4)]
                if alpha > 0:
                    rgba = [int(min(floor(aragaba[j] / alpha * 255), 255)) for j in range(4)]
                    rgba[3] = int(min(floor(alpha * 255), 255))
                else:
                    rgba = [0 for j in range(4)]
                for c in range(self.layout[2]):
                    array[x, y, c] = rgba[c]
#                print(array[x,y])

        writer = png.Writer(width=self.layout[1], height=self.layout[0], size=None, greyscale=False, alpha=True, bitdepth=8, palette=None, transparent=None, background=None, gamma=None, compression=None, interlace=False, bytes_per_sample=None, planes=None, colormap=None, maxval=None, chunk_limit=1048576, x_pixels_per_unit=None, y_pixels_per_unit=None, unit_is_meter=False)

        f = open(name, "wb")
        writer.write(f, np.reshape(array, (self.layout[0], self.layout[1] * self.layout[2] )))
        f.close()

