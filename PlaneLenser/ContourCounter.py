import math

class ContourCounter:
    """
    Counts the lengths of contours on a 2d array and the surfaces within them.
    """

    def __init__(self, data):
        self.data = data

    def Measure(self, contourLevels):
        """
        Measures the length of contours of specified levels, and the surface within them.
        contourLevels is a list of contours, or a single value.
        """
        
        if isinstance(contourLevels, (int, float)):
            contourLevels = [contourLevels]
        
        contours = sorted(contourLevels)
        
        def valueToContourIndex(value):
            """
            Does what is says. Maps float value from self.data to
            index in contourLevels.
            """
            c = -1 # contour index
            while (c + 1 < len(contourLevels)) and value > contourLevels[c + 1]:
                c += 1
            return c

        result = []
        for it in contourLevels:
            result.append({
                "level" : it,
                "surface" : 0,
                "circumference" : 0, # all borders
                "circumferenceLower" : 0, # only borders with lower values
                "circumferenceUpper" : 0, # only borders with upper values
                "circumference_over_sqrtSurface" : None,
                "circumferenceLower_over_sqrtSurface" : None,
                "circumferenceUpper_over_sqrtSurface" : None,
                "integral" : 0
             })

        def BumpContour(now, before):
            if now < 0 or now == before:
                return
            result[now]["circumference"] += 1
            if now > before:
                result[now]["circumferenceLower"] += 1
            elif now < before:
                result[now]["circumferenceUpper"] += 1

        def BumpContours(now, before):
            BumpContour(now, before)
            BumpContour(before, now)
        
        # HERE WE GO!
        shape = self.data.shape
        previousContour = valueToContourIndex(self.data[0][0])
        for y in range(shape[0]):
            valuesFromRowAbove = [valueToContourIndex(it) for it in self.data[y - 1 if y > 0 else 0]]
            for x in range(shape[1]):
                c = valueToContourIndex(self.data[y][x])
                cAbove = valuesFromRowAbove[x]
                BumpContours(previousContour, c)
                BumpContours(cAbove, c)
                previousContour = c
                if c > -1:
                    result[c]["surface"] += 1
                    result[c]["integral"] += self.data[x][y]

        for it in result:
            if not it["surface"] == 0:
                sqtsur = math.sqrt(it["surface"])
                it["circumference_over_sqrtSurface"] = it["circumference"] / sqtsur
                it["circumferenceLower_over_sqrtSurface"] = it["circumferenceLower"] / sqtsur
                it["circumferenceUpper_over_sqrtSurface"] = it["circumferenceUpper"] / sqtsur
        return result
