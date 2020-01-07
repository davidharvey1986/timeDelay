


class LSSTredshiftDistribution:

    def __init__( self, solidAngle, faintEndSlope, brightEndSlope, breakMagnitude, phiStar=5.34e−6):
        self.solidAngle = solidAngle
        self.faintEndSlope = faintEndSlope
        self.brightEndSlope = brightEndSlope
        self.breakMagnitude = breakMagnitude


    def getdPhidM( self ):

        
