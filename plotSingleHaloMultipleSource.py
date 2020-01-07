
from convolveDistributionWithLineOfSight import *

def main():
    jsonFile ='../output/CDM/z_0.20/B005_cluster_0_2_total_sph.fits.py.raw.json'


    jsonData = json.load(open(jsonFile, 'rb'))

    singleSource = jsonData[-1]
    xPosition = np.array([])
    yPosition = np.array([])
    zPosition = np.array([])
    print(singleSource.keys())
    for iImageFamily in range(len(singleSource['positionX'])):
            
            

            noZeros=np.array(singleSource['doubles'][iImageFamily])!=0 
            imagesInd =  np.argsort(np.abs(singleSource['doubles'][iImageFamily])[noZeros])[0:2]



            xPosition = \
              np.append(xPosition, np.array(singleSource['positionX'][iImageFamily])[noZeros][imagesInd] )
            yPosition = \
              np.append(yPosition, np.array(singleSource['positionY'][iImageFamily])[noZeros][imagesInd] )
            zPosition =\
              np.append(zPosition, np.abs(1./np.array(singleSource['doubles'][iImageFamily])[noZeros][imagesInd]))


    plt.scatter( xPosition, yPosition, c=np.log10(zPosition))
    plt.show()
    




if __name__ == '__main__':
    main()
