from powerLawFit import *



def main():


    x = np.linspace(0,3, 100) 

    p=[10**0.8,60., 200, -1, 0., 1., 0.1,0.1]
    f, g, y = doubleBrokenPowerLaw(x,*p)
    plt.plot(x,f)
    plt.plot(x,g)
    plt.plot(x,y)
    plt.show()


if __name__ == '__main__':
    main()
