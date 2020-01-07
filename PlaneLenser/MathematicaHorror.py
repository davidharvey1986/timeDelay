import math
import numpy as np
# put in a separate file because it should never be seen...

def NSISThetasFromBeta(beta, epsilon2, thetaE):
    def Sqrt(x):
        return np.sqrt(x) if x >= 0 else np.Infinity
    def Power(x, y):
        # check that y is integer:
        willWork = np.isfinite(x) and np.isfinite(y)
        if willWork and not np.equal(np.mod(x, 1), 0):
            willWork = willWork and x > 0
        if willWork and y < 0:
            willWork = willWork and not x == 0
        return x**y if willWork else np.Infinity

    return [(3*beta - Sqrt(3)*Sqrt(Power(beta,2) + 2*Power(thetaE,2) +\
         (Power(beta,4) - 2*Power(beta,2)*Power(thetaE,2) -\
            12*epsilon2*Power(thetaE,2) + Power(thetaE,4))/\
          Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
            36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
            3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
            6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
               (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                 Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                 Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
           0.3333333333333333) + Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
           36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
           3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
           6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
              (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
          0.3333333333333333)) - 3*Sqrt(Power(beta,2) + Power(thetaE,2) +\
         (-Power(beta,2) + Power(thetaE,2))/3. +\
         (-Power(beta,4) + 2*Power(beta,2)*Power(thetaE,2) +\
            12*epsilon2*Power(thetaE,2) - Power(thetaE,4))/\
          (3.*Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
              36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
              3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
              6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
                 (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                   Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                   Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
             0.3333333333333333)) -\
         Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
            36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
            3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
            6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
               (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                 Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                 Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
           0.3333333333333333)/3. -\
         (2*Sqrt(3)*beta*Power(thetaE,2))/\
          Sqrt(Power(beta,2) + 2*Power(thetaE,2) +\
            (Power(beta,4) - 2*Power(beta,2)*Power(thetaE,2) -\
               12*epsilon2*Power(thetaE,2) + Power(thetaE,4))/\
             Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
               36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
               3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
               6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
                  (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                    Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                    Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
              0.3333333333333333) +\
            Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
              36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
              3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
              6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
                 (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                   Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                   Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
             0.3333333333333333))))/6.,\
   (3*beta - Sqrt(3)*Sqrt(Power(beta,2) + 2*Power(thetaE,2) +\
         (Power(beta,4) - 2*Power(beta,2)*Power(thetaE,2) -\
            12*epsilon2*Power(thetaE,2) + Power(thetaE,4))/\
          Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
            36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
            3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
            6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
               (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                 Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                 Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
           0.3333333333333333) + Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
           36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
           3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
           6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
              (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
          0.3333333333333333)) + 3*Sqrt(Power(beta,2) + Power(thetaE,2) +\
         (-Power(beta,2) + Power(thetaE,2))/3. +\
         (-Power(beta,4) + 2*Power(beta,2)*Power(thetaE,2) +\
            12*epsilon2*Power(thetaE,2) - Power(thetaE,4))/\
          (3.*Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
              36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
              3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
              6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
                 (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                   Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                   Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
             0.3333333333333333)) -\
         Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
            36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
            3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
            6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
               (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                 Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                 Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
           0.3333333333333333)/3. -\
         (2*Sqrt(3)*beta*Power(thetaE,2))/\
          Sqrt(Power(beta,2) + 2*Power(thetaE,2) +\
            (Power(beta,4) - 2*Power(beta,2)*Power(thetaE,2) -\
               12*epsilon2*Power(thetaE,2) + Power(thetaE,4))/\
             Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
               36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
               3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
               6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
                  (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                    Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                    Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
              0.3333333333333333) +\
            Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
              36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
              3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
              6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
                 (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                   Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                   Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
             0.3333333333333333))))/6.,\
   (3*beta + Sqrt(3)*Sqrt(Power(beta,2) + 2*Power(thetaE,2) +\
         (Power(beta,4) - 2*Power(beta,2)*Power(thetaE,2) -\
            12*epsilon2*Power(thetaE,2) + Power(thetaE,4))/\
          Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
            36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
            3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
            6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
               (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                 Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                 Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
           0.3333333333333333) + Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
           36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
           3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
           6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
              (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
          0.3333333333333333)) - 3*Sqrt(Power(beta,2) + Power(thetaE,2) +\
         (-Power(beta,2) + Power(thetaE,2))/3. +\
         (-Power(beta,4) + 2*Power(beta,2)*Power(thetaE,2) +\
            12*epsilon2*Power(thetaE,2) - Power(thetaE,4))/\
          (3.*Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
              36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
              3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
              6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
                 (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                   Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                   Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
             0.3333333333333333)) -\
         Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
            36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
            3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
            6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
               (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                 Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                 Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
           0.3333333333333333)/3. +\
         (2*Sqrt(3)*beta*Power(thetaE,2))/\
          Sqrt(Power(beta,2) + 2*Power(thetaE,2) +\
            (Power(beta,4) - 2*Power(beta,2)*Power(thetaE,2) -\
               12*epsilon2*Power(thetaE,2) + Power(thetaE,4))/\
             Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
               36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
               3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
               6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
                  (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                    Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                    Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
              0.3333333333333333) +\
            Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
              36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
              3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
              6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
                 (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                   Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                   Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
             0.3333333333333333))))/6.,\
   (3*beta + Sqrt(3)*Sqrt(Power(beta,2) + 2*Power(thetaE,2) +\
         (Power(beta,4) - 2*Power(beta,2)*Power(thetaE,2) -\
            12*epsilon2*Power(thetaE,2) + Power(thetaE,4))/\
          Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
            36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
            3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
            6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
               (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                 Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                 Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
           0.3333333333333333) + Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
           36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
           3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
           6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
              (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
          0.3333333333333333)) + 3*Sqrt(Power(beta,2) + Power(thetaE,2) +\
         (-Power(beta,2) + Power(thetaE,2))/3. +\
         (-Power(beta,4) + 2*Power(beta,2)*Power(thetaE,2) +\
            12*epsilon2*Power(thetaE,2) - Power(thetaE,4))/\
          (3.*Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
              36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
              3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
              6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
                 (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                   Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                   Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
             0.3333333333333333)) -\
         Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
            36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
            3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
            6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
               (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                 Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                 Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
           0.3333333333333333)/3. +\
         (2*Sqrt(3)*beta*Power(thetaE,2))/\
          Sqrt(Power(beta,2) + 2*Power(thetaE,2) +\
            (Power(beta,4) - 2*Power(beta,2)*Power(thetaE,2) -\
               12*epsilon2*Power(thetaE,2) + Power(thetaE,4))/\
             Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
               36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
               3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
               6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
                  (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                    Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                    Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
              0.3333333333333333) +\
            Power(Power(beta,6) - 3*Power(beta,4)*Power(thetaE,2) -\
              36*epsilon2*Power(thetaE,4) - Power(thetaE,6) +\
              3*Power(beta,2)*Power(thetaE,2)*(-6*epsilon2 + Power(thetaE,2)) +\
              6*Sqrt(3)*Sqrt(epsilon2*Power(thetaE,4)*\
                 (-Power(beta,6) - Power(beta,4)*(epsilon2 - 3*Power(thetaE,2)) +\
                   Power(4*epsilon2*thetaE + Power(thetaE,3),2) +\
                   Power(beta,2)*(20*epsilon2*Power(thetaE,2) - 3*Power(thetaE,4)))),\
             0.3333333333333333))))/6.]
