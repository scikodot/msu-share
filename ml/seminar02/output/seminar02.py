import numpy as np

def a1(x):
    return 1 if np.sqrt(4 / np.pi - 1) <= x <= 3 else -1
    
def R():
    return 0.299958
    
def a2(x):
    # A priori probabilities
    py1, py2 = 0.4, 0.6

    # Means
    m11, m21 = -0.10135129584864251, 0.7919239376266681
    m12, m22 = 1.018233909079016, 0.4972632323757254

    # Standard deviations
    s11, s21 = 1.1701143965335394, 2.0140666524900777
    s12, s22 = 1.0109836175071003, 1.918251875918784

    # Distributions
    f = lambda x1, x2 : py1/(2*np.pi*s11*s21) * np.exp(-1/2*(((x1-m11)/s11)**2 + ((x2-m21)/s21)**2))
    g = lambda x1, x2 : py2/(2*np.pi*s12*s22) * np.exp(-1/2*(((x1-m12)/s12)**2 + ((x2-m22)/s22)**2))

    # Classify
    return -1 if f(*x) >= g(*x) else 1
    
if __name__ == '__main__':
    print(a1(0.0), R(), a2([0.0, 0.0]))
    
    with open('seminar02_task1.txt', 'w') as f:
        for i in range(-50, 50):
            x = i/10.0
            y = a1(x)
            f.write('%d ' % y)
            
    with open('seminar02_task2.txt', 'w') as f:
        f.write('%.3f' % R())   
    
    with open('seminar02_task3.txt', 'w') as f:
        for i in range(-50, 50):
            x1 = i/10.0
            for j in range(-50, 50):
                x2 = j/10.0
                y = a2([x1, x2])
                f.write('%d ' % y) 