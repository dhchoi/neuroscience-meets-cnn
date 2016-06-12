import pickle
import numpy as np

x = pickle.load(open('x_lda.p'))

r = np.corrcoef(x)

print r.shape

for i in range(9):
    print 'looking ', i+1, 'th subject\'s correlation'
    tot = 0
    for row in range(360*i, 360*i+360):
        for col in range(0, 3240):
           if col >= 360*i and col < 360*i+360:
               continue
           else:
               tot += r[row][col]

    print 'total : ', tot
    
