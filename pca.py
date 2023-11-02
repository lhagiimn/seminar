import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# import module
import pandas as pd

# assign data
dataFrame = pd.DataFrame({'Name': [' RACHEL  ', ' MONICA  ', ' PHOEBE  ',
                                   '  ROSS    ', 'CHANDLER', ' JOEY    '],

                          'Age': [30, 35, 37, 33, 34, 30],

                          'Salary': [100000, 93000, 88000, 120000, 94000, 95000],

                          'JOB': ['DESIGNER', 'CHEF', 'MASUS', 'PALENTOLOGY',
                                  'IT', 'ARTIST']})

print(dataFrame.query('Salary  <= 100000 not ( Age > 40) & JOB.str.startswith("C").values'))
exit()

data1 = np.random.random(50)
data2 = 2*data1+0.3*np.random.random(50)

# plt.scatter(x=data1, y=data2)
# plt.show()

print(np.hstack((data1[:, np.newaxis], data2[:, np.newaxis])).shape)

pca = PCA(n_components=2).fit_transform(np.hstack((data1[:, np.newaxis], data2[:, np.newaxis])))
