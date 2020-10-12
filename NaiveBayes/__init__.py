'''
@Project:MachineLearningAction

@Author:lincoln

@File:__init__.py

@Time:2020-06-11 12:54:57

@Description:
'''
import numpy as np
a = np.array([[0,1,2,3],[0,1,2,3]])
print(a/3)
for i,j in zip(a,range(len(a))):
    print(i,j)
