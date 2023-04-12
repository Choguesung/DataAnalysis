# import pymysql
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# conn = pymysql.connect(
#     host="localhost",
#     user="root",
#     port=3306,
#     password="qwer1234",
#     db="world",
#     charset="utf8")
#
# curs = conn.cursor()
#
#
# continent = ['Asia','Europe','North America','South America','Africa','Oceania']
# lst = []
# population = [0] * 6
#
# j = 0
#
# sql = "select Name,Continent,Population from country"
# curs.execute(sql)
# lst.append(curs.fetchall())
#
# for i in lst[0]:
#     if i[1] == 'Asia':
#        population[0] += i[2]
#     elif i[1] == 'Europe':
#          population[1] += i[2]
#     elif i[1] == 'North America':
#          population[2] += i[2]
#     elif i[1] == 'South America':
#          population[3] += i[2]
#     elif i[1] == 'Africa':
#          population[4] += i[2]
#     elif i[1] == 'Oceania':
#          population[5] += i[2]
#
# x = np.arange(6)
#
# plt.title('world')
# plt.bar(x,population)
# plt.xticks(x,continent)
#
# plt.show()
#
#111
# #gbgn

import sklearn

print(sklearn.__version__)
