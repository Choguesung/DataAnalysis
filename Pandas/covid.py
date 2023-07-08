import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#
conn = pymysql.connect(
    host="localhost",
    user="root",
    port=3306,
    password="qwer1234",
    db="covid",
    charset="utf8")


curs = conn.cursor()

sql = "select distinct dong,geonsu from covid"

curs.execute(sql)

lst = curs.fetchall()

print(lst, sep=' ')

