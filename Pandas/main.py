import pymysql

conn = pymysql.connect(host="localhost", user="root",port="3307", password="qwer1234", db="youtube", charset="uft8")
curs = conn.cursor()

sql = "select * from user"

curs.execute(sql)

