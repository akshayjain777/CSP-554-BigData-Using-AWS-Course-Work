peoplec = spark.read.csv('hdfs:///user/hadoop/people.csv')
peoplec.show()
peoplec.printSchema()

