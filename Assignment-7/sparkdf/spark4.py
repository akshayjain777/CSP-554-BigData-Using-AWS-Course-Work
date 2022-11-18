peoplech = spark.read.csv('hdfs:///user/hadoop/peopleh.csv', header=True)
peoplech.show()
peoplech.printSchema()

