peoplet = spark.read.text('hdfs:///user/hadoop/people.txt')
peoplet.show()
peoplet.printSchema()

