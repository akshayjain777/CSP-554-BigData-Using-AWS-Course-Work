peoplej = spark.read.json('hdfs:///user/hadoop/people.json')
peoplej.show()
peoplej.printSchema()

