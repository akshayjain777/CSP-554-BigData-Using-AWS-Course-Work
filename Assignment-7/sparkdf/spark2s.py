from pyspark.sql.types import *

struct1 = StructType().add("name", StringType(), True).add("age",IntegerType(), True)
peoplet = spark.read.schema(struct1).text('hdfs:///user/hadoop/people.txt')
peoplet.show()
peoplet.printSchema()
