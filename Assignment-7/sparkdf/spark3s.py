from pyspark.sql.types import *

struct1 = StructType().add("name", StringType(), True).add("age",IntegerType(), True)
peoplec = spark.read.schema(struct1).csv('hdfs:///user/hadoop/people.csv')
peoplec.show()
peoplec.printSchema()

