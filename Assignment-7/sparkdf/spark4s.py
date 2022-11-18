from pyspark.sql.types import *

struct1 = StructType().add("schema_defined_name", StringType(), True).add("schema_defined_age",IntegerType(), True)
peoplech = spark.read.schema(struct1).csv('hdfs:///user/hadoop/peopleh.csv', header=True)
peoplech.show()
peoplech.printSchema()

