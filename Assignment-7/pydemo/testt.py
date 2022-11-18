lines=sc.textFile('/user/hadoop/twinkle.txt')
upper=lines.map(lambda line: line.upper())
words= lines.flatMap(lambda line: line.split(" "))

