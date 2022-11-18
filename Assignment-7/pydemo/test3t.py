lines = sc.textFile('/user/hadoop/twinkle.txt')
lineLengths = lines.map(lambda s: len(s))
totalLength = lineLengths.reduce(lambda a, b: a + b)
