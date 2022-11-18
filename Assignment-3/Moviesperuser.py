from mrjob.job import MRJob

class MRWovieUserRating(MRJob):

    def mapper(self, _, line):
        (userId,mvId,rating,timestamp) = line.split(',')
        yield userId, 1

    def combiner(self, userId, counts):
        yield userId, sum(counts)

    def reducer(self, userId, counts):
        yield userId, sum(counts)


if __name__ == '__main__':
    MRMovieUserRating.run()


