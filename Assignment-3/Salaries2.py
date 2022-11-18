from mrjob.job import MRJob

class MRSalaries(MRJob):

    def mapper(self, _, line):
        (name,jobTitle,agencyID,agency,hireDate,annualSalary,grossPay) = line.split('\t')
        if float(annualSalary) >= 100,000.00:
            yield "High", 1
        if(float(annualSalary) >= 50000.00 and float(annualSalary) <= 99999.99 :
             yield "Medium", 1
        else:
             yield "Low", 1

    def combiner(self, annualSalary, counts):
        yield annualSalary, sum(counts)

    def reducer(self, annualSalary, counts):
        yield annualSalary, sum(counts)


if __name__ == '__main__':
    MRSalaries.run()


