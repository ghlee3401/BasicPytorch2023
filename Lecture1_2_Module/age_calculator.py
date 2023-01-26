class AgeCalulator:
    def __init__(self, year):
        self.this_year = year 
    
    def cal_age(self, birth_years):
        age_list = list()
        for birth_year in birth_years:
            age = self.this_year - birth_year + 1
            age_list.append(age)
        return age_list