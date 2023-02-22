from family import Family
from age_calculator import AgeCalulator

# Step1 : 가족 설정
Noh = Family("노")
names = Noh.get_family_full_names()
birth_years = Noh.get_family_full_birth()

print("가족 이름 : ", names)
print("가족 탄생 해 : ", birth_years)

NohAgeCalculator = AgeCalulator(2023)
ages = NohAgeCalculator.cal_age(birth_years)

print("가족 나이 : ", ages)

# Step2 : 가족 추가 
Noh.add("유미", 1993)
names = Noh.get_family_full_names()
birth_years = Noh.get_family_full_birth()
ages = NohAgeCalculator.cal_age(birth_years)

print("가족 이름 : ", names)
print("가족 탄생 해 : ", birth_years)
print("가족 나이 : ", ages)