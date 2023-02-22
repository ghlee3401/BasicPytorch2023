class Family:
    def __init__(self, lastname):
        self.lastname = lastname
        self.family_names = ["영심"] # 이름 
        self.family_birth = [1988] # 태어난 해
        self._set_family_full_names() # 클래스 내부의 함수를 시작하자마자 실행합니다.
        self._set_family_full_birth()

    def _set_family_full_names(self): # 클래스 내부에서 사용하는 함수는 언더바(_)를 붙여서 내부에서만 사용하는 것으로 약속합니다. 
        self.family_full_names = list() 
        for name in self.family_names: 
            self.family_full_names.append(self.lastname + name)

    def get_family_full_names(self):
        return self.family_full_names
    
    def _set_family_full_birth(self):
        self.family_full_birth = list() 
        for birth in self.family_birth: 
            self.family_full_birth.append(birth)
        
    def get_family_full_birth(self):
        return self.family_full_birth
        
    def add(self, name, birth): # add 메소드는 family_names에 추가만하고 반환값은 없는 메소드 입니다. 
        self.family_names.append(name)
        self.family_birth.append(birth)
        self._set_family_full_names()
        self._set_family_full_birth()