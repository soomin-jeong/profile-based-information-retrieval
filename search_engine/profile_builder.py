
class Profile:

    def __init__(self, name: str, interest: list):
        self.interest = interest
        self.name = name

    def __repr__(self):
        return str(self.interest)

    def get_interests(self):
        return self.interest


class Interest:

    def __init__(self):
        self.profiles = list()
        self.interests = set()

    def insert_profile(self, profile: Profile):
        self.profiles.append(profile)
        self.interests.update(profile.get_interests())

    def get_interests(self):
         return self.interests


# politics, soccer, music, car, films
user1 = Profile("user1", ["politics", "soccer"])
user2 = Profile("user2", ["music", "films"])
user3 = Profile("user3", ["car", "politics"])
user4 = Profile("user4", ["soccer"])

profiles = [user1, user2, user3, user4]

interest_integrater = Interest()

for each in profiles:
    interest_integrater.insert_profile(each)

