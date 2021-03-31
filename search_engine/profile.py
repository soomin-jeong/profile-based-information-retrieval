import pandas as pd


class UserProfile:

    def __init__(self, name: str, interest: list):
        self.interest = interest

    def __repr__(self):
        return str(self.interest)


class ProfileIntegrater:

    def __init__(self):
        self.profiles = []

    def insert_profile(self, profile: UserProfile):
        self.profiles.append(profile)
        # TODO: generate a profile dataframe
        return None

    def show_profiles(self):
        print(self.profiles)


# politics, soccer, music, car, films
user1 = UserProfile("user1", ["politics, soccer"])
user2 = UserProfile("user2", ["music", "films"])
user3 = UserProfile("user3", ["car", "politics"])
user4 = UserProfile("user4", ["soccer"])

profile_integrater = ProfileIntegrater()

for each in [user1, user2, user3, user4]:
    profile_integrater.insert_profile(each)


profile_integrater.show_profiles()