
import pandas as pd

class UserProfile:

    def __init__(self, interest: list):
        self.interest = interest

    def get_interest(self):
        return self.interest


class ProfileIntegrater:
    profiles = pd.DataFrae()

    def insert_profile(self, profile: UserProfile):
        # TODO: generate a profile dataframe
        return None


user1 = UserProfile(["politics, soccer"])
user2 = UserProfile(["music", "films"])
user3 = UserProfile(["car", "politics"])
user4 = UserProfile(["soccer"])

print(user1.get_interest())
