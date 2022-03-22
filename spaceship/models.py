from context.models import Model
from icecream import ic


class SpaceshipModel:
    model = Model()

    def preprocess(self, train, test):
        this = self.model.dataset
        this.train = self.model.load_dataset(train)
        this.test = self.model.load_dataset(test)
        self.print_head(this.train, 10)
        self.print_head(this.test, 10)

    @staticmethod
    def print_head(df, num=5):
        ic(df.head(num))
