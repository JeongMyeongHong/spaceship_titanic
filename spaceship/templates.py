from icecream import ic
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc, font_manager
import pandas as pd

from context.models import Model

rc('font', family=font_manager.FontProperties(fname='C:/Windows/Fonts/gulim.ttc').get_name())


class SpaceTemplates:
    model = Model()

    def __init__(self, file_name):
        self.file_name = file_name
        self.entity = self.model.load_dataset(file_name)

    def visualize(self) -> None:
        this = self.entity
        # ic(f'{self.file_name.split(".")[0]}의 컬럼 : {this.columns}')
        # ic(f'{self.file_name.split(".")[0]}의 상위 5행 : {this.head()}')
        this = self.extract_title_from_name(this)
        ic(this.head(20))
        # self.drawVRDeck(this)

    @staticmethod
    def plus_value(dict, i):
        dict[i] += 1

    @staticmethod
    def calc_trans_probability_HomePlanet(this) -> None:
        """
        거주(출발) 행성   /   인원수 /   전송 확률
        Earth   /   4602    /   30.51
        Europa  /   2131    /   91.55
        Mars    /   1759    /   52.30
        Null    /   201     /   51.24
        """
        this['HomePlanet'].fillna('Null', inplace=True)
        ic(this['HomePlanet'].value_counts())
        cnt_home = {'Europa': 0, 'Earth': 0, 'Mars': 0, 'Null': 0}
        [SpaceTemplates.plus_value(cnt_home, i) if j is True else None
         for i, j in zip(this['HomePlanet'], this['Transported'])]
        [ic(100 * i / j) for i, j in zip(list(cnt_home.values()), this['HomePlanet'].value_counts())]

    @staticmethod
    def calc_trans_probability_Cryosleep(this) -> None:
        """
        캡슐에서 수면 취하는 것
        냉동수면? 같은 느낌
        수면 여부   /   인원수 /   전송 확률
        True   /   3037    /   45.65
        False  /   5439    /   58.91
        Null    /   217     /   48.85
        """
        this['CryoSleep'].fillna('Null', inplace=True)
        ic(this['CryoSleep'].value_counts())
        cnt_home = {True: 0, False: 0, 'Null': 0}
        [SpaceTemplates.plus_value(cnt_home, i) if j is True else None
         for i, j in zip(this['CryoSleep'], this['Transported'])]
        [ic(j, 100 * i / j) for i, j in zip(list(cnt_home.values()), this['CryoSleep'].value_counts())]

    @staticmethod
    def extract_title_from_name(this):
        this['Title'] = this.Cabin.str[0:5:2]
        return this

    @staticmethod
    def calc_trans_probability_Cabin(this) -> None:
        """
        The cabin number where the passenger is staying.
                승객이 머물고 있는 캐빈 번호.
        Takes the form deck/num/side, where side can be either P for Port or S for Starboard
                deck/num/side의 형태를 취한다, side는 좌측(P) 또는 우측(S)이라고 할 수 있다.
        수면 여부   /   인원수 /   전송 확률
        True   /   3037    /   45.65
        False  /   5439    /   58.91
        Null    /   217     /   48.85
        """
        this['Cabin'].fillna('Null', inplace=True)
        ic(this['Cabin'].value_counts())
        cnt_home = {True: 0, False: 0, 'Null': 0}
        [SpaceTemplates.plus_value(cnt_home, i) if j is True else None
         for i, j in zip(this['Cabin'], this['Transported'])]
        [ic(j, 100 * i / j) for i, j in zip(list(cnt_home.values()), this['Cabin'].value_counts())]


    @staticmethod
    def drawVRDeck(this) -> None:
        # labels = [0, 1, 2, 3, 4, 5, 6, 7]
        # this['VRDeck'] = pd.qcut(this['VRDeck'], q=5, duplicates='drop')
        print(this['VRDeck'].value_counts(dropna=False))
