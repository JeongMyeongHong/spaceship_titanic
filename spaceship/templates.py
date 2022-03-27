import numpy as np
from icecream import ic
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
        # self.cTP_HomePlanet(this)
        # self.cTP_Cryosleep(this)
        self.cTP_Cabin(this) # 캐빈만 분석하면 됨.
        # self.cTP_Destination(this)
        # self.cTP_Age(this)
        # self.cTP_Vip(this)
        # self.cTP_RoomService(this)
        # self.cTP_FoodCourt(this)
        # self.cTP_ShoppingMall(this)
        # self.cTP_Spa(this)
        # self.cTP_VRDeck(this)

    @staticmethod
    def plus_value(dict, i):
        dict[i] += 1

    @staticmethod
    def cTP_HomePlanet(this) -> None:
        """
        거주(출발) 행성   /   인원수 /   전송 확률
        Earth   /   4602    /   42.39
        Europa  /   2131    /   65.88
        Mars    /   1759    /   52.30
        Null    /   201     /   51.24
        """
        this['HomePlanet'].fillna('Null', inplace=True)
        ic(this['HomePlanet'].value_counts())
        cnt_home = {'Earth': 0, 'Europa': 0, 'Mars': 0, 'Null': 0}
        [SpaceTemplates.plus_value(cnt_home, i) if j is True else None
         for i, j in zip(this['HomePlanet'], this['Transported'])]
        [ic(100 * i / j) for i, j in zip(list(cnt_home.values()), this['HomePlanet'].value_counts())]

    @staticmethod
    def cTP_Cryosleep(this) -> None:
        """
        캡슐에서 수면 취하는 것
        냉동수면? 같은 느낌
        수면 여부   /   인원수 /   전송 확률
        False  /   5439    /   32.89
        True   /   3037    /   81.76
        Null    /   217     /   48.85
        """
        this['CryoSleep'].fillna('Null', inplace=True)
        ic(this['CryoSleep'].value_counts())
        cnt_home = {False: 0, True: 0, 'Null': 0}
        [SpaceTemplates.plus_value(cnt_home, i) if j is True else None
         for i, j in zip(this['CryoSleep'], this['Transported'])]
        [ic(j, 100 * i / j) for i, j in zip(list(cnt_home.values()), this['CryoSleep'].value_counts())]

    @staticmethod
    def split_info_from_Cabin(this):
        this['Title'] = this['Cabin'].str.split('/')
        this['Title'].fillna('Null', inplace=True)
        print(this['Title'].values[0][0])
        return this

    @staticmethod
    def cTP_Cabin(this) -> None:
        """
        마지막에 다시 하자...
        The cabin number where the passenger is staying.
                승객이 머물고 있는 캐빈 번호.
        Takes the form deck/num/side, where side can be either P for Port or S for Starboard
                deck/num/side의 형태를 취한다, side는 좌측(P) 또는 우측(S) 이다.
        deck   인원수 /   전송 확률
        F    2794   /
        G    2559   /
        E     876   /
        B     779   /
        C     747   /
        D     478   /
        A     256   /
        T       5   /
        Null     199/

        num  인원수 /   전송 확률
        1       3439
        2       1219
        3        686
        5        626
        4        590
        9        509
        8        486
        6        469
        7        452
        Null     199
        0         18


        """
        this['Cabin'].fillna('Null', inplace=True)
        cnt_home = {True: 0, False: 0, 'Null': 0}
        [SpaceTemplates.plus_value(cnt_home, i) if j is True else None
         for i, j in zip(this['Cabin'], this['Transported'])]
        [ic(j, 100 * i / j) for i, j in zip(list(cnt_home.values()), this['Cabin'].value_counts())]

    @staticmethod
    def cTP_Destination(this):
        """
            목적지       /   인원수   /  전송 확률
        TRAPPIST-1e    /   5915    /   47.12
        55 Cancri e    /   1800    /   61.0
        PSO J318.5-22  /   796     /   50.38
        Null           /   182     /   50.55
        """
        this['Destination'].fillna('Null', inplace=True)
        print(this['Destination'].value_counts())
        cnt_destination = {'TRAPPIST-1e': 0, '55 Cancri e': 0, 'PSO J318.5-22': 0, 'Null': 0}
        [SpaceTemplates.plus_value(cnt_destination, i) if j is True else None
         for i, j in zip(this['Destination'], this['Transported'])]
        [ic(100 * i / j) for i, j in zip(list(cnt_destination.values()), this['Destination'].value_counts())]

    @staticmethod
    def cTP_Age(this):
        """
        연령  /  인원수   /   전송 확률
        5    /  2501   /    47.42
        6    /  2430   /    48.97
        4    /  1784   /    45.96
        3    /  780    /    56.15
        1    /  466    /    76.82
        2    /  299    /    59.20
        7    /  254    /    46.85
        0    /  179    /    50.28
        """
        this['Age'].fillna(-0.5, inplace=True)
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        labels = [0, 1, 2, 3, 4, 5, 6, 7]
        this['Age'] = pd.cut(this['Age'], bins=bins, right=False, labels=labels)
        print(this['Age'].value_counts())

        cnt_destination = {5: 0, 6: 0, 4: 0, 3: 0, 1: 0, 2: 0, 7: 0, 0: 0}
        [SpaceTemplates.plus_value(cnt_destination, i) if j is True else None
         for i, j in zip(this['Age'], this['Transported'])]
        [ic(100 * i / j) for i, j in zip(list(cnt_destination.values()), this['Age'].value_counts())]

    @staticmethod
    def cTP_Vip(this):
        """
        VIP      /  인원수   /   전송 확률
        False    /  2430   /    50.63
        Null    /  1784   /    51.23
        True    /  2501   /    38.19
        """
        this['VIP'].fillna('Null', inplace=True)
        print(this['VIP'].value_counts())
        cnt_vip = {False: 0, 'Null': 0, True: 0}
        [SpaceTemplates.plus_value(cnt_vip, i) if j is True else None
         for i, j in zip(this['VIP'], this['Transported'])]
        [ic(j, 100 * i / j) for i, j in zip(list(cnt_vip.values()), this['VIP'].value_counts())]

    @staticmethod
    def cTP_RoomService(this):
        """
        RS_Cnt /  인원수   /   전송 확률
        1      /  5999   /    61.44
        100    /  1314   /    23.97
        1000   /  601    /    12.98
        10     /  598    /    36.12
        0      /  181    /    45.86
        """
        this['RoomService'].fillna(-0.5, inplace=True)
        bins = [-1, 0, 10, 100, 1000, np.inf]
        labels = [0, 1, 10, 100, 1000]
        this['RoomService'] = pd.cut(this['RoomService'], bins=bins, right=False, labels=labels)
        print(this['RoomService'].value_counts())

        cnt_RoomService = {1: 0, 100: 0, 1000: 0, 10: 0, 0: 0}
        [SpaceTemplates.plus_value(cnt_RoomService, i) if j is True else None
         for i, j in zip(this['RoomService'], this['Transported'])]
        [ic(100 * i / j) for i, j in zip(list(cnt_RoomService.values()), this['RoomService'].value_counts())]

    @staticmethod
    def cTP_FoodCourt(this):
        """
        FC_Cnt /  인원수   /   전송 확률
        1      /  5892   /    56.33
        100    /  1178   /    31.49
        1000   /  864    /    53.82
        10     /  576    /    21.53
        0      /  183    /    54.10
        """
        this['FoodCourt'].fillna(-0.5, inplace=True)
        bins = [-1, 0, 10, 100, 1000, np.inf]
        labels = [0, 1, 10, 100, 1000]
        this['FoodCourt'] = pd.cut(this['FoodCourt'], bins=bins, right=False, labels=labels)
        print(this['FoodCourt'].value_counts())

        cnt_FoodCourt = {1: 0, 100: 0, 1000: 0, 10: 0, 0: 0}
        [SpaceTemplates.plus_value(cnt_FoodCourt, i) if j is True else None
         for i, j in zip(this['FoodCourt'], this['Transported'])]
        [ic(100 * i / j) for i, j in zip(list(cnt_FoodCourt.values()), this['FoodCourt'].value_counts())]

    @staticmethod
    def cTP_ShoppingMall(this):
        """
        SM_Cnt /  인원수   /   전송 확률
        1      /  6088   /    57.03
        100    /  1307   /    32.75
        10     /  708    /    20.48
        1000   /  382    /    57.33
        0      /  208    /    54.81
        """
        this['ShoppingMall'].fillna(-0.5, inplace=True)
        bins = [-1, 0, 10, 100, 1000, np.inf]
        labels = [0, 1, 10, 100, 1000]
        this['ShoppingMall'] = pd.cut(this['ShoppingMall'], bins=bins, right=False, labels=labels)
        print(this['ShoppingMall'].value_counts())

        cnt_ShoppingMall = {1: 0, 100: 0, 10: 0, 1000: 0, 0: 0}
        [SpaceTemplates.plus_value(cnt_ShoppingMall, i) if j is True else None
         for i, j in zip(this['ShoppingMall'], this['Transported'])]
        [ic(100 * i / j) for i, j in zip(list(cnt_ShoppingMall.values()), this['ShoppingMall'].value_counts())]

    @staticmethod
    def cTP_Spa(this):
        """
        Spa_Cnt /  인원수   /   전송 확률
        1      /  5850   /    61.81
        100    /  1327   /    25.85
        10     /  704    /    36.65
        1000   /  629    /    11.13
        0      /  183    /    49.73
        """
        this['Spa'].fillna(-0.5, inplace=True)
        bins = [-1, 0, 10, 100, 1000, np.inf]
        labels = [0, 1, 10, 100, 1000]
        this['Spa'] = pd.cut(this['Spa'], bins=bins, right=False, labels=labels)
        print(this['Spa'].value_counts())

        cnt_Spa = {1: 0, 100: 0, 10: 0, 1000: 0, 0: 0}
        [SpaceTemplates.plus_value(cnt_Spa, i) if j is True else None
         for i, j in zip(this['Spa'], this['Transported'])]
        [ic(100 * i / j) for i, j in zip(list(cnt_Spa.values()), this['Spa'].value_counts())]

    @staticmethod
    def cTP_VRDeck(this):
        """
        VR_Cnt /  인원수   /   전송 확률
        1      /  5974   /    60.83
        100    /  1235   /    27.13
        10     /  675    /    34.96
        1000   /  621    /    12.08
        0      /  188    /    52.13
        """
        this['VRDeck'].fillna(-0.5, inplace=True)
        bins = [-1, 0, 10, 100, 1000, np.inf]
        labels = [0, 1, 10, 100, 1000]
        this['VRDeck'] = pd.cut(this['VRDeck'], bins=bins, right=False, labels=labels)
        print(this['VRDeck'].value_counts())

        cnt_VRDeck = {1: 0, 100: 0, 10: 0, 1000: 0, 0: 0}
        [SpaceTemplates.plus_value(cnt_VRDeck, i) if j is True else None
         for i, j in zip(this['VRDeck'], this['Transported'])]
        [ic(100 * i / j) for i, j in zip(list(cnt_VRDeck.values()), this['VRDeck'].value_counts())]
