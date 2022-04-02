import numpy as np
import pandas as pd
from context.models import Model
from icecream import ic


class SpaceshipModel:
    model = Model()

    def preprocess(self, train, test):
        this = self.model.dataset
        this.train = self.model.load_dataset(train)
        this.test = self.model.load_dataset(test)
        this.id = this.test['PassengerId']
        this.label = this.train['Transported']
        self.drop_feature(this, 'Transported', 'Name')
        self.homePlanet_nominal(this)
        self.CryoSleep_nominal(this)
        self.destination_nominal(this)
        self.age_ratio(this)
        self.vip_nominal(this)
        self.roomService_ratio(this)
        self.foodCourt_ratio(this)
        self.shoppingMall_ratio(this)
        self.spa_ratio(this)
        self.vRDeck_ratio(this)


        self.print_head(this.train, 60)
        self.print_head(this.test, 20)

    @staticmethod
    def print_head(df, num=5):
        ic(df.head(num))

    @staticmethod
    def drop_feature(this, *feature):
        [i.drop(list(feature), axis=1, inplace=True) for i in [this.train, this.test]]
        return this

    @staticmethod
    def make_nominal(this, title, map_dict):
        for these in [this.train, this.test]:
            these[title].fillna('Null', inplace=True)
            these[title] = these[title].map(map_dict)
        return this

    @staticmethod
    def homePlanet_nominal(this) -> None:
        """
        거주(출발) 행성   /   인원수 /   전송 확률
        Earth   /   4602    /   42.39
        Europa  /   2131    /   65.88
        Mars    /   1759    /   52.30
        Null    /   201     /   51.24
        """
        map_homePlanet = {'Earth': 1, 'Europa': 2, 'Mars': 3, 'Null': 3}
        return SpaceshipModel.make_nominal(this, 'HomePlanet', map_homePlanet)

    @staticmethod
    def CryoSleep_nominal(this) -> None:
        """
        캡슐에서 수면 취하는 것
        냉동수면? 같은 느낌
        수면 여부   /   인원수 /   전송 확률
        True   /   3037    /   81.76
        False  /   5439    /   32.89
        Null    /   217     /   48.85
        """
        map_cryosleep = {False: 0, True: 1, 'Null': 0}
        return SpaceshipModel.make_nominal(this, 'CryoSleep', map_cryosleep)

    @staticmethod
    def split_info_from_Cabin(this):
        this['Title'] = this['Cabin'].str.split('/')
        this['Title'].fillna('Null', inplace=True)
        # print(this['Title'].head(20))
        for i in this['Title']:
            print(this['Title'][i])
        return this

    @staticmethod
    def cTP_Cabin(this) -> None:
        """
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
        [ic(j, 100 * i / j) for i, j in zip(list(cnt_home.values()), this['Cabin'].value_counts())]

    @staticmethod
    def destination_nominal(this):
        """
            목적지       /   인원수   /  전송 확률
        TRAPPIST-1e    /   5915    /   47.12
        55 Cancri e    /   1800    /   61.0
        PSO J318.5-22  /   796     /   50.38
        Null           /   182     /   50.55
        """
        map_destination = {'TRAPPIST-1e': 1, '55 Cancri e': 2, 'PSO J318.5-22': 3, 'Null': 3}
        return SpaceshipModel.make_nominal(this, 'Destination', map_destination)

    @staticmethod
    def age_ratio(this):
        """
        연령  /  인원수   /   전송 확률
        1    /  466    /    76.82
        2    /  299    /    59.20
        3    /  780    /    56.15
        4    /  1784   /    45.96
        5    /  2501   /    47.42
        6    /  2430   /    48.97
        7    /  254    /    46.85
        0    /  179    /    50.28
        """
        title = 'Age'
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        labels = [0, 1, 2, 3, 4, 5, 6, 7]
        for these in [this.train, this.test]:
            these[title].fillna(-0.5, inplace=True)
            these[title] = pd.cut(these[title], bins=bins, right=False, labels=labels)
        return this

    @staticmethod
    def vip_nominal(this):
        """
        VIP      /  인원수   /   전송 확률
        True    /  199   /    38.19
        False    /  8291   /    50.63
        Null    /  203   /    51.23
        """
        map_vip = {True: 1, False: 2, 'Null': 2}
        return SpaceshipModel.make_nominal(this, 'VIP', map_vip)

    @staticmethod
    def roomService_ratio(this):
        """
        RS_Cnt /  인원수   /   전송 확률
        0      /  181    /    45.86
        1      /  5999   /    61.44
        10     /  598    /    36.12
        100    /  1314   /    23.97
        1000   /  601    /    12.98
        """
        title = 'RoomService'
        bins = [-1, 0, 10, 100, 1000, np.inf]
        labels = [0, 1, 2, 3, 4]
        for these in [this.train, this.test]:
            these[title].fillna(-0.5, inplace=True)
            these[title] = pd.cut(these[title], bins=bins, right=False, labels=labels)
        return this

    @staticmethod
    def foodCourt_ratio(this):
        """
        FC_Cnt /  인원수   /   전송 확률
        0      /  183    /    54.10
        1      /  5892   /    56.33
        10     /  576    /    21.53
        100    /  1178   /    31.49
        1000   /  864    /    53.82
        """
        title = 'FoodCourt'
        bins = [-1, 0, 10, 100, 1000, np.inf]
        labels = [0, 1, 2, 3, 4]
        for these in [this.train, this.test]:
            these[title].fillna(-0.5, inplace=True)
            these[title] = pd.cut(these[title], bins=bins, right=False, labels=labels)
        return this

    @staticmethod
    def shoppingMall_ratio(this):
        """
        SM_Cnt /  인원수   /   전송 확률
        0      /  208    /    54.81
        1      /  6088   /    57.03
        10     /  708    /    20.48
        100    /  1307   /    32.75
        1000   /  382    /    57.33
        """
        title = 'ShoppingMall'
        bins = [-1, 0, 10, 100, 1000, np.inf]
        labels = [0, 1, 2, 3, 4]
        for these in [this.train, this.test]:
            these[title].fillna(-0.5, inplace=True)
            these[title] = pd.cut(these[title], bins=bins, right=False, labels=labels)
        return this

    @staticmethod
    def spa_ratio(this):
        """
        Spa_Cnt /  인원수   /   전송 확률
        0      /  183    /    49.73
        1      /  5850   /    61.81
        10     /  704    /    36.65
        100    /  1327   /    25.85
        1000   /  629    /    11.13
        """
        title = 'Spa'
        bins = [-1, 0, 10, 100, 1000, np.inf]
        labels = [0, 1, 2, 3, 4]
        for these in [this.train, this.test]:
            these[title].fillna(-0.5, inplace=True)
            these[title] = pd.cut(these[title], bins=bins, right=False, labels=labels)
        return this

    @staticmethod
    def vRDeck_ratio(this):
        """
        VR_Cnt /  인원수   /   전송 확률
        0      /  188    /    52.13
        1      /  5974   /    60.83
        10     /  675    /    34.96
        100    /  1235   /    27.13
        1000   /  621    /    12.08
        """
        title = 'VRDeck'
        bins = [-1, 0, 10, 100, 1000, np.inf]
        labels = [0, 1, 2, 3, 4]
        for these in [this.train, this.test]:
            these[title].fillna(-0.5, inplace=True)
            these[title] = pd.cut(these[title], bins=bins, right=False, labels=labels)
        return this
