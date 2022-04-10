import pandas as pd

from spaceship.models import SpaceshipModel
from spaceship.templates import SpaceTemplates

if __name__ == '__main__':
    def print_menu():
        return '1.템플릿 2.모델 3. 러닝'


    while 1:
        menu = input(print_menu())
        if menu == '1':
            print(' #### 1. 템플릿 #### ')
            templates = SpaceTemplates('train.csv')
            templates.visualize()
        elif menu == '2':
            print(' #### 2. 모델 #### ')
            model = SpaceshipModel()
            model.preprocess(train='train.csv', test='test.csv')

        elif menu == '3':
            print(' #### 3. 러닝 #### ')
            model = SpaceshipModel()
            model.learning(train='train.csv', test='test.csv')
        break
