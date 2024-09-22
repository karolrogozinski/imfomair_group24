import re
import random
import sys

import pandas as pd
import numpy as np

from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import CountVectorizer

import textdistance

from src.models import Model


class DialogSMLogic:
    """ Basic implementation of the logic of dialog restaurant state machine.
        Allows to hold conversation by state transition made base on given sentence.

    States are described in README file. Class refers to DialogSMOutputs in order to get outputs for the user.
    Conversation ends when users sentence speechAct is recognized as 'bye'

    :ivar next_state: state to consider in the next transition
    :ivar current_field: optional variable indicating the current field of conversation if applicable
    :ivar current_speech_act: speech act of previously given sentence
    :ivar dialog_args: additional dialog arguments for DialogSMOutputs class
    :ivar current_restaurant: previously suggested restaurant
    :ivar preferences: dictionary storing user preferences for all fields
    :ivar antipathies: dictionary storing user antipathies for all fields
    :ivar transition_dict: dictionary with references for state functions

    :param possible_choices: dict with lists of possible choices for each field
    :param model: classifier to predict speechActs
    :param vectorizer: vectorizer to used for bow operation on user sentences
    :param restaurants: list containing all possible restaurant that can be suggested to a user
    :param random_suggestion: hyperparameter how often restaurant suggestion should be made without information
        about all fields
    :param max_distance: hyperparameter what is maximum Lavenshtein distance to approve typo in word

    """
    def __init__(self, possible_choices: Dict[str, List[str]], model: Model, vectorizer: CountVectorizer,
                 restaurants: pd.DataFrame, random_suggestion: float = 0.2, max_distance: int = 1) -> None:
        self.next_state: int = 0
        self.current_field: Tuple[str] = tuple()
        self.current_speech_act: str = 'info'
        self.dialog_args: Tuple = tuple()
        self.current_restaurant: pd.DataFrame = pd.DataFrame()

        DialogSMOutputs.get_dialog_option(self.next_state, self.dialog_args)

        self.preferences: dict = {
            'food': [],
            'area': [],
            'pricerange': [],
        }
        self.antipathies: dict = {
            'food': [],
            'area': [],
            'pricerange': [],
        }
        self.transition_dict = {
            0: self.__state_0,
            1: self.__state_1,
            2: self.__state_2,
            3: self.__state_3,
            4: self.__state_4,
            5: self.__state_5,
            6: self.__state_6,
        }

        self.possible_choices: dict = possible_choices
        self.model: Model = model
        self.vectorizer: CountVectorizer = vectorizer
        self.restaurants_base: pd.DataFrame = restaurants
        self.random_suggestion_th: float = random_suggestion
        self.max_distance: int = max_distance

    def state_transition(self, sentence: str):
        self.current_speech_act = self.__recognize_speech_act(sentence)
        if self.current_speech_act == 'bye':
            DialogSMLogic.__exit()
        self.transition_dict[self.next_state](sentence)
        DialogSMOutputs.get_dialog_option(self.next_state, self.dialog_args)

    def __state_0(self, sentence: str) -> None:
        self.next_state = 1
        self.transition_dict[self.next_state](sentence)

    def __state_1(self, sentence: str) -> None:
        info_exists = self.__update_all_preferences(sentence)
        unknown_fields = self.__get_unknown_fields()

        if info_exists:
            if len(unknown_fields) == 0:
                self.next_state = 4
                self.transition_dict[self.next_state](sentence)
            else:
                if np.random.randint(0, 100) / 100 >= self.random_suggestion_th:
                    self.next_state = 3
                    self.dialog_args = tuple([random.choice(unknown_fields)])
                else:
                    self.next_state = 4
                    self.transition_dict[self.next_state](sentence)
            self.current_field = tuple()
        else:
            self.next_state = 2
            self.dialog_args = tuple(unknown_fields)

    def __state_2(self, sentence: str) -> None:
        self.next_state = 1
        self.transition_dict[self.next_state](sentence)

    def __state_3(self, sentence: str) -> None:
        self.current_field = self.dialog_args
        self.next_state = 1
        self.transition_dict[self.next_state](sentence)

    def __state_4(self, sentence: str) -> None:
        self.__find_restaurant()

        if self.current_restaurant.empty:
            self.next_state = 3
            self.dialog_args = tuple([])
        else:
            self.next_state = 5
            tmp_options = [self.current_restaurant.restaurantname.iloc[0]]
            for key in self.preferences.keys():
                info = self.current_restaurant[key] if self.current_restaurant[key].iloc[0] in self.preferences[key] \
                    else pd.Series([''])
                tmp_options.append(info.iloc[0])

            self.dialog_args = tuple(tmp_options)

    def __state_5(self, sentence: str) -> None:
        if self.current_speech_act == 'reqalts':
            if self.__update_all_preferences(sentence):
                self.next_state = 1
            else:
                self.next_state = 4
            self.transition_dict[self.next_state](sentence)
        elif self.current_speech_act in ('negate', 'inform'):
            self.next_state = 1
            self.transition_dict[self.next_state](sentence)
        elif self.current_speech_act in ('request', 'confirm'):
            self.__parse_request(sentence)
            self.next_state = 6

    def __state_6(self, sentence: str) -> None:
        self.next_state = 5
        self.transition_dict[self.next_state](sentence)

    def __recognize_speech_act(self, sentence: str) -> str:
        """Recognize speach act of given sentence using text classification model
        """
        sentence = self.vectorizer.transform(pd.Series(sentence.lower()))
        speech_act = self.model.predict(sentence)[0]

        return speech_act

    def __update_all_preferences(self, sentence: str) -> bool:
        info_exists = False
        for preference in self.current_field or self.preferences.keys():
            if self.__update_preference(sentence, preference):
                info_exists = True
        return info_exists

    def __update_preference(self, sentence: str, field: str) -> bool:
        """ Update preferences for specified field based on given phrase
            and searches for antipathies that appear after the word 'no'.
        """
        negations = re.findall(r'(?:no|not) ([a-zA-Z]+)', sentence)

        info_exists = self.__update_likes(sentence, field, self.max_distance)
        self.__update_dislikes(negations, field, self.max_distance)

        if self.current_field and self.__is_dontcare(sentence) and not info_exists:
            self.preferences[field] = self.possible_choices[field].tolist()
            info_exists = True

        return info_exists

    @staticmethod
    def __is_dontcare(sentence: str) -> bool:
        """ Predict if the user speechAct was dontcare, since is not one of the model class
        """
        dontcare_words = ['any', 'dontcare', 'doesnt matter']
        return any(word in sentence for word in dontcare_words)

    def __update_likes(self, sentence: str, field: str, max_distance: int) -> bool:
        current_word = None
        info_exists = False

        for word in sentence.split():
            for value in self.possible_choices[field]:
                if textdistance.levenshtein(word, value) <= max_distance:
                    current_word = value
                    max_distance = textdistance.levenshtein(word, value)

        if current_word:
            info_exists = True
            if current_word not in self.preferences[field]:
                self.preferences[field].append(current_word)
        return info_exists

    def __update_dislikes(self, negations: List[str], field: str, max_distance: int) -> None:
        current_word = None

        for word in negations:
            for value in self.possible_choices[field]:
                if textdistance.levenshtein(word, value) <= max_distance:
                    current_word = value
                    max_distance = textdistance.levenshtein(word, value)

        if current_word:
            self.antipathies[field].append(current_word)
            self.preferences[field].remove(current_word)

    def __get_unknown_fields(self) -> List[str]:
        """Returns a list of fields for which the user has not specified a preference.
        """
        unknown_fields = [field for field, preferences in self.preferences.items()
                          if len(preferences) == 0]
        return unknown_fields

    def __find_restaurant(self) -> None:
        """ Finds a restaurant based on user likes and dislikes.
            In case of many possibilities, randomly chooses one of them, trying to avoid the previously chosen one.
        """
        possible_flag = 0
        possible_restaurants = pd.DataFrame()

        for key in self.preferences.keys():
            if len(self.preferences[key]) > 0:
                # preferences
                if possible_flag > 0:
                    possible_restaurants = possible_restaurants[possible_restaurants[key].isin(self.preferences[key])]
                else:
                    possible_restaurants = self.restaurants_base[self.restaurants_base[key].isin(self.preferences[key])]
                    possible_flag = 1

            # antipathies
            if possible_flag > 0:
                possible_restaurants = possible_restaurants[~possible_restaurants[key].isin(self.antipathies[key])]
            else:
                possible_restaurants = self.restaurants_base[~self.restaurants_base[key].isin(self.antipathies[key])]
                possible_flag = 1

        print(possible_restaurants.shape, self.current_restaurant.shape)
        if not possible_restaurants.empty and not self.current_restaurant.empty:
            current_name = self.current_restaurant.restaurantname.iloc[0]
            possible_restaurants = possible_restaurants[
                possible_restaurants.restaurantname != current_name]
        if not possible_restaurants.empty:
            self.current_restaurant = possible_restaurants.sample()
        else:
            self.current_restaurant = pd.DataFrame()

    def __parse_request(self, sentence: str) -> None:
        """ Parses the record for types of requested fields.
            Similarly to searching preferences Lavenshtein distance is applied.
        """
        request_dict = {
            'food': ['cuisine', 'food', 'food type', 'dish', 'meal', 'menu', 'entree'],
            'address': ['place', 'where', 'area', 'address', 'postcode', 'addr', 'location', 'street', 'city',
                        'zipcode'],
            'phone': ['telephone', 'phone', 'mobile', 'phone', 'contact', 'cell'],
            'pricerange': ['price', 'price range', 'cost'],
        }

        requests = list()

        for word in sentence.split():
            for request in request_dict.keys():
                for value in request_dict[request]:
                    if textdistance.levenshtein(word, value) <= self.max_distance:
                        requests.append(value)
                        continue

        self.__prepare_requested_fields(list(set(requests)) or [random.choice(list(request_dict.keys()))])

    def __prepare_requested_fields(self, requests: List[str]) -> None:
        """Sets up information for dialogs based on requested fields.
        """
        dialog_args = list()
        dialog_args.append(self.current_restaurant.restaurantname.iloc[0])
        for field in ('food', 'phone', 'pricerange'):
            dialog_args.append(self.current_restaurant[field].iloc[0] if field in requests else '')
        dialog_args.append([self.current_restaurant.area.iloc[0], self.current_restaurant.addr.iloc[0],
                            self.current_restaurant.postcode.iloc[0]] if 'address' in requests else [])

        self.dialog_args = dialog_args

    @staticmethod
    def __exit() -> None:
        """ Close the program
        """
        DialogSMOutputs.get_dialog_option(-1, [])
        sys.exit()


class DialogSMOutputs:
    """Class that implements dialog options for each state of Dialog State Machine.

        Dialog options outputs when entering the state - BEFORE applying state logic.
        Each of them takes dialog options as input to parametrize sentences.
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_dialog_option(state_number: int, options: Tuple = ()) -> None:
        transition_dict = {
            0: DialogSMOutputs.__state_0,
            1: DialogSMOutputs.__state_1,
            2: DialogSMOutputs.__state_2,
            3: DialogSMOutputs.__state_3,
            4: DialogSMOutputs.__state_4,
            5: DialogSMOutputs.__state_5,
            6: DialogSMOutputs.__state_6,
            -1: DialogSMOutputs.__exit,
        }
        return transition_dict[state_number](options)

    @staticmethod
    def __state_0(_: Tuple) -> None:
        print("""Hello, welcome to the Cambridge Restaurant System!
You can search for restaurants by area, price range or cuisine.
How can I help you?""")

    @staticmethod
    def __state_1(_: Tuple) -> None:
        pass

    @staticmethod
    def __state_2(options: Tuple) -> None:
        if len(options) == 0:
            text = 'any preferences'
        elif len(options) == 1:
            text = options[0]
        elif len(options) == 2:
            text = options[0] + ' and ' + options[1]
        else:
            text = options[0] + ', ' + options[1] + ' and ' + options[2]
        print(f"""There is no restaurant with given parameters.
Please provide {text} again.""".replace(
            'food', 'cuisine type').replace('pricerange', 'price range'))

    @staticmethod
    def __state_3(options: Tuple):
        if not options:
            text = ''
        else:
            text = ' ' + options[0] + ' '
        print(f"""I am not sure about the results. Could you provide another{text}preference?""".replace(
            'food', 'cuisine type').replace('_', ' '))

    @staticmethod
    def __state_4(_: Tuple) -> None:
        pass

    @staticmethod
    def __state_5(options: Tuple) -> None:
        if len(options) == 5:
            print('Do you have any more questions or comments about suggested restaurant?')
            return

        name = options[0]
        food = options[1]
        area = options[2]
        price_range = options[3]

        text = 'It is '
        if food and area and price_range:
            text += f'{price_range}, {food} restaurant in the {area} part of the town.'
        elif food and area:
            text += f'{food} restaurant in the {area} part of town.'
        elif area and price_range:
            text += f'{price_range} restaurant in the {area} part of the town.'
        elif food and price_range:
            text += f'{price_range}, {food} restaurant.'
        elif food or price_range:
            text += f'{price_range + food} restaurant.'
        else:
            text += f'restaurant in the {area} part of the town.'

        print(f"""My suggestion is {name}. \n{text}
Do you have any other preferences or this suggestion satisfies you and want to hear more details?""")

    @staticmethod
    def __state_6(options: Tuple) -> None:
        name = options[0]
        food = options[1]
        phone = options[2]
        price_range = options[3]

        text = ''

        if food or price_range or options[4]:
            text += name.title() + ' is '

            if food and price_range:
                text += food + ', ' + price_range + 'restaurant'
            elif food:
                text += food + 'restaurant'
            else:
                text += price_range + 'restaurant'

            if options[4]:
                area = options[4][0]
                address = options[4][1]
                zipcode = options[4][2]
                text += ' located in the ' + area + 'part of the town at ' + address + ', ' + zipcode

            text += '.'

        if phone:
            text += ' To contact them you can call ' + phone + '.'

        print(text or 'Sorry I do not understand your request.')

    @staticmethod
    def __exit(_: Tuple) -> None:
        print('Thank you for using Cambridge Restaurant System. Bye!')
        sys.exit()
