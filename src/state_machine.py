import re
import random
import sys
import time

import pandas as pd
import numpy as np

from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import CountVectorizer

import textdistance
import pyttsx3

from src.models import Model


DONTCARE_WORDS = ['any', 'dontcare', 'doesnt matter', "don't care", "doesn't matter", 'whatever', 'dont care']
REQUEST_WORDS = {
    'food': ['cuisine', 'food', 'food type', 'dish', 'meal', 'menu', 'entree'],
    'address': ['place', 'where', 'area', 'address', 'addr', 'location', 'street', 'city',],
    'postcode': ['zipcode', 'postcode', 'code'],
    'phone': ['telephone', 'phone', 'mobile', 'phone', 'contact', 'cell'],
    'pricerange': ['price', 'price range', 'cost', 'cheap', 'expensive', 'moderate'],
    # 'food_quality': ['decent', 'fast food', 'good food', 'fast', 'good', 'decent'],
    # 'crowdedness': ['busy', 'quiet'],
    # 'length_of_stay': ['long', 'medium', 'short']
}
POSSIBLE_SECONDARY_PREF_WORDS = {
    'touristic': ['touristic'] ,
    'seats': ['assigned', 'seats'],
    'children': ['child', 'children', 'kid'],
    'romantic': ['romantic', 'anniversary', 'wife', 'husband', 'romance']
}


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
    :param delay: delay before system response (s)

    """
    def __init__(self, possible_choices: Dict[str, List[str]], model: Model, vectorizer: CountVectorizer,
                 restaurants: pd.DataFrame, random_suggestion: float = 0.0, max_distance: int = 1, delay: int = 0,
                 tts: bool = False) -> None:
        self.next_state: int = 0
        self.current_field: Tuple[str] = tuple()
        self.current_speech_act: str = 'info'
        self.dialog_args: Tuple = tuple()
        self.current_restaurant: pd.DataFrame = pd.DataFrame()
        self.suggested_restaurants: pd.DataFrame = pd.DataFrame()
        
        # consequent list, antecedent list and respective truth values
        self.consequents = []
        self.antecedens: dict = {
            'food_quality': None,
            'crowdedness': None,
            'length_of_stay': None,
            'area': None,
            'pricerange': None,
            'food': None
        }
        self.truth_table: dict = {
            'touristic': None,
            'seats': None,
            'romantic': None,
            'children': None
        }

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
        self.secondary_preferences: dict = {
            'touristic': [],
            'seats': [],
            'children': [],
            'romantic': []
        }
        self.secondary_antipathies: dict = {
            'touristic': [],
            'seats': [],
            'children': [],
            'romantic': []
        }
        self.transition_dict = {
            0: self.__state_0,
            1: self.__state_1,
            2: self.__state_2,
            3: self.__state_3,
            4: self.__state_4,
            5: self.__state_5,
            6: self.__state_6,
            7: self.__state_7,
            8: self.__state_8,
        }

        self.possible_choices: dict = possible_choices
        self.model: Model = model
        self.vectorizer: CountVectorizer = vectorizer
        self.restaurants_base: pd.DataFrame = restaurants
        self.random_suggestion_th: float = random_suggestion
        self.max_distance: int = max_distance
        self.delay: int = delay
        self.dialog_machine: DialogSMOutputs = DialogSMOutputs(tts=tts)

        self.dialog_machine.get_dialog_option(self.next_state, self.dialog_args)

    def state_transition(self, sentence: str):
        self.current_speech_act = self.__recognize_speech_act(sentence)
        if self.current_speech_act == 'bye':
            self.__exit()
        self.transition_dict[self.next_state](sentence)
        time.sleep(self.delay)
        self.dialog_machine.get_dialog_option(self.next_state, self.dialog_args)

    def __state_0(self, sentence: str) -> None:
        self.next_state = 1
        self.transition_dict[self.next_state](sentence)

    def __state_1(self, sentence: str) -> None:
        self.__reset_inference_rules()
        info_exists = self.__update_all_preferences(sentence)
        unknown_fields = self.__get_unknown_fields()

        if info_exists:
            if len(unknown_fields) == 0:
                self.next_state = 4
                self.transition_dict[self.next_state](sentence)
            else:
                # This else cannot happen any more since we come to 4 only if we know all the initial preferences.
                if np.random.randint(0, 100) / 100 >= self.random_suggestion_th:
                    self.next_state = 3
                    # self.dialog_args = tuple([random.choice(unknown_fields)])
                    self.dialog_args = tuple(unknown_fields)
                else:
                    self.next_state = 4
                    self.transition_dict[self.next_state](sentence)
            self.current_field = tuple()
        else:
            self.next_state = 2
            self.dialog_args = tuple(unknown_fields)

    def __state_2(self, sentence: str) -> None:
        if self.current_speech_act in ('inform', 'confirm', 'reqalts', 'negate'):
            self.next_state = 1
            self.transition_dict[self.next_state](sentence)

    def __state_3(self, sentence: str) -> None:
        self.__reset_inference_rules()
        if self.current_speech_act in ('inform', 'confirm', 'reqalts', 'negate'):
            self.current_field = self.dialog_args
            self.next_state = 1
            self.transition_dict[self.next_state](sentence)
        elif self.current_speech_act in ('request', 'confirm'):
            self.__parse_request(sentence)
            self.next_state = 6

    def __state_4(self, sentence: str) -> None:
        found_new_restaurant = self.__find_restaurant()
        # if there no restaurant matching the initial preferences, go to state 3
        if self.current_restaurant.empty:
            self.next_state = 3
            self.dialog_args = tuple([])

        # if there are some restaurants matching the preferences
        else:
            self.next_state = 8
            tmp_options = [self.current_restaurant.restaurantname.iloc[0]]

            if found_new_restaurant:
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
            self.next_state = 7
            self.transition_dict[self.next_state](sentence)
        elif self.current_speech_act in ('request', 'confirm'):
            self.__parse_request(sentence)
            self.next_state = 6

    def __state_6(self, sentence: str) -> None:
        self.next_state = 5
        self.transition_dict[self.next_state](sentence)

    def __state_7(self, sentence: str) -> None:
        self.suggested_restaurants = pd.DataFrame()
        self.current_restaurant = pd.DataFrame()
        self.next_state = 1
        self.transition_dict[self.next_state](sentence)

    def __state_8(self, sentence: str) -> None:
        # some consuqents create speech act null, so here we include null too.
        suggested = False
        if self.current_speech_act in ('inform', 'confirm', 'reqalts', 'null', 'negate'):

            # consequents to work on
            consequent_list = []

            info_exists = self.__update_all_preferences(sentence, from_8=True)
            
            # I need to make the matching now.
            pref_exists = False
            for pref in self.secondary_preferences.values():
                if len(pref) > 0: #or FIRST_TIME_8:
                    pref_exists = True
                    break
            if pref_exists:
                # we do the rule matching
                # first check for the field of POSSIBLE_SECONDARY_PREF_WORDS
                
                for field in POSSIBLE_SECONDARY_PREF_WORDS.keys():
                    if len(self.secondary_preferences[field]) > 0:
                        consequent_list.append(field)

                # once we have the list we start matching rules for each of them.
                # TODO: I could have done it in the for loop above too but gonna take a look at it if I have time.
                for index, restaurant in self.possible_restaurants.iterrows():
                    for consequent in consequent_list:

                        t_val = self.__try_all_rules(restaurant=restaurant, consequent=consequent)

                        # enter the truth value for the consequent.
                        # self.truth_table[consequent] = t_val
                        if not t_val:
                            # if any of the rules evaluate to false for a restaurant, remove the restaurant from possible restaurants list.
                            self.possible_restaurants = self.possible_restaurants[self.possible_restaurants['restaurantname'] != restaurant['restaurantname']]
                
                # sample a new current restaurant
                if not self.possible_restaurants.empty and not self.suggested_restaurants.empty:
                    self.possible_restaurants = self.possible_restaurants[
                        ~self.possible_restaurants.restaurantname.isin(self.suggested_restaurants.restaurantname)]
                    suggested = True
                if not self.possible_restaurants.empty:
                    self.current_restaurant = self.possible_restaurants.sample()
                    suggested = True
                    if self.suggested_restaurants.empty:
                        self.suggested_restaurants = self.current_restaurant
                        self.last_suggested_restaurant = self.current_restaurant
                    else:
                        self.suggested_restaurants = pd.concat([self.suggested_restaurants, self.current_restaurant],
                                                            ignore_index=True)
                        self.last_suggested_restaurant = self.current_restaurant
                        
                try:
                    if not self.last_suggested_restaurant.empty:
                        first_suggestion = False
                except:
                    first_suggestion = True

                if self.current_restaurant.empty or first_suggestion:
                    self.next_state = 2
                    self.dialog_args = tuple([])
                else:
                    self.next_state = 5
                    tmp_options = [self.last_suggested_restaurant.restaurantname.iloc[0]]

                    if suggested:
                        for key in self.preferences.keys():
                            info = self.current_restaurant[key] if self.current_restaurant[key].iloc[0] in self.preferences[key] \
                                    else pd.Series([''])
                            tmp_options.append(info.iloc[0])

                        tmp_options.append(self.antecedens)
                        tmp_options.append(self.truth_table)

                    self.dialog_args = tuple(tmp_options)
            # if there are no secondary preferences:
            else:
                self.next_state = 5
                #found_new_restaurant = self.__find_restaurant()

                if not self.possible_restaurants.empty and not self.suggested_restaurants.empty:
                    self.possible_restaurants = self.possible_restaurants[
                        ~self.possible_restaurants.restaurantname.isin(self.suggested_restaurants.restaurantname)]
                    suggested = True
                if not self.possible_restaurants.empty:
                    self.current_restaurant = self.possible_restaurants.sample()
                    suggested = True
                    if self.suggested_restaurants.empty:
                        self.suggested_restaurants = self.current_restaurant
                        self.last_suggested_restaurant = self.current_restaurant
                    else:
                        self.suggested_restaurants = pd.concat([self.suggested_restaurants, self.current_restaurant],
                                                            ignore_index=True)
                        self.last_suggested_restaurant = self.current_restaurant

                tmp_options = [self.current_restaurant.restaurantname.iloc[0]]

                for key in self.preferences.keys():
                    info = self.current_restaurant[key] if self.current_restaurant[key].iloc[0] in self.preferences[key] \
                            else pd.Series([''])
                    tmp_options.append(info.iloc[0])

                self.dialog_args = tuple(tmp_options)

                if self.possible_restaurants.empty:
                    tmp_options = [self.current_restaurant.restaurantname.iloc[0]]
                    self.dialog_args = tuple(tmp_options)
           
    def __recognize_speech_act(self, sentence: str) -> str:
        """Recognize speach act of given sentence using text classification model
        """
        sentence = self.vectorizer.transform(pd.Series(sentence.lower()))
        speech_act = self.model.predict(sentence)[0]

        return speech_act

    def __update_all_preferences(self, sentence: str, from_8=False) -> bool:
        info_exists = False

        if not from_8:
            for preference in self.current_field or self.preferences.keys():
                if self.__update_preference(sentence, preference, from_8=from_8):
                    info_exists = True
        else:
            for second_pref in self.secondary_preferences.keys():
                if self.__update_preference(sentence, second_pref, from_8=from_8):
                    info_exists = True
        return info_exists

    def __update_preference(self, sentence: str, field: str, from_8=False) -> bool:
        """ Update preferences for specified field based on given phrase
            and searches for antipathies that appear after the word 'no'.
        """
        negations = re.findall(r'(?:no|not) ([a-zA-Z]+)', sentence)

        info_exists = self.__update_likes(sentence, field, self.max_distance, from_8=from_8)
        self.__update_dislikes(negations, field, self.max_distance, from_8=from_8)

        if self.current_field and self.__is_dontcare(sentence) and not info_exists:
            self.preferences[field] = self.possible_choices[field].tolist()
            info_exists = True

        return info_exists

    @staticmethod
    def __is_dontcare(sentence: str) -> bool:
        """ Predict if the user speechAct was dontcare, since is not one of the model class
        """
        return any(word in sentence for word in DONTCARE_WORDS)

    def __update_likes(self, sentence: str, field: str, max_distance: int, from_8=False) -> bool:
        current_word = None
        info_exists = False

        if not from_8:
            for word in sentence.split():
                for value in self.possible_choices[field]:
                    if textdistance.levenshtein(word, value) <= max_distance:
                        current_word = value
                        max_distance = textdistance.levenshtein(word, value)
        else:
            for word in sentence.split():
                for value in POSSIBLE_SECONDARY_PREF_WORDS[field]:
                    if textdistance.levenshtein(word, value) <= max_distance:
                        current_word = value
                        max_distance = textdistance.levenshtein(word, value)


        if current_word:
            info_exists = True
            if not from_8:
                if current_word not in self.preferences[field]:
                    self.preferences[field] = [current_word]
                    if current_word in self.antipathies[field]:
                        self.antipathies[field].remove(current_word)
            else:
                if current_word not in self.secondary_preferences[field]:
                    self.secondary_preferences[field] = [current_word]
                    if current_word in self.secondary_antipathies[field]:
                        self.secondary_preferences[field].remove(current_word)

        return info_exists

    def __update_dislikes(self, negations: List[str], field: str, max_distance: int, from_8=False) -> None:
        current_word = None

        for word in negations:
            for value in self.possible_choices[field]:
                if textdistance.levenshtein(word, value) <= max_distance:
                    current_word = value
                    max_distance = textdistance.levenshtein(word, value)

        if current_word:
            if not from_8:
                self.antipathies[field].append(current_word)
                if current_word in self.preferences[field]:
                    self.preferences[field].remove(current_word)
            #else:
            #    self.secondary_antipathies[field].append(current_word)
            #    if current_word in self.secondary_preferences[field]:
            #        self.secondary_preferences[field].remove(current_word)

    def __get_unknown_fields(self) -> List[str]:
        """Returns a list of fields for which the user has not specified a preference.
        """
        unknown_fields = [field for field, preferences in self.preferences.items()
                          if len(preferences) == 0]
        return unknown_fields

    def __try_all_rules(self, restaurant, consequent) -> bool:
        truth_table = {'touristic': None,
                       'seats': None,
                       'romantic': None,
                       'children': None}

        if consequent == "touristic":
            # Rule 1:
            if restaurant['food_quality'] == "good" and restaurant['pricerange'] == "cheap":
                truth_table['touristic'] = True
                self.antecedens["food_quality"] = "good"
                self.antecedens["pricerange"] = "cheap"
                self.antecedens["food"] = "not romanian"
            else:
                truth_table['touristic'] = False
            # Rule 2: here False overrides True due to order, so contradiction resolved.
            if restaurant['food'] == "romanian":
                truth_table['touristic'] = False
                
        if consequent == "seats":
            # Rule 3:
            if restaurant['crowdedness'] == 'busy':
                truth_table['seats'] = True
                self.antecedens["crowdedness"] = "busy"
            else:
                truth_table['seats'] = False

        if consequent == "children":
            # Rule 4:
            if restaurant['length_of_stay'] == "long":
                truth_table['children'] = False
            # Rule 4.5:
            else:
                truth_table['children'] = True
                self.antecedens["length_of_stay"] = "not long"
            
        if consequent == "romantic":
            # Rules 6: here False overrides True due to order, so contradiction resolved.
            if restaurant["length_of_stay"] == "long":
                truth_table['romantic'] = True
                self.antecedens["food_quality"] = "good"
                self.antecedens["crowdedness"] = "not busy"
            else:
                truth_table['romantic'] = False
            # Rules 5:
            if restaurant["crowdedness"] == "busy":
                truth_table['romantic'] = False
            
        # for the restaurant, if any field evaluates to 0, return False:
        # Since this function already runs for only 1 consequent, we can update the truth table of the class directly.
        for t_val in truth_table.values():
            if t_val == False:
                return False
            
        for key, val in truth_table.items():
            if val == True:
                self.truth_table[consequent] = True
            
        self.consequents.append(consequent)
        return True
                        
    
    def __reset_inference_rules(self):
        """ In states 1 and 3, we need to reset inference rule data. """
        self.consequents = []
        self.antecedens: dict = {
            'food_quality': None,
            'crowdedness': None,
            'length_of_stay': None,
            'area': None,
            'pricerange': None,
            'food': None
        }
        self.truth_table: dict = {
            'touristic': None,
            'seats': None,
            'romantic': None,
            'children': None
        }
        self.secondary_preferences: dict = {
            'touristic': [],
            'seats': [],
            'romantic': [],
            'children': []
        }

    def __find_restaurant(self) -> bool:
        """ Finds a restaurant based on user likes and dislikes.
            In case of many possibilities, randomly chooses one of them, trying to avoid the previously chosen one.
        """
        possible_flag = 0
        self.possible_restaurants = pd.DataFrame()
        found_new_restaurant = False

        for key in self.preferences.keys():
            if len(self.preferences[key]) > 0:
                # preferences
                if possible_flag > 0:
                    self.possible_restaurants = self.possible_restaurants[self.possible_restaurants[key].isin(self.preferences[key])]
                else:
                    self.possible_restaurants = self.restaurants_base[self.restaurants_base[key].isin(self.preferences[key])]
                    possible_flag = 1

            # antipathies
            if possible_flag > 0:
                self.possible_restaurants = self.possible_restaurants[~self.possible_restaurants[key].isin(self.antipathies[key])]
            else:
                self.possible_restaurants = self.restaurants_base[~self.restaurants_base[key].isin(self.antipathies[key])]
                possible_flag = 1

        if not self.possible_restaurants.empty and not self.suggested_restaurants.empty:
            self.possible_restaurants = self.possible_restaurants[
                ~self.possible_restaurants.restaurantname.isin(self.suggested_restaurants.restaurantname)]
        if not self.possible_restaurants.empty:
            self.current_restaurant = self.possible_restaurants.sample()
            found_new_restaurant = True
        # else:
        #     self.current_restaurant = pd.DataFrame()

        return found_new_restaurant

    def __parse_request(self, sentence: str) -> None:
        """ Parses the record for types of requested fields.
            Similarly to searching preferences Lavenshtein distance is applied.
        """
        requests = list()

        for word in sentence.split():
            for request in REQUEST_WORDS.keys():
                for value in REQUEST_WORDS[request]:
                    if textdistance.levenshtein(word, value) <= self.max_distance:
                        requests.append(request)
                        continue

        self.__prepare_requested_fields(list(set(requests)) or [random.choice(list(REQUEST_WORDS.keys()))])

    def __prepare_requested_fields(self, requests: List[str]) -> None:
        """Sets up information for dialogs based on requested fields.
        """
        try:
            self.current_restaurant = self.last_suggested_restaurant
        except:
            pass
        dialog_args = list()
        dialog_args.append(self.current_restaurant.restaurantname.iloc[0])
        for field in ('food', 'phone', 'pricerange', 'postcode'):
            dialog_args.append(self.current_restaurant[field].iloc[0] if field in requests else '')
        dialog_args.append([self.current_restaurant.area.iloc[0], self.current_restaurant.addr.iloc[0],
                            ] if 'address' in requests else [])

        self.dialog_args = dialog_args

    def __exit(self) -> None:
        """ Close the program
        """
        self.dialog_machine.get_dialog_option(-1, ())
        sys.exit()


class DialogSMOutputs:
    """Class that implements dialog options for each state of Dialog State Machine.

        Dialog options outputs when entering the state - BEFORE applying state logic.
        Each of them takes dialog options as input to parametrize sentences.
    """
    def __init__(self, tts: bool = False) -> None:
        self.tts = tts
        if tts:
            self.tts_engine = pyttsx3.init()

    def get_dialog_option(self, state_number: int, options: Tuple = ()) -> None:
        transition_dict = {
            0: DialogSMOutputs.__state_0,
            1: DialogSMOutputs.__state_1,
            2: DialogSMOutputs.__state_2,
            3: DialogSMOutputs.__state_3,
            4: DialogSMOutputs.__state_4,
            5: DialogSMOutputs.__state_5,
            6: DialogSMOutputs.__state_6,
            7: DialogSMOutputs.__state_7,
            8: DialogSMOutputs.__state_8,
            -1: DialogSMOutputs.__exit,
        }
        text = transition_dict[state_number](options)
        print('ASSISTANT: ', text)
        if self.tts:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    @staticmethod
    def __state_0(_: Tuple) -> str:
        text = """Hello, welcome to the Cambridge Restaurant System!
You can search for restaurants by area, price range or cuisine.
How can I help you?"""
        return text

    @staticmethod
    def __state_1(_: Tuple) -> None:
        pass

    @staticmethod
    def __state_2(options: Tuple) -> str:
        if len(options) == 0:
            text = 'any preferences'
        elif len(options) == 1:
            text = options[0]
        elif len(options) == 2:
            text = options[0] + ' and ' + options[1]
        else:
            text = options[0] + ', ' + options[1] + ' and ' + options[2]
        text = f"""There is no restaurant with given parameters.
Please provide {text} again.""".replace(
            'food', 'cuisine type').replace('pricerange', 'price range')
        return text

    @staticmethod
    def __state_3(options: Tuple) -> str:
        pref = ''
        text = 'I am not sure about the results.'

        # find which preferences are missing
        # missing_preferences = ""
        
        join_options = lambda options: ', '.join(options)
        text += f" The missing fields are: {join_options(options)}."
        #if len(options) == 1:
        #    pref = options[0] + ' '

        #text += f""" Could you provide another {pref}preference?""".replace(
        #    'food', 'cuisine type').replace('_', ' ')
        return text

    @staticmethod
    def __state_4(_: Tuple) -> None:
        pass

    @staticmethod
    def __state_5(options: Tuple) -> str:
        if len(options) == 1:
            text = f'There are no more options, so as the last one was {options[0]}. \n'
        # elif len(options) == 5:
        #    text = 'Do you have any more questions or comments about suggested restaurant?'
        else:
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

            text = f'My suggestion is {name}. \n{text}'


            # now we work on the inference related preferences:

            if len(options) > 4:
                # consequents = options[5].keys()
                truth_table = options[5]
                antecedents = options[4]
                antecedent_string = ""
                for key, val in truth_table.items():
                    # if the truth table says true for a consequent
                    if val:
                        consequent = key
                        # then we build the antecedent list
                        if antecedents["food_quality"]:
                            if antecedent_string != "":
                                antecedent_string = f"{antecedent_string} and the food quality is {antecedents["food_quality"]}"
                            else:
                                antecedent_string = f"{antecedent_string} the food quality is {antecedents["food_quality"]}"
                        if antecedents["crowdedness"]:
                            if antecedent_string != "":
                                antecedent_string = f"{antecedent_string} and the occupancy is {antecedents["crowdedness"]}"
                            else:
                                antecedent_string = f"{antecedent_string} the occupancy is {antecedents["crowdedness"]}"
                        if antecedents["length_of_stay"]:
                            if antecedent_string != "":
                                antecedent_string = f"{antecedent_string} and allowed stay duration is {antecedents["length_of_stay"]}"
                            else:
                                antecedent_string = f"{antecedent_string} the allowed stay duration is {antecedents["length_of_stay"]}"
                        if antecedents["area"]:
                            if antecedent_string != "":
                                antecedent_string = f"{antecedent_string} and is in {antecedents["area"]}"
                            else:
                                antecedent_string = f"{antecedent_string} is not in {antecedents["area"]}"
                        if antecedents["pricerange"]:
                            if antecedent_string != "":
                                antecedent_string = f"{antecedent_string} and pricerange is {antecedents["pricerange"]}"
                            else:
                                antecedent_string = f"{antecedent_string} the pricerange is {antecedents["pricerange"]}"
                        if antecedents["food"]:
                            if antecedent_string != "":
                                antecedent_string = f"{antecedent_string} and the cousine is {antecedents["food"]}"
                            else:
                                antecedent_string = f"{antecedent_string} the cousine is {antecedents["food"]}"
                    
                if consequent == "seats":
                    text = f"{text} \n\nThe restaurant is assigned {consequent} because {antecedent_string}.\n"
                else:
                    text = f"{text} \n\nThe restaurant is {consequent} because {antecedent_string}.\n"


        if len(options) != 5:
            text += '\nDo you have any other preferences or this suggestion satisfies you and want to hear more details?'
        return text

    @staticmethod
    def __state_6(options: Tuple) -> str:
        name = options[0]
        food = options[1]
        phone = options[2]
        price_range = options[3]
        zipcode = options[4]

        text = ''

        if food or price_range or options[5]:
            text += name.title() + ' is '

            if food and price_range:
                text += food + ', ' + price_range + ' restaurant'
            elif food:
                text += food + ' restaurant'
            else:
                text += price_range + ' restaurant'

            if options[5]:
                area = options[5][0]
                address = options[5][1]
                text += ' located in the ' + area + 'part of the town at ' + address

            text += '. '

        if zipcode:
            text += 'The zipcode is ' + zipcode + '.'

        if phone:
            text += 'To contact them you can call ' + phone + '.'

        return text or 'Sorry I do not understand your request.'

    @staticmethod
    def __state_7(options: Tuple) -> None:
        pass

    @staticmethod
    def __state_8(options: Tuple) -> str:
        text = """Do you have additional requirements?"""
        return text

    @staticmethod
    def __exit(_: Tuple) -> str:
        return 'Thank you for using Cambridge Restaurant System. Bye!'
