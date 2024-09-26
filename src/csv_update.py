import pandas as pd
import numpy as np

def copy_csv(filename):
    df = pd.read_csv(filename+ '.csv')
    df.to_csv(filename + "_copy_.csv")


# copy_csv('C:/Users/Galya/Documents/School/MAIR/restaurant-recommendations-dialog-system/data/restaurant_info')

df = pd.read_csv('C:/Users/Galya/Documents/School/MAIR/restaurant-recommendations-dialog-system/data/restaurant_info_copy_.csv')
df['food_quality'] = np.random.choice(['good food', 'fast food', 'decent'], size=len(df))
df['crowdedness'] = np.random.choice(['busy', 'quiet'], size=len(df))
df['length_of_stay'] = np.random.choice(['long stay', 'short stay', 'medium stay'], size=len(df))
df.to_csv("C:/Users/Galya/Documents/School/MAIR/restaurant-recommendations-dialog-system/data/restaurant_info_copy_.csv",index=False)