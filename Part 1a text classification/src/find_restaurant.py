import pandas as pd
#I'm not sure if it's good and if it hold in all the edge cases
def find_restaurant(data, preferences):
    food_pref = preferences.get('food', [])
    price_pref = preferences.get('price range', [])
    area_pref = preferences.get('area', [])

    if food_pref:
        data = data[data['food_type'].isin(food_pref)]
    if price_pref:
        data = data[data['price_range'].isin(price_pref)]
    if area_pref:
        data = data[data['area'].isin(area_pref)]
    
    if not data.empty:
        return data.head(1)
    else:
        return pd.DataFrame()