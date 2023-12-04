import pandas as pd

data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

def p1():

    print('*Problem1\nPrint the top 10 players in hits, batting average, homerun, and on base percentage for each year from 2015 to 2018.')
    year_list = ['2015', '2016', '2017', '2018']
    col_list = ['H', 'avg', 'HR', 'OBP']

    for year in year_list:
        year_df = data_df.loc[data_df['year'] == int(year)]
        for col in col_list:
            year_df = year_df.sort_values(by=col, ascending=False)
            top_ten = year_df[['batter_name', col]][:10]   
            print(f'in year {year}\n')
            for i in range(top_ten.shape[0]):
                name='batter_name'
                print(f'top {i+1} {col} player is: {top_ten[name].iloc[i]} (value : {top_ten[col].iloc[i]})')
            print("----------------------------------")

    return 0

def p2():

    print("\n*Print the player with the highest war by position in 2018")
    p_list = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
    year_df = data_df.loc[data_df['year'] == 2018]
    year_df = year_df.sort_values(by='war', ascending=False)
    for position in p_list:
        position_df = year_df.loc[year_df['cp']==position]
        player = position_df['batter_name'].iloc[0]
        print(f'in position : {position}, the highest war is : {player}')
    return 0


def p3():

    print("\n*Among R, H, HR, RBI, SB, war, avg, OBP, and SLG, which has the highest correlation with salary?")
    my_list = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']
    best_co_value = 0.0
    highest = ''
    for item in my_list:
        co_value=data_df[item].corr(data_df['salary'])
        if co_value > best_co_value:
            best_co_value = co_value
            highest = item
    print(f'{highest} has the highest correlation with salary.')

    return 0

p1()
p2()
p3()