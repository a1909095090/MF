import pandas as pd
from MF import MF
import  torch
epochs = 1
df  = pd.read_csv('ratings.csv')
convert_dict = {'userId': int,
                'movieId': int,
                'rating': float}
df = df.astype(convert_dict)

def data_split(data):
    train_data, vali_data, test_data = [], [], []
    user_sum = len(df['userId'].unique())
    item_sum = 0
    d = {}
    ans = [[0,0,0] for i in range(user_sum)]

    for i in range(len(data)):
        user=df.loc[i,"userId"]-1
        item=df.loc[i,"movieId"]-1
        rating=df.loc[i,"rating"]
        if (df.loc[i,"movieId"]-1) in d.keys() :
            item = d[df.loc[i,"movieId"]-1]
        else:
            d[df.loc[i,"movieId"]-1]=item_sum
            item=item_sum
            item_sum =item_sum+1
        data=[user,item,rating]
        if ans[user][0] > ans [user][1] * 3 :

            vali_data.append(data)
            ans[user][1]=ans[user][1]+1
        elif ans[user][0]>ans[user][2] * 3  :
            test_data.append(data)
            ans[user][2]=ans[user][2]+1
        else:
            train_data.append(data)
            # print(df.iloc[i,:3].to_numpy())
            ans[user][0] = ans[user][0] + 1
    return train_data,vali_data,test_data,user_sum,item_sum

# print(len(df))

# for
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_data, vali_data, test_data, user_nums, item_nums=  data_split(df)

    model = MF(user_nums=user_nums,item_nums=item_nums)

    for epoch in range(epochs):
        print(f"epoch :{epoch}")
        model.fit(train_data)
        model.test(vali_data)

    model.test(test_data)
