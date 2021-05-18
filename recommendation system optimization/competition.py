#Method description
#For this project, I mainly use model based method to do this project. The model I choose is xgbregressor with learning rate = 0.23, since at this learning rate, the rmse is lowest.(It's most balanced since very high rmse may increase rmse and error rate and low rmse will incur more computation time) Another important thing I do to reduce rmse is to increase features to be trained. Especially for business.json and user.json. there are lots of features to learn, including all numerical features and also non-numerical feature. like for true or false value, we can convert 'true' into 1 and 'false' into 0. so this was transferred into numerical values and can be added into xgbregressor. Moreover, for numerical features value that vary for every different business_id, adding such features into model can significantly decrease rmse.  Also, since adding more features need more memory, so I expand executor memory and driver memory into 16g. Finally, I found that rmse in model based is significantly lower than rmse in user-based and item-based. So Mainly I use model based to do this project.



import json
import time
from itertools import combinations
from pyspark import SparkConf, SparkContext
from operator import add
import csv
import json
import os
import sys
import xgboost as xgb
import numpy as np
import math


start_time = time.time()
folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

#folder_path = '/Users/ruichao/Desktop/hw1/all_data'
#test_file_name = '/Users/ruichao/Desktop/hw1/all_data/yelp_val.csv'
#output_file_name = '/Users/ruichao/Desktop/result.csv'
conf = SparkConf().setAppName("553hw3").setMaster('local[*]').set('spark.executor.memory','10g').set('spark.driver.memory','10g')
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

train_path = folder_path + '/yelp_train.csv'
train = sc.textFile(train_path)
header = train.first()
train = train.filter(lambda i: i != header).map(lambda i: i.split(","))

test = sc.textFile(test_file_name)
test_rdd = sc.textFile(test_file_name)
test_header = test_rdd.first()
test_jh = test_rdd.filter(lambda x: x != test_header)
header_test = test.first()
test = test.filter(lambda i: i != header_test).map(lambda i: i.split(","))
test_use = test.sortBy(lambda i:((i[0]),(i[1]))).persist()
#test_jh = test.filter(lambda i: i != header_test)


def user_dict_info(data):
    a = data.map(lambda x: ((x[0]), ((x[1]), float(x[2])))).groupByKey().sortByKey()
    b = a.mapValues(dict).collectAsMap()
    return b
def business_dict_info(data):
    a = data.map(lambda x: ((x[1]), ((x[0]), float(x[2])))).groupByKey().sortByKey()
    b = a.mapValues(dict).collectAsMap()
    return b

user_rating_info = user_dict_info(train)
business_rating_info = business_dict_info(train)
user_rating_avg_info = train.map(lambda i: (i[0],(float(i[2])))).combineByKey(lambda x:(x,1),lambda y,x:(y[0]+x,y[1]+1),lambda y,z:(y[0]+z[0],y[1]+z[1]))
user_rating_avg_info = user_rating_avg_info.mapValues(lambda i:i[0]/i[1])
user_rating_avg_info = {i:j for i, j in user_rating_avg_info.collect()}


business_rating_avg_info = train.map(lambda i: (i[1],(float(i[2])))).combineByKey(lambda x:(x,1),lambda y,x:(y[0]+x,y[1]+1),lambda y,z:(y[0]+z[0],y[1]+z[1]))
business_rating_avg_info = business_rating_avg_info.mapValues(lambda i:i[0]/i[1])
business_rating_avg_info = {i:j for i, j in business_rating_avg_info.collect()}

def pearson_calculation(rating_list_business,rating_list_neighbor,numerator,deno_business,deno_neighbour,avg_neighbor,avg_business):
    for i in range(len(rating_list_business)):
        numerator += (rating_list_business[i]-avg_business)*(rating_list_neighbor[i]-avg_neighbor)
        deno_business += (rating_list_business[i]-avg_business)**2
        deno_neighbour += (rating_list_neighbor[i]-avg_neighbor)**2

    return numerator, deno_business, deno_neighbour
def pearson(neighbour,user,business,business_avg):
    business_rating_lst = []
    neighbour_rating_lst = []
    business_rating_neighbour = business_rating_info.get(neighbour)
    avg_business_rating_neighbour = business_rating_avg_info.get(neighbour)
    for id in user:
        if business_rating_neighbour.get(id) == True:
            business_score = business.get(id)
            neighbour_score = business_rating_neighbour.get(id)
            business_rating_lst.append(business_score)
            neighbour_rating_lst.append(neighbour_score)

    if len(business_rating_lst)>0:
        numerator_default, business_denominator_default,neighbour_denominator_default = 0,0,0
        numerator_use, business_denominator, neighbour_denominator = pearson_calculation(business_rating_lst,neighbour_rating_lst,numerator_default,business_denominator_default,neighbour_denominator_default,avg_business_rating_neighbour,business_avg)
        denominator_overall = (business_denominator*neighbour_denominator)**0.5
        if numerator_use != 0 and denominator_overall !=0:
            pearson_coef = numerator_use/denominator_overall
        elif numerator_use == 0 and denominator_overall !=0:
            pearson_coef = 0
        elif numerator_use == 0 and denominator_overall ==0:
            pearson_coef = 1
        else:
            pearson_coef = -1
    else:
        pearson_coef = float(business_avg/avg_business_rating_neighbour)

    return pearson_coef


def prediction_coef_calculation(pearson_coef_lst):
    numerator = 0
    denominator = 0
    pearson_coef_lst = sorted(pearson_coef_lst,key=lambda x:-x[0])
    for i in range(len(pearson_coef_lst)):
        numerator += pearson_coef_lst[i][0]*pearson_coef_lst[i][1]
        denominator += abs(pearson_coef_lst[i][0])
    pred_value = numerator/denominator
    return pred_value


def final_calculation(rdd):
    user_info, business_info = rdd[0], rdd[1]
    if business_info in business_rating_info:
        avg_business_rating = business_rating_avg_info.get(business_info)
        user = list(business_rating_info.get(business_info))
        user_single = business_rating_info.get(business_info)
        if user_rating_info.get(user_info) is not None:
            user_rating_info_lst = list(user_rating_info.get(user_info))
            if len(user_rating_info_lst)>0:
                pearson_lst = []
                for i in user_rating_info_lst:
                    cur_neighbor_score = business_rating_info.get(i).get(user_info)
                    pearson_coef = pearson(i,user,user_single,avg_business_rating)
                    if pearson_coef > 0:
                        if pearson_coef>1:
                           pearson_coef = 1/pearson_coef
                        pearson_lst.append((pearson_coef,cur_neighbor_score))
                #ck(pearson_lst,user_info,user_rating_info_lst,business_rating_info,user,user_single,avg_business_rating)
                pred_coef = prediction_coef_calculation(pearson_lst)
                return pred_coef
            else:
                return avg_business_rating
        else:
            return avg_business_rating
    else:
        return str(user_rating_avg_info.get(user_info))

final_result = test_use.map(final_calculation).collect()
cf_coef = np.asarray(final_result,dtype='float')
user_path = folder_path + '/user.json'
user = sc.textFile(user_path)

business_path = folder_path + '/business.json'
business = sc.textFile(business_path)

tip_path = folder_path + '/tip.json'
tip = sc.textFile(tip_path)

review_train_path = folder_path + '/review_train.json'
review_train = sc.textFile(review_train_path)

def get_price_range(attributes, key):
    if attributes:
        if key in attributes.keys():
            return int(attributes.get(key))
    return 0


def category_count(category):
    if category is None:
        return 0
    else:
        new = category.split(',')
        count = len(new)
        return count

def time_total(attributes,key1):
    if attributes:
        time_ = 0
        if key1 in attributes.keys():
            mon = attributes.get(key1)
            time_56 = int(mon[5:7])
            time_01 = int(mon[0:2])
            time_ += time_56-time_01
        return time_
    return 0
def judge_tf(attributes,key):
    if attributes:
        if key in attributes.keys():
            if attributes.get(key) == 'True':
                return 1
            else:
                return 0
    return 0
def business_parking(mother,mother_key):
    if mother:
        if mother_key in mother.keys():
            tt = mother.get(mother_key)
            count = tt.count('True')
            return count
        return 0
    return 0




user_json = user.map(json.loads).map(
    lambda i: ((i["user_id"], (i["useful"],i['compliment_hot'],i['fans'], i["review_count"], i["average_stars"],i['compliment_funny'],i['compliment_more'],i['compliment_cool'],i['compliment_profile'],
                               i['compliment_note'],i['compliment_cute'],i['compliment_list'],i['compliment_plain'],i['compliment_writer'],i['compliment_photos'])))).collectAsMap()
business_json = business.map(json.loads).map(lambda i: ((i["business_id"], (i["review_count"], i["stars"],i['is_open'],
                                            i['latitude'],i['longitude'],get_price_range(i['attributes'],'RestaurantsPriceRange2')
                                            ,judge_tf(i['attributes'],'BusinessAcceptsCreditCards'),judge_tf(i['attributes'],'BikeParking'),judge_tf(i['attributes'],'OutdoorSeating')                                    ,judge_tf(i['attributes'],'RestaurantsGoodForGroups'),judge_tf(i['attributes'],'RestaurantsDelivery'),judge_tf(i['attributes'],'Caters'),judge_tf(i['attributes'],'HasTV'),judge_tf(i['attributes'],'RestaurantsReservations'),judge_tf(i['attributes'],'RestaurantsTableService'),judge_tf(i['attributes'],'OutdoorSeating'),judge_tf(i['attributes'],'ByAppointmentOnly'),judge_tf(i['attributes'],'RestaurantsTakeOut'),judge_tf(i['attributes'],'AcceptsInsurance'),judge_tf(i['attributes'],'WheelchairAccessible'),judge_tf(i['attributes'],'GoodForKids'))))).collectAsMap()
tip_json = tip.map(json.loads).map(lambda i:((i["user_id"], (i["likes"])))).collectAsMap()
review_train_json = review_train.map(json.loads).map(lambda i: ((i["user_id"], (i["stars"], i["useful"],i['funny'],i['cool'])))).collectAsMap()



def preprocess_data(data,user_json,business_json,review_train_json,default):
    user_id = data[0]
    business_id = data[1]
    if default == False:
        rating = data[2]
    else:
        rating = -1
    if user_id in user_json.keys() and business_id in business_json.keys() and user_id in review_train_json.keys():
        useful, compliment_hot,fans, reviewcount_user, averagestar_user,funny_user,more_user,cool_user,profile_user,note_user,cute_user,list_user,plain_user,writer_user,photo_user = user_json[user_id]
        reviewcount_business, averagestar_business,open_business,lat,lon,price_range,accept_card,parking,outdoor,aa,bb,cc,dd,ee,ff,gg,hh,ii,jj,kk,ll = business_json[business_id]
        #likes = tip_json[user_id]
        star_r,useful_r,funny_r,cool_r = review_train_json[user_id]


        useful, compliment_hot,fans,reviewcount_user, averagestar_user,funny_user,more_user,cool_user,profile_user,note_user,cute_user,list_user,plain_user,writer_user,photo_user,reviewcount_business, averagestar_business,open_business,lat,lon,price_range,accept_card,parking,outdoor,aa,bb,cc,dd,ee,ff,gg,hh,ii,jj,kk,ll,star_r,useful_r,funny_r,cool_r,rating = float(useful),float(compliment_hot),float(fans), float(reviewcount_user),float(averagestar_user),float(funny_user),float(more_user),float(cool_user),float(profile_user),float(note_user),float(cute_user),float(list_user),float(plain_user),float(writer_user),float(photo_user),float(reviewcount_business),float(averagestar_business),float(open_business),float(lat),float(lon),float(price_range),float(accept_card),float(parking),float(outdoor),float(aa),float(bb),float(cc),float(dd),float(ee),float(ff),float(gg),float(hh),float(ii),float(jj),float(kk),float(ll),float(star_r),float(useful_r),float(funny_r),float(cool_r),float(rating)


        return [user_id,business_id,useful,compliment_hot,fans, reviewcount_user, averagestar_user,funny_user,more_user,cool_user,profile_user,note_user,cute_user,list_user,plain_user,writer_user,photo_user,reviewcount_business, averagestar_business,open_business,lat,lon,price_range,accept_card,parking,outdoor,aa,bb,cc,dd,ee,ff,gg,hh,ii,jj,kk,ll,star_r,useful_r,funny_r,cool_r,rating]
    else:
        return [user_id,business_id,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]



def create_train_or_test(data,tf):
    train_xgb = data.map(lambda i: preprocess_data(i, user_json,business_json,review_train_json,tf)).collect()
    train_xgb_array = np.array(train_xgb)
    len_cell = len(train_xgb_array[0])
    x = np.array(train_xgb_array[:,2:len_cell-1],dtype = 'float')
    y = np.array(train_xgb_array[:,-1],dtype = 'float')
    return x,y
def create_test_dataset(data,tf):
    test_xgb = data.map(lambda i: preprocess_data(i, user_json,business_json,review_train_json,tf)).collect()
    test_xgb_array = np.array(test_xgb)
    return test_xgb_array
def get_feature_list(row):
    features = []
    features.extend(user_json.get(row[0]))
    features.extend(business_json.get(row[1]))
    return features



x_train, y_train = create_train_or_test(train,False)
x_test, y_test = create_train_or_test(test,True)


xgb_used = xgb.XGBRegressor(learning_rate = 0.23)
xgb_used.fit(x_train,y_train)

prediction_coef = xgb_used.predict(x_test)
test_dataset = create_test_dataset(test,True)

combined_coef = 0.9999*prediction_coef + (1-0.9999)*cf_coef
results = np.c_[test_dataset[:,:2],combined_coef]

def csv_writing(path,input):
    file = open(path,mode='w')
    ww = csv.writer(file,delimiter=',',quoting=csv.QUOTE_MINIMAL)
    ww.writerow(['user_id','business_id','prediction'])
    for i in input:
        ww.writerow([str(i[0]), str(i[1]), float(i[2])])
    file.close()

csv_writing(output_file_name,results)
end_time = time.time()-start_time

val = sc.textFile(test_file_name)
hv = val.first()
valid = val.filter(lambda i: i!=hv).map(lambda i:i.split(",")).sortBy(lambda i:((i[0],i[1]))).persist()
val_rdd = val.filter(lambda i: i!=hv)
ordd = sc.textFile(output_file_name)
oh = ordd.first()
od = ordd.filter(lambda x: x != oh).map(lambda x: x.split(','))
odd = od.map(lambda i: (((i[0]), (i[1])), float(i[2])))
tdd = val_rdd.map(lambda i: i.split(",")).map(lambda i: (((i[0]), (i[1])), float(i[2])))
together = tdd.join(odd).map(lambda i: (abs(i[1][0] - i[1][1])))


rmsee = together.map(lambda x:x**2).reduce(lambda x,y:x+y)
rmse = math.sqrt(rmsee/odd.count())




print('Error Distributions: ' + '\n')
d_01 = together.filter(lambda i: i >= 0 and i < 1).count()
d_12 = together.filter(lambda i: i >= 1 and i < 2).count()
d_23 = together.filter(lambda i: i >= 2 and i < 3).count()
d_34 = together.filter(lambda i: i >= 3 and i < 4).count()
d_l4 = together.filter(lambda i: i >= 4).count()
print(">=0 and <1: ", d_01)
print(">=1 and <2: ", d_12)
print(">=2 and <3: ", d_23)
print(">=3 and <4: ", d_34)
print(">=4: ", d_l4)





print("RMSE: " + str(rmse))



print("Execution Time: "+str(end_time))