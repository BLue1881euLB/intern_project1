import pandas as pd
import numpy as np
import psycopg2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
class fct_sale_predict:
    def __init__(self):
        self.initial()

    # 训练
    def train(self, data,today):
        self.today = pd.date_range(today,today)[0]
        self.trans_train_input(data)
        self.get_feature()
        #self.feature_matrix_GBR()
        #self.feature_metrix_LR()
        self.GridSearch()
        self.Weight_Reg()
        self.output()

    # 预测
    def predict(self):
        self.future = pd.DataFrame(index=self.date_ahead.index)
        self.date_ahead.groupby("product_code").apply(self.predict_method_hq)
        self.date_ahead_low.groupby("product_code").apply(self.predict_method_lq)

    # 输出测试结果
    def output(self):

        self.pre = {}
        self.pre_err = pd.DataFrame(index=self.date_ahead.index)
        self.Reg = {}
        self.LR = {}
        self.w_gbr = {}
        self.feature.groupby("product_code").apply(self.test_method_hq)
        self.date_ahead_low.groupby("product_code").apply(self.test_method_lq)

    # 初始化
    def initial(self):
        self.source_engine = 'postgresql+psycopg2://user:password@host:port/dbname'
        self.engine = create_engine(self.source_engine)
        self.conn = psycopg2.connect('dbname=dbname user=user password=password host=host port=port')
        self.date_ahead = pd.read_sql("select * from rst.dim_product_salted_period", con=self.conn)
        self.date_ahead.set_index('product_code', inplace=True)
        self.date_ahead.rename(columns={'period_amount': 'delay'}, inplace=True)
        self.date_ahead['delay'] = self.date_ahead['delay'] + 1

    # 转化train输入
    def trans_train_input(self, data):
        self.day_volume = data.rename(columns={'qty_sum': 'day_volume',"date_part":"week"}).set_index('sale_date')
        # 把历史特征表读进来
        self.feature = pd.read_sql('''select * from rst.fct_salted_sale_feat2 where sale_date >= '%s'::date -365
                                        ''' % (self.today.strftime('%Y%m%d')),
                                   con=self.conn)
        if self.feature.shape[0] > 0:
            self.feature.set_index("sale_date",inplace=True)
            self.feature.index = pd.to_datetime(self.feature.index)
            self.feature.sort_index(inplace=True)

        self.day_volume.index = pd.to_datetime(self.day_volume.index)
        self.date_max = data['sale_date'].values.max().strftime('%Y/%m/%d')

        self.date_holiday = pd.read_sql(
            "select date,holiday from rst.dim_holiday where date >='%s'::date"%(data["sale_date"].values.min().strftime('%Y/%m/%d')), con=self.conn).set_index('date')
        self.date_holiday.index = pd.to_datetime(self.date_holiday.index)
        self.day_volume['month'] = list(map(lambda x: int(x.strftime('%m')), self.day_volume.index))
        self.day_volume['year'] = list(map(lambda x: int(x.strftime('%Y')), self.day_volume.index))
        self.day_volume['week_num'] = list(map(lambda x: int(x.strftime('%W')), self.day_volume.index))
        self.day_volume['holiday'] = list(
            map(lambda x: int(self.date_holiday.loc[x, 'holiday']), self.day_volume.index))
        self.day_volume['week_ocp'] = np.array(list(map(lambda x: round(self.day_volume.iloc[x]['day_volume'] /
                                                                        sum(self.day_volume[((self.day_volume[
                                                                                                  'product_code'] ==
                                                                                              self.day_volume.iloc[x][
                                                                                                  'product_code'])
                                                                                             & (self.day_volume[
                                                                                                    'week_num'] ==
                                                                                                self.day_volume.iloc[x][
                                                                                                    'week_num']))][
                                                                                'day_volume']), 2),
                                                        range(self.day_volume.shape[0]))))
        self.drop_abnormal()

    #去除outliers
    def drop_outliers_method(self,df):
        Mean = df["day_volume"].mean()
        Std = df["day_volume"].std()
        return df[((df['day_volume'] <= 2 * Std + Mean) & (df["day_volume"] >= Mean -2 * Std)) & (df["holiday"] != 1)]

    # 去除异常值和假日
    def drop_abnormal(self):
        self.df_drop = self.day_volume.groupby(["product_code","year","month"]).apply(self.drop_outliers_method)
        self.df_drop.drop(["year","month","product_code"],axis=1,inplace=True)
        self.df_drop.reset_index(inplace=True)
        self.df_drop.drop_duplicates(["sale_date","product_code"],inplace=True)
        self.df_drop.set_index("sale_date",inplace=True)
        self.low_volume_code = ['05060024', '05324003', '05020018', '05010003']
        self.date_ahead_low = self.date_ahead[self.date_ahead.index.isin(self.low_volume_code)]
        self.date_ahead.drop(self.low_volume_code, inplace=True)
        self.df_low = self.df_drop[self.df_drop['product_code'].isin(self.low_volume_code)]
        self.df_drop = self.df_drop[~self.df_drop['product_code'].isin(self.low_volume_code)]
        self.divide = pd.DataFrame()
        self.df_drop.groupby("product_code").apply(self.count_divide)
        self.df_drop.loc[:,'day_attr'] = list(map(self.day_attr, range(self.df_drop.shape[0])))
        self.df_drop.drop(["year","month","week","week_num","holiday","week_ocp"],axis=1,inplace = True)

    #特征
    def get_feature(self):
        self.feature1 = {}
        self.feature2 = {}
        self.train_col = ['Mov_average_20',
                          '1_delta', '2_delta', '3_delta', '4_delta', '5_delta', '6_delta', '7_delta',
                          'day_attr_0', 'day_attr_1', 'day_cur_0', 'day_cur_1']
        self.train_col1 = ['Mov_average_20', 'day_attr_0', "day_attr_1"]

        self.feature_append = self.df_drop.groupby("product_code").apply(self.feature_method)
        if self.feature_append.shape[0] > 0:
            self.feature_append.drop("product_code", inplace=True, axis=1)
            self.feature_append.reset_index(inplace=True)
            self.feature_append.drop_duplicates(["product_code", "sale_date"], inplace=True)
            exist = "append" if self.feature.shape[0] > 0 else "replace"
            self.conn.close()
            print(self.feature_append.shape)
            self.feature_append.to_sql('fct_salted_sale_feat2', schema='rst', con=self.engine, if_exists=exist,index=False)
            self.conn = psycopg2.connect('dbname=dbname user=user password=password host=host port=port')
            self.feature_append.set_index("sale_date", inplace=True)
            self.feature = pd.concat([self.feature, self.feature_append], axis=0).sort_index()

        pass

    #自动调参
    def GridSearch(self):
        self.gs_code = []
        self.clf = {}

        self.param = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                      'n_estimators': [40, 50, 80, 100,200,300],
                      'max_depth': [2, 3]
                      }

        self.feature.groupby("product_code").apply(self.gridsearch_method)

    # 统计1234/567和12345/67区分度
    def count_divide(self,df):
        code = df.iloc[0]["product_code"]
        if df.shape[0] > 0:
            data_1234 = df[(df['week'] > 0) & (df['week'] < 5)].sort_index()
            data_567 = df[(df['week'] > 4)|(df['week'] == 0)].sort_index()
            self.divide.loc[code, '1234-567'] = np.mean(data_567['week_ocp']) - np.mean(data_1234['week_ocp'])
            data_12345 = df[(df['week'] > 0) & (df['week'] < 6)].sort_index()
            data_67 = df[(df['week'] > 5) | (df['week'] == 0)].sort_index()
            self.divide.loc[code, '12345-67'] = np.mean(data_67['week_ocp']) - np.mean(data_12345['week_ocp'])

    # 日属性
    def day_attr(self, x):
        code = self.df_drop.iloc[x]['product_code']
        if self.divide.loc[code, '1234-567'] > self.divide.loc[code, '12345-67']:
            return 0 if self.df_drop.iloc[x]['week'] in list(range(1, 5)) else 1
        else:
            return 0 if self.df_drop.iloc[x]['week'] in list(range(1, 6)) else 1

    #构造feature方法
    def feature_method(self,df):
        code = df.iloc[0]['product_code']
        delay = int(self.date_ahead.loc[code, 'delay'])
        m = 20
        df.sort_index(inplace=True)
        data = df.sort_index()
        if self.feature.shape[0] > 0:
            date_min = self.feature[self.feature["product_code"] == code].index.max()
            date_min = pd.date_range(date_min,date_min)[0] + 1
        else:
            date_min = df.index.min()
        date_max = list(data.index)[-1]
        date_max = pd.date_range(date_max, date_max)[0]
        #print(date_max)
        date_range = pd.date_range(date_min, date_max)
        date_pre = self.today + delay - 1
        self.feature1[code] = pd.DataFrame()
        date_range1 = list(date_range)
        date_range1.append(date_pre)
        #print(date_range1)
        for i, date in enumerate(date_range1):
            df.loc[date, 'Mov_average_20'] = round(
                np.mean(data.loc[(date - m - delay + 1):(date - delay), 'day_volume']), 2)
            for j in range(7):
                a = list(data.loc[date - delay - j - 1:date - delay - j - 1, 'day_volume'].values)
                a.append(0)
                b = list(data.loc[date - delay - j:date - delay - j, 'day_volume'].values)
                b.append(0)

                if a[0] != 0:
                    df.loc[date, str(j + 1) + str('_delta')] = round((a[0] - b[0]) / a[0],
                                                                                         2) * 100
                elif b[0] != 0:
                    df.loc[date, str(j + 1) + str('_delta')] = round((a[0] - b[0]) / b[0],
                                                                                         2) * 100
                else:
                    df.loc[date, str(j + 1) + str('_delta')] = 0

            if date in data.index:
                if len(list(data.loc[(date - delay):(date - delay), 'day_volume'])) > 0:
                    df.loc[date, 'True_volume'] = (data.loc[date, 'day_volume'] - data.loc[
                        date - delay, 'day_volume']) / data.loc[date - delay, 'day_volume'] * 100
                else:
                    df.loc[date, 'True_volume'] = (data.loc[date, 'day_volume'] -
                                                                       df.loc[
                                                                           date, 'Mov_average_20']) / \
                                                                      df.loc[
                                                                          date, 'Mov_average_20'] * 10
                df.loc[date, 'day_attr_0'] = 0 if data.loc[date, 'day_attr'] else 1
                df.loc[date, 'day_attr_1'] = 1 if data.loc[date, 'day_attr'] else 0

                day_cur = data.loc[(date - delay):(date - delay), 'day_attr'].values[0] if len(
                    list(data.loc[(date - delay):(date - delay), 'day_attr'])) > 0 else 0
                df.loc[date, 'day_cur_0'] = 0 if day_cur else 1
                df.loc[date, 'day_cur_1'] = 1 if day_cur else 0

            elif date == date_pre:
                if self.divide.loc[code, '12345-67'] < self.divide.loc[code, '1234-567']:
                    df.loc[date,'day_attr'] = 0 if int(date.strftime('%w')) in list(
                            range(1, 5)) else 1
                    df.loc[date, 'day_attr_1'] = 0 if int(date.strftime('%w')) in list(
                            range(1, 5)) else 1
                    df.loc[date, 'day_attr_0'] = 1 if int(date.strftime('%w')) in list(
                            range(1, 5)) else 0
                else:
                    df.loc[date, 'day_attr'] = 0 if int(date.strftime('%w')) in list(
                            range(1, 6)) else 1
                    df.loc[date, 'day_attr_1'] = 0 if int(date.strftime('%w')) in list(
                            range(1, 6)) else 1
                    df.loc[date, 'day_attr_0'] = 1 if int(date.strftime('%w')) in list(
                            range(1, 6)) else 0

                df.loc[date,'True_volume'] = 0
                df.loc[date,"product_code"] = code
                df.loc[date,"day_volume"] = 0
                df.loc[date, 'day_cur_0'] = 0 if data.loc[date_max,'day_attr'] else 1
                df.loc[date, 'day_cur_1'] = 1 if data.loc[date_max,'day_attr'] else 0


        #print((df.sort_index()).tail())
        df.dropna(inplace=True)
        self.feature1[code] = df.tail(1)[self.train_col]
        self.feature2[code] = df.tail(1)[self.train_col1]
        #print(self.feature1[code])
        #print(self.feature2[code])
        df.drop(date, inplace=True)
        df.sort_index(inplace=True)
        return df
        pass

    #参数搜索方法
    def gridsearch_method(self,df):
        code = df.iloc[0]["product_code"]
        if code not in self.gs_code:
            self.gs_code.append(code)
            df.sort_index(inplace=True)
            from sklearn.model_selection import GridSearchCV
            from sklearn.model_selection import PredefinedSplit
            from sklearn.ensemble import GradientBoostingRegressor
            train_feature = df.head(df.shape[0] - 18).iloc[:][self.train_col]
            train_real = df.head(df.shape[0] - 18).iloc[:]["True_volume"]
            val_split = np.zeros(train_feature.shape[0])
            val_split[:(train_feature.shape[0] - 18)] = -1
            ps = PredefinedSplit(test_fold=val_split)
            GBR = GradientBoostingRegressor(random_state=0)
            self.clf[code] = GridSearchCV(GBR, self.param, scoring='neg_mean_absolute_error', cv=ps)
            self.clf[code].fit(train_feature, train_real)
            print(code, self.clf[code].best_params_)
        pass

    # 通过预测的差值计算预测真值
    def delta_real(self, delta_Series, code):
        import pandas as pd
        real_Series = pd.Series(index=delta_Series.index)
        data = self.df_drop[self.df_drop['product_code'] == code].sort_index()
        delay = self.date_ahead.loc[code, 'delay']
        feature = self.feature[self.feature["product_code"]==code].sort_index()
        for date in delta_Series.index:
            date = pd.date_range(date, date)[0]
            if len(data.loc[date - delay:date - delay, 'day_volume']) > 0:
                real_Series.loc[date] = round((delta_Series.loc[date] * data.loc[date - delay, 'day_volume'] / 100 +
                                               data.loc[date - delay, 'day_volume']), 2)
            else:
                real_Series.loc[date] = round(
                    (delta_Series.loc[date] * feature.loc[date, 'Mov_average_20'] / 100 +
                     feature.loc[date, 'Mov_average_20']), 2)

        return real_Series

        # 返回相对误差绝对值的平均值

    #计算mape
    def Error(self, true_list, pre_list):
        return np.mean(abs(true_list - pre_list) / true_list * 100)

    #计算LR和GBR的权重
    def Weight_Reg(self):
        self.valid_err = pd.DataFrame(index=self.date_ahead.index)
        self.valid = {}
        self.feature.groupby("product_code").apply(self.weight_cal)

    #计算权重的方法
    def weight_cal(self,df):
        code = df.iloc[0]["product_code"]

        train_feature = df.head(df.shape[0] - 18).iloc[:][self.train_col]
        train_real = df.head(df.shape[0] - 18).iloc[:]["True_volume"]
        train_feature1 = df.head(df.shape[0] - 18).iloc[:][self.train_col1]
        train_real1 = df.head(df.shape[0] - 18).iloc[:]["day_volume"]

        GBR = GradientBoostingRegressor(loss=self.clf[code].best_params_['loss'],
                                        n_estimators=self.clf[code].best_params_['n_estimators'],
                                        max_depth=self.clf[code].best_params_['max_depth'],
                                        random_state=0).fit(train_feature.head(train_feature.shape[0] - 18),
                                                            train_real.head(train_feature.shape[0] - 18).values)

        LR = LinearRegression().fit(train_feature1.head(train_feature1.shape[0] - 18),
                                    train_real1.head(train_feature1.shape[0] - 18).values)

        self.valid[code] = pd.DataFrame(index=train_feature.tail(18).index)

        self.valid[code]['predict_GBR'] = self.delta_real(
            pd.Series(GBR.predict(train_feature.tail(18)), index=train_feature.tail(18).index), code)

        self.valid[code]['predict_LR'] = pd.Series(LR.predict(train_feature1.tail(18)),
                                                   index=train_feature1.tail(18).index)
        self.valid[code]['real'] = pd.Series(train_real1.tail(18).values,index = train_feature1.tail(18).index)

        self.valid_err.loc[code, 'mape_GBR'] = round(self.Error(self.valid[code]['real'], self.valid[code]['predict_GBR']), 2)
        self.valid_err.loc[code, 'mae_GBR'] = round(np.mean(abs(self.valid[code]['real'] - self.valid[code]['predict_GBR'])), 2)
        self.valid_err.loc[code, 'mape_LR'] = round(self.Error(self.valid[code]['real'], self.valid[code]['predict_LR']), 2)
        self.valid_err.loc[code, 'mae_LR'] = round(np.mean(abs(self.valid[code]['real'] - self.valid[code]['predict_LR'])), 2)

        pass

    #高销预测输出
    def predict_method_hq(self,df):
        code = df.index[0]
        pre_val1 = self.Reg[code].predict(self.feature1[code])[0]
        pre_val2 = self.LR[code].predict(self.feature2[code])[0]
        date = list(self.feature1[code].index)[0]
        date = pd.date_range(date, date)[0]
        delay = self.date_ahead.loc[code, 'delay'] - 1
        self.future.loc[code, 'product_code'] = code
        self.future.loc[code, 'product_name'] = self.date_ahead.loc[code, 'product_name']
        self.future.loc[code, 'creation_date'] = self.today
        self.future.loc[code, 'predict_date'] = self.today + delay
        data = self.df_drop[self.df_drop['product_code'] == code].sort_index()
        if self.date_holiday.loc[self.today + delay, 'holiday'] != '1':

            self.future.loc[code, 'predict_qty_sum'] = round(((pre_val1 * data.iloc[-1]['day_volume'] / 100 +
                                                               data.iloc[-1]['day_volume']) * self.w_gbr[code] + (
                                                                      1 - self.w_gbr[code]) * pre_val2), 2)
        else:
            print(self.date_ahead.loc[code, 'product_name'] + "节假日predict罢工")
        real = pd.read_sql(
            "select * from rst.fct_salted_sale_feat where sale_date = '%s'::date and product_code ='%s'"
            % (date.strftime('%Y%m%d'), code), con=self.conn)
        if real.shape[0] > 0:
            self.future.loc[code, 'real_qty_sum'] = real.iloc[0]['qty_sum']

        pass

    #高销测试输出
    def test_method_hq(self,df):
        df.sort_index(inplace=True)
        code = df.iloc[0]["product_code"]

        self.w_gbr[code] = (1 / (float(self.valid_err.loc[code, 'mape_GBR']))) / (
                (1 / float(self.valid_err.loc[code, 'mape_GBR']))
                + (1 / float(self.valid_err.loc[code, 'mape_LR'])))


        train_feature = df.head(df.shape[0] - 18).iloc[:][self.train_col]
        train_real = df.head(df.shape[0] - 18).iloc[:]["True_volume"]
        test_feature = df.tail(18).iloc[:][self.train_col]

        train_feature1 = df.head(df.shape[0] - 18).iloc[:][self.train_col1]
        train_real1 = df.head(df.shape[0] - 18).iloc[:]["day_volume"]
        test_feature1 = df.tail(18).iloc[:][self.train_col1]
        test_real1 = df.tail(18).iloc[:]['day_volume']

        self.Reg[code] = GradientBoostingRegressor(loss=self.clf[code].best_params_['loss'],
                                                   n_estimators=self.clf[code].best_params_['n_estimators'],
                                                   max_depth=self.clf[code].best_params_['max_depth'],
                                                   random_state=0).fit(train_feature, train_real.values)

        self.LR[code] = LinearRegression().fit(train_feature1, train_real1.values)
        self.pre[code] = pd.DataFrame(index=df.tail(18).index)
        self.pre[code]['real'] = test_real1.values
        self.pre[code]['predict'] = (
                self.delta_real(pd.Series(self.Reg[code].predict(test_feature), index=test_feature.index),
                                code) * self.w_gbr[code] +
                pd.Series(self.LR[code].predict(test_feature1), index=df.tail(18).index) * (
                        1 - self.w_gbr[code]))

        self.pre_err.loc[code, 'mape'] = str(
            round(self.Error(self.pre[code]['real'], self.pre[code]['predict']), 2)) + '%'
        self.pre_err.loc[code, 'mae'] = round(np.mean(abs(self.pre[code]['real'] - self.pre[code]['predict'])), 2)

        self.Reg[code] = GradientBoostingRegressor(loss=self.clf[code].best_params_['loss'],
                                                   n_estimators=self.clf[code].best_params_['n_estimators'],
                                                   max_depth=self.clf[code].best_params_['max_depth'],
                                                   random_state=0).fit(df.iloc[:][self.train_col],
                                                                       df.iloc[:]['True_volume'].values)

        self.LR[code] = LinearRegression().fit(df.iloc[:][self.train_col1],
                                               df.iloc[:]['day_volume'].values)
        pass

    #低销预测输出
    def predict_method_lq(self,df):
        code = df.index[0]
        if code != '05020018':
            data = self.df_low[self.df_low['product_code'] == code].sort_index()
            delay = self.date_ahead_low.loc[code, 'delay'] - 1
            date = self.today
            self.future.loc[code, 'product_code'] = code
            self.future.loc[code, 'product_name'] = self.date_ahead_low.loc[code, 'product_name']
            self.future.loc[code, 'creation_date'] = date
            self.future.loc[code, 'predict_date'] = date + delay
            if self.date_holiday.loc[self.today + delay, 'holiday'] != '1':
                self.future.loc[code, 'predict_qty_sum'] = round(
                    np.mean(data.loc[date - delay - 9:date - delay, 'day_volume']), 2)
            else:
                print(self.date_ahead_low.loc[code, 'product_name'] + "节假日predict罢工")
            real = pd.read_sql(
                "select * from rst.fct_salted_sale_feat where sale_date = '%s'::date and product_code ='%s'"
                % ((date + delay).strftime('%Y%m%d'), code), con=self.conn)
            if real.shape[0] > 0:
                self.future.loc[code, 'real_qty_sum'] = real.iloc[0]['qty_sum']
        pass

    # 低销测试输出
    def test_method_lq(self, df):
        code = df.index[0]
        if code != '05020018':
            data = self.df_low[self.df_low['product_code'] == code].sort_index()
            self.pre[code] = pd.DataFrame(index=data.tail(18).index)
            self.pre[code]['real'] = data.tail(18).loc[:, 'day_volume']
            self.pre[code]['predict'] = pd.Series(index=data.tail(18).index)
            for i, date in enumerate(data.tail(18).index):
                date = pd.date_range(date, date)[0]
                self.pre[code]['predict'].loc[date] = round(np.mean(data.loc[date - 19:date]['day_volume'].values),
                                                            2)
            self.pre_err.loc[code, 'mape'] = str(
                round(self.Error(self.pre[code]['real'], self.pre[code]['predict']), 2)) + '%'
            self.pre_err.loc[code, 'mae'] = round(np.mean(abs(self.pre[code]['real'] - self.pre[code]['predict'])),
                                                  2)
        pass
