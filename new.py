import psycopg2
from dateutil.parser import parse
from time import time
from sqlalchemy import create_engine
import pandas as pd
import fct_salted_sale_predict

def update_salted_sale_predict_(today):
    start_time = time()
    # 读取数据
    conn = psycopg2.connect('dbname=dbname user=user password=password host=host port=port')
    origin_data = pd.read_sql('''select * from rst.fct_salted_sale_feat
                                 where sale_date >= '%s'::date -365
                                 and sale_date < '%s'::date''' % (today.strftime('%Y%m%d'), today.strftime('%Y%m%d')),
                              con=conn)

    # 训练
    salted_model = fct_salted_sale_predict.fct_sale_predict()
    salted_model.train(origin_data,today)
    # 预测
    salted_model.predict()
    print("预测结果：")
    print(salted_model.future)
    # 输出相应指标
    print('测试集指标：')
    print(salted_model.pre_err)
    salted_predict_result_df = salted_model.future

    
    # 更新数据库
    conn = psycopg2.connect('dbname=dbname user=user password=password host=host port=port')
    predict_data_need_to_fill = pd.read_sql("select * from rst.fct_salted_sale_predict where predict_date='%s'::date-1"%today.strftime("%Y%m%d"),con=conn)
    feat_data_real = pd.read_sql("select product_code,qty_sum as real_qty_sum ,sale_date as predict_date from rst.fct_salted_sale_feat where sale_date='%s'::date-1"%(today.strftime('%Y%m%d')),con=conn)

    cur = conn.cursor()
    cur.execute(
        '''delete from rst.fct_salted_sale_predict where creation_date = '%s'::date''' % (today.strftime('%Y%m%d')))

    cur.execute(
        "delete from rst.fct_salted_sale_predict where predict_date = '%s'::date-1"%(today.strftime('%Y%m%d'))
    )
    conn.commit()
    conn.close()
    predict_data_filled = pd.merge(predict_data_need_to_fill.drop('real_qty_sum',axis=1),feat_data_real,how="left")

    source_engine = 'postgresql+psycopg2://user:password@host:port/dbname'
    engine = create_engine(source_engine)
    pd.concat([salted_predict_result_df,predict_data_filled],axis=0).to_sql('fct_salted_sale_predict', schema='rst', con=engine, if_exists='append', index=False)
    salted_model.feature.to_csv("feature.csv")
    print("总用时：%d 秒"%(int(time()-start_time)))


today = parse('20180725').date()
update_salted_sale_predict_(today)


