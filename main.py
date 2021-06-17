from config import args
from etl import ETL, creates_products, save_df_to_pickle, load_pickle
import models

# df = ETL(args)
df = load_pickle('df')
# products = creates_products(df)
# save_df(df, 'df')
# models.analyze_products_poly(products,3)

models.orders_by_day(df)
orders = models.analyze_order_one_day(df)




##


# c.analyze_products_poly()
# c.cat_cumulatives()
# c.sales_hist()
# a = c.df
# asoc = c.analyze_assoc()
# a = c.df
# from models import analyze_assoc
#
# analyze_assoc(c.df)
