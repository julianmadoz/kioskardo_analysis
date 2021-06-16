from config import args
from etl import ETL, creates_products

df = ETL(args)
products = creates_products(df)







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
