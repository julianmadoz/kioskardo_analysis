from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

args={}
args['file'] = '/home/julianmadoz/Nextcloud2/kioskardo_analysis/orders.csv'
args['months_ago'] = 4
args['weeks_prediction'] = 2
args['top_items'] = 2
args['plots'] = 2
args['date_start'] = datetime.now() + relativedelta(months=-args['months_ago'])
args['date_finish'] = datetime.now() + relativedelta(weeks=args['weeks_prediction'])
args['date_range'] = pd.date_range(args['date_start'],args['date_finish'], freq = 'd').to_series().apply(lambda x: x.timestamp()).reset_index().drop(columns=['index'])[0].to_numpy()
