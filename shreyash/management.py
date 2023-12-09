from django.shortcuts import render
from django.contrib.auth.decorators import login_required,user_passes_test
from ..models import *
from accounts.models import Management, CustomUser
from .new_corp import *

from ..form import SelectInsuranceCompany,SelectCorporate,SelectBranch,SelectHeadOffice,SelectNumberofDays
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pandas.tseries.offsets import MonthEnd
import statsmodels.api as sm
import plotly.offline as pyo
from plotly.offline import plot
from plotly.offline import iplot
import plotly.figure_factory as ff
import babel
from babel.numbers import format_currency
import textwrap
from textwrap import wrap
import re
from datetime import timedelta
from sqlalchemy import create_engine
from django_plotly_dash import DjangoDash
import dash
from dash import dash_table
from pptx import Presentation
import tempfile
# Comma Seperation to values
import locale





from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import joblib

# Create your views here.
from django.http import HttpResponse
from django.contrib import messages
from django.views import View
from django.db.models import Q


import time
import timeit
import numpy as np
from datetime import datetime, timedelta,date,time,timezone
import plotly.express as px
import statsmodels.api as sm
import pyodbc
from django.contrib.auth.views import PasswordResetView
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm

from django.db.models import Subquery, OuterRef
import os
Driver = 'SQL Server'
Server = r'MDINETPROJECTS\Analytics'
Database = 'Enrollment'
Claims_Database = 'Claims_SLA'
UID = 'mdianalytics'
PWD= 'mdianalytics@123'
Database_Connection = f'mssql://{UID}:{PWD}@{Server}/{Database}?driver={Driver}'
Claims_DB_Connection = f'mssql://{UID}:{PWD}@{Server}/{Claims_Database}?driver={Driver}'



connection = pyodbc.connect(driver='{SQL Server}', host=Server, database=Database,
                     user=UID, password=PWD)

claims_connection = pyodbc.connect(driver='{SQL Server}', host=Server, database=Claims_Database,
                     user=UID, password=PWD)


config = {'responsive': True, 'displaylogo': False}



def is_Management(self):
    if str(self.user_type) == 'Management':
        return True
    else:
        return False
rec_login_required = user_passes_test(lambda u: True if u.is_Management else False, login_url='/')

def Management_login_required(view_func):
    decorated_view_func = login_required(rec_login_required(view_func), login_url='/')

    return decorated_view_func

# @recruiter_login_required
# def index(request):
#     return render(request, 'index.html')
#
#
# @Management_login_required
# class DashboardView(View):
#     def get_prediction(self,request):
#         daily_outcount = DailyOsCount.objects.all().values()
#         daily_outcountdf = pd.DataFrame(daily_outcount)
#         daily_outcountdf['date_1'] = pd.to_datetime(daily_outcountdf['date_1'])
#         daily_outcountdf['os_count'] = daily_outcountdf['os_count'].astype('int')
#
#         # dropping duplicate dates
#         daily_outcountdf.sort_values('date_1', inplace=True)
#         all_dup_dates = daily_outcountdf['date_1'].value_counts()[daily_outcountdf['date_1'].value_counts() > 1].index
#         # daily_outcountdf['os_count'] = daily_outcountdf['os_count'].astype('int')
#         for date in all_dup_dates:
#             max_os = daily_outcountdf[daily_outcountdf['date_1'] == date]['os_count'].max()
#             idx = daily_outcountdf[
#                 (daily_outcountdf['date_1'] == date) & (daily_outcountdf['os_count'] != max_os)].index
#             daily_outcountdf.drop(idx, inplace=True)
#
#         daily_outcountdf.drop_duplicates('date_1', inplace=True)
#
#         # here when as freq is used then missing dates are added
#         daily_outcountdf = daily_outcountdf.set_index('date_1').asfreq('D')
#
#         if daily_outcountdf.isna().sum()['os_count'] != 0:
#             daily_outcountdf["os_count"] = daily_outcountdf["os_count"].interpolate(method='linear')
#
#         daily_outcountdf['os_count'] = daily_outcountdf['os_count'].astype(int)
#         dailyout_data = daily_outcountdf['os_count']
#
#         # start_date =  date.today()
#         # end_date = start_date + timedelta(days=7)
#
#         order = (0, 1, 0)
#         seasonal_order = (1, 1, 1, 7)
#         mod = sm.tsa.statespace.SARIMAX(dailyout_data,
#                                         order=order,
#                                         seasonal_order=seasonal_order,
#                                         enforce_stationarity=False,
#                                         enforce_invertibility=False)
#         result = mod.fit()
#
#         # prediction
#         # no of predicted days
#
#         form = SelectNumberofDays(request.POST or None)
#         if request.method == 'POST':
#             d1 = request.POST.get('No_days')
#         # prediction
#         # no of predicted days
#         # d1 = 14
#         # no of actual days shown in plot
#             d2 = 7
#
#             pred = result.get_prediction(start=len(dailyout_data), end=len(dailyout_data) + d1 - 1, dynamic=False)
#             prediction = (pred.predicted_mean).astype(int)
#
#             # adding last day of actual value to prediction to get contionuos line in plot
#             last_row = daily_outcountdf[['os_count']].iloc[-1:]
#             last_row.columns = ['predicted_mean']
#             prediction_df = pd.concat([last_row, pd.DataFrame(prediction)])
#             # prediction_df = pd.concat([daily_outcountdf[['os_count']][-1:],prediction_df],axis=1)
#
#             # connecting both df to plot
#
#             df_fin = pd.concat([daily_outcountdf[['os_count']][-d2:], prediction_df], axis=1)
#             df_fin.columns = ['Actual', 'Predicted']
#
#             # plot
#             line = px.line(df_fin)
#             pred_plot = plot(line, output_type='div')
#         return render(request, 'webapp/leader/query.html', pred_plot)
#
#     def get_data(self):
#         all_out = Allicdailyoutstanding.objects.all().values()
#         branch = BranchPortal.objects.all().values()
#         all_outdf = pd.DataFrame(all_out)
#         branch_master = pd.DataFrame(branch)
#         return all_outdf,branch_master
#
#     def lastdocument_recived(self,all_outdf):
#         all_outdf['last_document_received'] = pd.to_datetime(all_outdf['last_document_received'], format='%y-%m-%d')
#         Last_Document_Received_Date = all_outdf['last_document_received']
#         return Last_Document_Received_Date
#
#     def today_tat_ldr(self,all_outdf,Last_Document_Received_Date):
#
#         TODAY_TAT_LDR = []
#         for d in Last_Document_Received_Date:
#             TODAY_TAT_LDR.append(today.date() - d.date())
#         all_outdf['TODAY_TAT_LDR'] = TODAY_TAT_LDR
#         all_outdf['TODAY_TAT_LDR'] = all_outdf['TODAY_TAT_LDR'].astype('timedelta64[D]').astype(int)
#
#         # LDR_BAND
#         conditions = [
#             (all_outdf['TODAY_TAT_LDR'] <= 0),
#             (all_outdf['TODAY_TAT_LDR'] > 0) & (all_outdf['TODAY_TAT_LDR'] <= 5),
#             (all_outdf['TODAY_TAT_LDR'] > 5) & (all_outdf['TODAY_TAT_LDR'] <= 10),
#             (all_outdf['TODAY_TAT_LDR'] > 10)
#         ]
#         # create a list of the values we want to assign for each condition
#         values = ['10 Days & ABOVE', '00-05 Days', '06-10 Days', '10 Days & ABOVE']
#
#         # create a new column and use np.select to assign values to it using our lists as arguments
#         all_outdf['LDR_BAND'] = np.select(conditions, values)
#
#         # TAT_LDR
#         TAT_LDR = []
#         for b in Last_Document_Received_Date:
#             TAT_LDR.append(EndOfMonth1.date() - b.date() - timedelta(days=1))
#         all_outdf['TAT_LDR'] = TAT_LDR
#         all_outdf['TAT_LDR'] = all_outdf['TAT_LDR'].astype('timedelta64[D]').astype(int)
#
#         # BAND_LDR_TAT
#
#         all_outdf.loc[(all_outdf['TAT_LDR'] > 30), 'BAND_LDR_TAT'] = 'Above 1 month'
#         all_outdf.loc[(all_outdf['TAT_LDR'] <= 30), 'BAND_LDR_TAT'] = 'Below 1 month'
#
#         # TAT_DOA
#         DOA = pd.to_datetime(all_outdf['doa'])
#         TAT_DOA = []
#         for a in DOA:
#             TAT_DOA.append(EndOfMonth1 - a.replace(tzinfo=None) - timedelta(days=1))
#         all_outdf['TAT_DOA'] = TAT_DOA
#         all_outdf['TAT_DOA'] = all_outdf['TAT_DOA'].astype('timedelta64[D]').astype('Int64')
#
#         # BAND_DOA_TAT
#         all_outdf.loc[(all_outdf['TAT_DOA'] > 30), 'BAND_DOA_TAT'] = 'Above 1 month'
#         all_outdf.loc[(all_outdf['BAND_DOA_TAT'] != 'Above 1 month'), 'BAND_DOA_TAT'] = 'Below 1 month'
#         BAND_DOA_TAT = all_outdf['BAND_DOA_TAT']
#
#         # TAT_FIRST_INTIMATION
#         all_outdf['first_intimation_date'] = pd.to_datetime(all_outdf['first_intimation_date'].dt.date)
#         First_Intimation_Date = pd.to_datetime(all_outdf['first_intimation_date'], format='%y-%m-%d')
#
#         TAT_FIRST_INTIMATION = []
#         for i in First_Intimation_Date:
#             TAT_FIRST_INTIMATION.append(EndOfMonth1 - i - timedelta(days=1))
#         all_outdf['TAT_FIRST_INTIMATION'] = TAT_FIRST_INTIMATION
#         all_outdf['TAT_FIRST_INTIMATION'] = all_outdf['TAT_FIRST_INTIMATION'].astype('timedelta64[D]').astype('Int64')
#
#         # YEAR and MONTH_NO
#
#         all_outdf['year_1'] = pd.DatetimeIndex(all_outdf['doa']).year
#         all_outdf['month_no'] = pd.DatetimeIndex(all_outdf['doa']).month
#
#         # BAND_FIRST_INTIMATION
#         all_outdf.loc[(all_outdf['TAT_FIRST_INTIMATION'] > 90), 'BAND_FIRST_INTIMATION'] = 'Above 3 months'
#         all_outdf.loc[(all_outdf[
#                            'BAND_FIRST_INTIMATION'] != 'Above 3 months'), 'BAND_FIRST_INTIMATION'] = 'Below 3 months'
#         BAND_FIRST_INTIMATION = all_outdf['BAND_FIRST_INTIMATION']
#
#         # DOA_< 1 MONTH_TILL_CURRENT_MONTH_END
#         result = all_outdf.assign(count=(all_outdf['BAND_DOA_TAT'] == 'Below 1 month')).groupby(
#             'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
#         result.rename(columns={'count': 'DOA_<_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
#         # doaless1month = result['DOA_<_1_MONTH_TILL_CURRENT_MONTH_END']
#
#         # DOA_> 1 MONTH_TILL_CURRENT_MONTH_END
#
#         result1 = all_outdf.assign(count=(all_outdf['BAND_DOA_TAT'] == 'Above 1 month')).groupby(
#             'servicing_branch', sort=False, as_index=False).agg({'count': sum})
#         result1.rename(columns={'count': 'DOA_>_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
#         doagreat1month = result1['DOA_>_1_MONTH_TILL_CURRENT_MONTH_END']
#
#         # LDR_<_1 MONTH_TILL_CURRENT_MONTH_END
#
#         result2 = all_outdf.assign(
#             count=(all_outdf['BAND_LDR_TAT'] == 'Below 1 month') & (all_outdf['revised_servicing_branch']).isin(
#                 branch_master['branch_name'])).groupby('revised_servicing_branch', sort=False,
#                                                        as_index=False).agg({'count': sum})
#         result2.rename(columns={'count': 'LDR_<_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
#
#         # LDR_>_1 MONTH_TILL_CURRENT_MONTH_END
#
#         result3 = all_outdf.assign(count=(all_outdf['BAND_LDR_TAT'] == 'Above 1 month')).groupby(
#             'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
#         result3.rename(columns={'count': 'LDR_>_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
#         ldrgreat1month = result3['LDR_>_1_MONTH_TILL_CURRENT_MONTH_END']
#
#         # FI_DATE_>3MONTH_TILL_CURRENT_MONTH_END
#
#         result4 = all_outdf.assign(count=(all_outdf['BAND_FIRST_INTIMATION'] == 'Above 3 months')).groupby(
#             'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
#         result4.rename(columns={'count': 'FI_DATE_>3MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
#         fidategreaterthreemonth = result4['FI_DATE_>3MONTH_TILL_CURRENT_MONTH_END']
#
#         # FI_DATE_<3MONTH_TILL_CURRENT_MONTH_END
#
#         result5 = all_outdf.assign(count=(all_outdf['BAND_FIRST_INTIMATION'] == 'Below 3 months')).groupby(
#             'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
#         result5.rename(columns={'count': 'FI_DATE_<3MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
#         fidatelessthreemonth = result5['FI_DATE_<3MONTH_TILL_CURRENT_MONTH_END']
#
#         # OS CLAIMS
#
#         result6 = pd.DataFrame(all_outdf.assign(
#             count=(all_outdf['revised_servicing_branch']).isin(all_outdf['revised_servicing_branch'])).groupby(
#             'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum}))
#         result6.rename(columns={'count': 'OS_CLAIMS'}, inplace=True)
#
#         # % OS CLAIMS
#
#         result6['per_OS_claims'] = ((result6['OS_CLAIMS'] / sum(result6['OS_CLAIMS'])) * 100).round(2)
#
#         # % OS CLAIMS > 1_MONTH_DOA
#
#         result1['per_OS_claims > 1_MONTH_DOA'] = (
#                 (result1['DOA_>_1_MONTH_TILL_CURRENT_MONTH_END'] / result6['OS_CLAIMS']) * 100).round(2)
#         osonemonthdoa = result1['per_OS_claims > 1_MONTH_DOA']
#         osonemonthdoaact = osonemonthdoa.to_string(index=False)
#         # % OS CLAIMS > 1_MONTH_LDR
#
#         result3['per_OS_claims > 1_MONTH_LDR'] = (
#                 (result3['LDR_>_1_MONTH_TILL_CURRENT_MONTH_END'] / result6['OS_CLAIMS']) * 100).round(2)
#
#         per_os_one_month_ldr = result3['per_OS_claims > 1_MONTH_LDR']
#         per_os_one_month_ldract = per_os_one_month_ldr.to_string(index=False)
#
#         # % OS CLAIMS > 3_MONTHS_FIRST_INTIMATION
#
#         result5['per_OS_claims > 3_MONTHS_FIRST_INTIMATION'] = (
#                 (fidategreaterthreemonth / result6['OS_CLAIMS']) * 100).round(2)
#         os_three_month_fi = result5['per_OS_claims > 3_MONTHS_FIRST_INTIMATION']
#         os_three_month_fiact = os_three_month_fi.to_string(index=False)
#
#         # Merging
#
#         BRANCH = pd.merge(result6, branch_master, left_on='revised_servicing_branch',
#                           right_on='branch_name', how='inner')
#
#         # Concating Results
#
#         BRANCH_UPDATED = pd.concat([BRANCH, result, result1, result2, result3, result4, result5], axis=1)
#
#         # Removing Duplicates
#
#         BRANCH_UPDATED = BRANCH_UPDATED.T.drop_duplicates().T
#
#         BRANCH1 = all_outdf.merge(branch_master[['branch_group_type', 'branch_name']], how='left',
#                                   left_on='servicing_branch', right_on='branch_name')
#
#         BRANCH_FINAL = pd.merge(BRANCH1, BRANCH_UPDATED, left_on='revised_servicing_branch',
#                                 right_on='revised_servicing_branch', how='inner')
#
#         BRANCH_UPDATED.rename(columns={'revised_servicing_branch': 'REVISED_SERVICING_BRANCH'}, inplace=True)
#
#         BRANCH_MAIN = BRANCH_UPDATED[(BRANCH_UPDATED['branch_group_type'] == '02 MAIN')]
#
#         BRANCH_HO = BRANCH_UPDATED[(BRANCH_UPDATED['branch_group_type'] == '01 MDI HO')]
#
#         BRANCH_OTHER = BRANCH_UPDATED[(BRANCH_UPDATED['branch_group_type'] == '03 OTHER')]
#
#         BRANCH_MAIN['RANK_MAIN_DOA'] = BRANCH_MAIN['per_OS_claims > 1_MONTH_DOA'].rank(method='min')
#
#         BRANCH_HO['RANK_HO_DOA'] = BRANCH_HO['per_OS_claims > 1_MONTH_DOA'].rank(method='min')
#
#         BRANCH_OTHER['RANK_OTHER_DOA'] = BRANCH_OTHER['per_OS_claims > 1_MONTH_DOA'].rank(method='min')
#
#         BRANCH_MAIN['RANK_MAIN_LDR'] = BRANCH_MAIN['per_OS_claims > 1_MONTH_LDR'].rank(method='min')
#
#         BRANCH_HO['RANK_HO_LDR'] = BRANCH_HO['per_OS_claims > 1_MONTH_LDR'].rank(method='min')
#
#         BRANCH_OTHER['RANK_OTHER_LDR'] = BRANCH_OTHER['per_OS_claims > 1_MONTH_LDR'].rank(method='min')
#
#         BRANCH_MAIN['RANK_MAIN_FI'] = BRANCH_MAIN['per_OS_claims > 3_MONTHS_FIRST_INTIMATION'].rank(method='min')
#
#         BRANCH_HO['RANK_HO_FI'] = BRANCH_HO['per_OS_claims > 3_MONTHS_FIRST_INTIMATION'].rank(method='min')
#
#         BRANCH_OTHER['RANK_OTHER_FI'] = BRANCH_OTHER['per_OS_claims > 3_MONTHS_FIRST_INTIMATION'].rank(
#             method='min')
#
#         # ####################################################RANK_HO_DOA#################################################
#
#         RANK_HO_DOA1 = BRANCH_HO[['REVISED_SERVICING_BRANCH', 'RANK_HO_DOA']].sort_values(by='RANK_HO_DOA',
#                                                                                           ascending=True).reset_index(
#             drop=True)
#         RANK_HO_DOA1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_HO_DOA': 'HO_DOA'}, inplace=True)
#
#         colorscale = [[0, '#272D31'], [.5, '#ffffff'], [1, '#ffffff']]
#         font = ['#FCFCFC', '#006400', '#52FF00', '#FFD900', '#FF7700', '#FF1500']
#
#         RANK_HO_DOA2 = ff.create_table(np.vstack([RANK_HO_DOA1.columns, RANK_HO_DOA1.values]), font_colors=font,
#                                        colorscale=colorscale)
#         RANK_HO_DOA2.update_layout(title_text='RANK ON > THAN 1 MONTH LAT')
#         RANK_HO_DOA2.layout.width = 250
#         RANK_HO_DOA = plot(RANK_HO_DOA2, output_type='div')
#         # RANK_HO_DOA = plot(RANK_HO_DOA2, output_type='div', config = {'staticPlot': True})
#
#         ##############################################RANK_MAIN_DOA1#######################################################
#         RANK_MAIN_DOA1 = BRANCH_MAIN[['REVISED_SERVICING_BRANCH', 'RANK_MAIN_DOA']].sort_values(
#             by='RANK_MAIN_DOA', ascending=True).reset_index(drop=True)
#         RANK_MAIN_DOA1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_MAIN_DOA': 'MAIN_DOA'}, inplace=True)
#
#         main_doa_font = ['#FCFCFC', '#00EE00', '#008B00', '#FFC500', '#FFB200', '#FF8C00', '#FF3030', '#FF5000',
#                          '#FF3C00', '#FF2900', '#FF1500']
#
#         RANK_MAIN_DOA2 = ff.create_table(np.vstack([RANK_MAIN_DOA1.columns, RANK_MAIN_DOA1.values]),
#                                          font_colors=main_doa_font,
#                                          colorscale=colorscale)
#         RANK_MAIN_DOA2.layout.width = 250
#         RANK_MAIN_DOA = plot(RANK_MAIN_DOA2, output_type='div')
#
#         ##############################################RANK_OTHER_DOA#######################################################
#
#         RANK_OTHER_DOA1 = BRANCH_OTHER[['REVISED_SERVICING_BRANCH', 'RANK_OTHER_DOA']].sort_values(
#             by='RANK_OTHER_DOA', ascending=True).reset_index(drop=True)
#         RANK_OTHER_DOA1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_OTHER_DOA': 'OTHER_DOA'},
#                                inplace=True)
#
#         other_doa_font = ['#FCFCFC', '#00EE00', '#008B00', '#FFC500', '#FFB200', '#FF8C00', '#FF3030', '#FF5000',
#                           '#FF3C00', '#FF2900', '#FF1500']
#
#         RANK_OTHER_DOA2 = ff.create_table(np.vstack([RANK_OTHER_DOA1.columns, RANK_OTHER_DOA1.values]))
#         RANK_OTHER_DOA2.layout.width = 250
#         RANK_OTHER_DOA = plot(RANK_OTHER_DOA2, output_type='div')
#
#         ############################################## RANK_MAIN_LDR ######################################################
#
#         RANK_MAIN_LDR1 = BRANCH_MAIN[['REVISED_SERVICING_BRANCH', 'RANK_MAIN_LDR']].sort_values(
#             by='RANK_MAIN_LDR', ascending=True).reset_index(drop=True)
#         RANK_MAIN_LDR1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_MAIN_LDR': 'MAIN_LDR'}, inplace=True)
#
#         RANK_MAIN_LDR2 = ff.create_table(np.vstack([RANK_MAIN_LDR1.columns, RANK_MAIN_LDR1.values]))
#         RANK_MAIN_LDR2.layout.width = 250
#         RANK_MAIN_LDR = plot(RANK_MAIN_LDR2, output_type='div')
#
#         ###### RANK_HO_LDR #####
#
#         RANK_HO_LDR1 = BRANCH_HO[['REVISED_SERVICING_BRANCH', 'RANK_HO_LDR']].sort_values(by='RANK_HO_LDR',
#                                                                                           ascending=True).reset_index(
#             drop=True)
#
#         RANK_HO_LDR1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_HO_LDR': 'HO_LDR'}, inplace=True)
#
#         RANK_HO_LDR2 = ff.create_table(np.vstack([RANK_HO_LDR1.columns, RANK_HO_LDR1.values]))
#         RANK_HO_LDR2.update_layout(title_text='RANK ON > 1 MONTH DOA')
#         RANK_HO_LDR2.layout.width = 250
#         RANK_HO_LDR = plot(RANK_HO_LDR2, output_type='div')
#
#         ###### RANK_OTHER_LDR ######
#
#         RANK_OTHER_LDR1 = BRANCH_OTHER[['REVISED_SERVICING_BRANCH', 'RANK_OTHER_LDR']].sort_values(
#             by='RANK_OTHER_LDR', ascending=True).reset_index(drop=True)
#         RANK_OTHER_LDR1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_OTHER_LDR': 'OTHER_LDR'},
#                                inplace=True)
#
#         RANK_OTHER_LDR2 = ff.create_table(np.vstack([RANK_OTHER_LDR1.columns, RANK_OTHER_LDR1.values]))
#         RANK_OTHER_LDR2.layout.width = 250
#         RANK_OTHER_LDR = plot(RANK_OTHER_LDR2, output_type='div')
#
#         ###### RANK_MAIN_FI ######
#
#         RANK_MAIN_FI1 = BRANCH_MAIN[['REVISED_SERVICING_BRANCH', 'RANK_MAIN_FI']].sort_values(by='RANK_MAIN_FI',
#                                                                                               ascending=True).reset_index(
#             drop=True)
#         RANK_MAIN_FI1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_MAIN_FI': 'MAIN_FI'}, inplace=True)
#
#         RANK_MAIN_FI2 = ff.create_table(np.vstack([RANK_MAIN_FI1.columns, RANK_MAIN_FI1.values]))
#         RANK_MAIN_FI2.layout.width = 250
#         RANK_MAIN_FI = plot(RANK_MAIN_FI2, output_type='div')
#
#         ###### RANK_HO_FI ######
#
#         RANK_HO_FI1 = BRANCH_HO[['REVISED_SERVICING_BRANCH', 'RANK_HO_FI']].sort_values(by='RANK_HO_FI',
#                                                                                         ascending=True).reset_index(
#             drop=True)
#         RANK_HO_FI1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_HO_FI': 'HO_FI'}, inplace=True)
#
#         RANK_HO_FI2 = ff.create_table(np.vstack([RANK_HO_FI1.columns, RANK_HO_FI1.values]))
#         RANK_HO_FI2.update_layout(title='RANK ON > 3 MONTH FIRST INTIMATION')
#         RANK_HO_FI2.layout.width = 250
#         RANK_HO_FI = plot(RANK_HO_FI2, output_type='div')
#
#         ###### RANK_OTHER_FI ######
#
#         RANK_OTHER_FI1 = BRANCH_OTHER[['REVISED_SERVICING_BRANCH', 'RANK_OTHER_FI']].sort_values(
#             by='RANK_OTHER_FI', ascending=True).reset_index(drop=True)
#         RANK_OTHER_FI1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_OTHER_FI': 'OTHER_FI'}, inplace=True)
#
#         RANK_OTHER_FI2 = ff.create_table(np.vstack([RANK_OTHER_FI1.columns, RANK_OTHER_FI1.values]))
#         RANK_OTHER_FI2.layout.width = 250
#         RANK_OTHER_FI = plot(RANK_OTHER_FI2, output_type='div')
#
#     mycontext = {
#         'pred_plot': pred_plot,
#         'out_count': out_count,
#         'RANK_HO_DOA': RANK_HO_DOA,
#         'RANK_MAIN_DOA': RANK_MAIN_DOA,
#         'RANK_OTHER_DOA': RANK_OTHER_DOA,
#         'RANK_HO_LDR': RANK_HO_LDR,
#         'RANK_MAIN_LDR': RANK_MAIN_LDR,
#         'RANK_OTHER_LDR': RANK_OTHER_LDR,
#         'RANK_HO_FI': RANK_HO_FI,
#         'RANK_MAIN_FI': RANK_MAIN_FI,
#         'RANK_OTHER_FI': RANK_OTHER_FI,
#     }
#
#




@Management_login_required
def dashboard(request):
    if request.user.is_authenticated:
        daily_outcount = DailyOsCount.objects.all().values()
        daily_outcountdf = pd.DataFrame(daily_outcount)
        daily_outcountdf['date_1'] = pd.to_datetime(daily_outcountdf['date_1'])
        # daily_outcountdf['os_count'] = daily_outcountdf['os_count'].astype('int')



        # dropping duplicate dates
        daily_outcountdf.sort_values('date_1',inplace=True)
        all_dup_dates = daily_outcountdf['date_1'].value_counts()[daily_outcountdf['date_1'].value_counts() > 1].index
        # daily_outcountdf['os_count'] = daily_outcountdf['os_count'].astype('int')
        for date in all_dup_dates:
            max_os = daily_outcountdf[daily_outcountdf['date_1'] == date]['os_count'].max()
            idx = daily_outcountdf[(daily_outcountdf['date_1'] == date) & (daily_outcountdf['os_count'] != max_os)].index
            daily_outcountdf.drop(idx, inplace=True)

        daily_outcountdf.drop_duplicates('date_1', inplace=True)

        # here when as freq is used then missing dates are added
        daily_outcountdf = daily_outcountdf.set_index('date_1').asfreq('D')


        if daily_outcountdf.isna().sum()['os_count'] != 0:
            daily_outcountdf["os_count"] = daily_outcountdf["os_count"].interpolate(method='linear')

        daily_outcountdf['os_count'] = daily_outcountdf['os_count'].astype(int)
        dailyout_data = daily_outcountdf['os_count']


        #start_date =  date.today()
        #end_date = start_date + timedelta(days=7)

        order = (0,1,0)
        seasonal_order = (1,1,1,7)
        mod = sm.tsa.statespace.SARIMAX(dailyout_data,
                                                 order=order,
                                                 seasonal_order=seasonal_order,
                                                 enforce_stationarity=False,
                                                 enforce_invertibility=False)
        result = mod.fit()

        # prediction
        # no of predicted days
        d1 = 14
        # no of actual days shown in plot
        d2 = 7

        pred = result.get_prediction(start=len(dailyout_data),end=len(dailyout_data)+d1-1,dynamic=False)
        prediction = (pred.predicted_mean).astype(int)

        # adding last day of actual value to prediction to get contionuos line in plot
        last_row = daily_outcountdf[['os_count']].iloc[-1:]
        last_row.columns = ['predicted_mean']
        prediction_df = pd.concat([last_row, pd.DataFrame(prediction)])
        # prediction_df = pd.concat([daily_outcountdf[['os_count']][-1:],prediction_df],axis=1)

        # connecting both df to plot

        df_fin = pd.concat([daily_outcountdf[['os_count']][-d2:],prediction_df],axis=1)
        df_fin.columns = ['Actual', 'Predicted']

        # plot
        line = px.line(df_fin)
        line.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        pred_plot = plot(line, output_type='div')

        # all_out_queryset = Allicdailyoutstanding.objects.all()
        # filtered_queryset = all_out_queryset.exclude(
        #     Q(revised_servicing_branch__exact='') | Q(revised_servicing_branch__isnull=True) |
        #     Q(servicing_branch__exact='') | Q(servicing_branch__isnull=True)
        # )
        all_out = Allicdailyoutstanding.objects.values('last_document_received','doa','first_intimation_date','revised_servicing_branch','servicing_branch')
        daily_settled = AllicSettledCasesOfYesterday.objects.values('corp_retail','lodgetype','utr_updatedate','last_document_received','lodgedate','br_cd_dor','ho_cd_dor','inwdt','gmc_category','actuallosstype','prs_date')


        out_count = all_out.count()
        settled_count = daily_settled.count()
        print(settled_count)
        all_outdf = pd.DataFrame(all_out)

        # Servicing Branch Cleaning
        all_outdf['revised_servicing_branch'].replace('MDINDIA HEALTH INSURANCE TPA PVT. LTD-INDORE', 'INDORE', inplace=True)
        all_outdf = all_outdf[all_outdf['revised_servicing_branch'].str.strip() != '']
        all_outdf = all_outdf[all_outdf['servicing_branch'].str.strip() != '']
        all_outdf['servicing_branch'] = all_outdf['servicing_branch'].str.upper()



        branch = BranchPortal.objects.all().values()
        branch_master = pd.DataFrame(branch)
        today = datetime.now()
        print(today)
        EndOfMonth1 = today + MonthEnd(1)

        # Todays_TAT_LDR
        all_outdf['last_document_received'] = pd.to_datetime(all_outdf['last_document_received'], format='%y-%m-%d')
        Last_Document_Received_Date = all_outdf['last_document_received']


        TODAY_TAT_LDR =[]
        for d in Last_Document_Received_Date:
            TODAY_TAT_LDR.append(today.date() - d.date())
        all_outdf['TODAY_TAT_LDR'] = TODAY_TAT_LDR
        all_outdf['TODAY_TAT_LDR'] = all_outdf['TODAY_TAT_LDR'].dt.days

        # LDR_BAND
        conditions = [
            (all_outdf['TODAY_TAT_LDR'] <= 0),
            (all_outdf['TODAY_TAT_LDR'] > 0) & (all_outdf['TODAY_TAT_LDR'] <= 5),
            (all_outdf['TODAY_TAT_LDR'] > 5) & (all_outdf['TODAY_TAT_LDR'] <= 10),
            (all_outdf['TODAY_TAT_LDR'] > 10)
        ]
        # create a list of the values we want to assign for each condition
        values = ['10 Days & ABOVE', '00-05 Days', '06-10 Days', '10 Days & ABOVE']

        # create a new column and use np.select to assign values to it using our lists as arguments
        all_outdf['LDR_BAND'] = np.select(conditions, values)

        # TAT_LDR
        TAT_LDR = []
        for b in Last_Document_Received_Date:
            TAT_LDR.append(EndOfMonth1.date() - b.date() - timedelta(days=1))
        all_outdf['TAT_LDR'] = TAT_LDR
        all_outdf['TAT_LDR'] = all_outdf['TAT_LDR'].dt.days

        # BAND_LDR_TAT

        all_outdf.loc[(all_outdf['TAT_LDR'] > 30), 'BAND_LDR_TAT'] = 'Above 1 month'
        all_outdf.loc[(all_outdf['TAT_LDR'] <= 30), 'BAND_LDR_TAT'] = 'Below 1 month'

        # TAT_DOA
        DOA = pd.to_datetime(all_outdf['doa'])
        TAT_DOA = []
        for a in DOA:
            TAT_DOA.append(EndOfMonth1 - a.replace(tzinfo=None) - timedelta(days=1))
        all_outdf['TAT_DOA'] = TAT_DOA
        all_outdf['TAT_DOA'] = all_outdf['TAT_DOA'].dt.days

        # BAND_DOA_TAT
        all_outdf.loc[(all_outdf['TAT_DOA'] > 30), 'BAND_DOA_TAT'] = 'Above 1 month'
        all_outdf.loc[(all_outdf['BAND_DOA_TAT'] != 'Above 1 month'), 'BAND_DOA_TAT'] = 'Below 1 month'
        BAND_DOA_TAT = all_outdf['BAND_DOA_TAT']

        # TAT_FIRST_INTIMATION
        all_outdf['first_intimation_date'] = pd.to_datetime(all_outdf['first_intimation_date'].dt.date)
        First_Intimation_Date = pd.to_datetime(all_outdf['first_intimation_date'], format='%y-%m-%d')

        TAT_FIRST_INTIMATION = []
        for i in First_Intimation_Date:
            TAT_FIRST_INTIMATION.append(EndOfMonth1 - i - timedelta(days=1))
        all_outdf['TAT_FIRST_INTIMATION'] = TAT_FIRST_INTIMATION
        all_outdf['TAT_FIRST_INTIMATION'] = all_outdf['TAT_FIRST_INTIMATION'].dt.days

        # YEAR and MONTH_NO

        all_outdf['year_1'] = pd.DatetimeIndex(all_outdf['doa']).year
        all_outdf['month_no'] = pd.DatetimeIndex(all_outdf['doa']).month

        # BAND_FIRST_INTIMATION
        all_outdf.loc[(all_outdf['TAT_FIRST_INTIMATION'] > 90), 'BAND_FIRST_INTIMATION'] = 'Above 3 months'
        all_outdf.loc[(all_outdf[
                           'BAND_FIRST_INTIMATION'] != 'Above 3 months'), 'BAND_FIRST_INTIMATION'] = 'Below 3 months'
        BAND_FIRST_INTIMATION = all_outdf['BAND_FIRST_INTIMATION']

        # DOA_< 1 MONTH_TILL_CURRENT_MONTH_END
        result = all_outdf.assign(count=(all_outdf['BAND_DOA_TAT'] == 'Below 1 month')).groupby(
            'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
        result.rename(columns={'count': 'DOA_<_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
        # doaless1month = result['DOA_<_1_MONTH_TILL_CURRENT_MONTH_END']

        # DOA_> 1 MONTH_TILL_CURRENT_MONTH_END

        result1 = all_outdf.assign(count=(all_outdf['BAND_DOA_TAT'] == 'Above 1 month')).groupby(
            'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})  ### Changed servicing_branch to revised_servicing_branch
        result1.rename(columns={'count': 'DOA_>_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
        doagreat1month = result1['DOA_>_1_MONTH_TILL_CURRENT_MONTH_END']

        # LDR_<_1 MONTH_TILL_CURRENT_MONTH_END

        result2 = all_outdf.assign(
            count=(all_outdf['BAND_LDR_TAT'] == 'Below 1 month') & (all_outdf['revised_servicing_branch']).isin(
                branch_master['branch_name'])).groupby('revised_servicing_branch', sort=False,
                                                       as_index=False).agg({'count': sum})
        result2.rename(columns={'count': 'LDR_<_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)

        # LDR_>_1 MONTH_TILL_CURRENT_MONTH_END

        result3 = all_outdf.assign(count=(all_outdf['BAND_LDR_TAT'] == 'Above 1 month')).groupby(
            'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
        result3.rename(columns={'count': 'LDR_>_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
        ldrgreat1month = result3['LDR_>_1_MONTH_TILL_CURRENT_MONTH_END']

        # FI_DATE_>3MONTH_TILL_CURRENT_MONTH_END

        result4 = all_outdf.assign(count=(all_outdf['BAND_FIRST_INTIMATION'] == 'Above 3 months')).groupby(
            'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
        result4.rename(columns={'count': 'FI_DATE_>3MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
        fidategreaterthreemonth = result4['FI_DATE_>3MONTH_TILL_CURRENT_MONTH_END']

        # FI_DATE_<3MONTH_TILL_CURRENT_MONTH_END

        result5 = all_outdf.assign(count=(all_outdf['BAND_FIRST_INTIMATION'] == 'Below 3 months')).groupby(
            'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
        result5.rename(columns={'count': 'FI_DATE_<3MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
        fidatelessthreemonth = result5['FI_DATE_<3MONTH_TILL_CURRENT_MONTH_END']

        # OS CLAIMS

        result6 = pd.DataFrame(all_outdf.assign(
            count=(all_outdf['revised_servicing_branch']).isin(all_outdf['revised_servicing_branch'])).groupby(
            'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum}))
        result6.rename(columns={'count': 'OS_CLAIMS'}, inplace=True)

        # % OS CLAIMS

        result6['per_OS_claims'] = ((result6['OS_CLAIMS'] / sum(result6['OS_CLAIMS'])) * 100).round(2)

        # % OS CLAIMS > 1_MONTH_DOA

        result1['per_OS_claims > 1_MONTH_DOA'] = (
                (result1['DOA_>_1_MONTH_TILL_CURRENT_MONTH_END'] / result6['OS_CLAIMS']) * 100).round(2)
        osonemonthdoa = result1['per_OS_claims > 1_MONTH_DOA']
        osonemonthdoaact = osonemonthdoa.to_string(index=False)
        # % OS CLAIMS > 1_MONTH_LDR

        result3['per_OS_claims > 1_MONTH_LDR'] = (
                (result3['LDR_>_1_MONTH_TILL_CURRENT_MONTH_END'] / result6['OS_CLAIMS']) * 100).round(2)

        per_os_one_month_ldr = result3['per_OS_claims > 1_MONTH_LDR']
        per_os_one_month_ldract = per_os_one_month_ldr.to_string(index=False)

        # % OS CLAIMS > 3_MONTHS_FIRST_INTIMATION

        result5['per_OS_claims > 3_MONTHS_FIRST_INTIMATION'] = (
                (fidategreaterthreemonth / result6['OS_CLAIMS']) * 100).round(2)
        os_three_month_fi = result5['per_OS_claims > 3_MONTHS_FIRST_INTIMATION']
        os_three_month_fiact = os_three_month_fi.to_string(index=False)

        # Merging

        BRANCH = pd.merge(result6, branch_master, left_on='revised_servicing_branch',
                          right_on='branch_name', how='inner')

        # Concating Results

        BRANCH_UPDATED = pd.concat([BRANCH, result, result1, result2, result3, result4, result5], axis=1)

        # Removing Duplicates

        BRANCH_UPDATED = BRANCH_UPDATED.T.drop_duplicates().T

        BRANCH1 = all_outdf.merge(branch_master[['branch_group_type', 'branch_name']], how='left',
                                  left_on='revised_servicing_branch', right_on='branch_name') ## Changed servicing_branch to revised_servicing_branch

        print("BRANCH1",BRANCH1)

        BRANCH_FINAL = pd.merge(BRANCH1, BRANCH_UPDATED, left_on='revised_servicing_branch',
                                right_on='revised_servicing_branch', how='inner')

        BRANCH_UPDATED.rename(columns={'revised_servicing_branch':'REVISED_SERVICING_BRANCH'}, inplace=True)

        BRANCH_MAIN = BRANCH_UPDATED[(BRANCH_UPDATED['branch_group_type'] == '02 MAIN')]

        BRANCH_HO = BRANCH_UPDATED[(BRANCH_UPDATED['branch_group_type'] == '01 MDI HO')]

        BRANCH_OTHER = BRANCH_UPDATED[(BRANCH_UPDATED['branch_group_type'] == '03 OTHER')]

        BRANCH_MAIN['RANK_MAIN_DOA'] = BRANCH_MAIN['per_OS_claims > 1_MONTH_DOA'].rank(method='min').astype(int)

        BRANCH_HO['RANK_HO_DOA'] = BRANCH_HO['per_OS_claims > 1_MONTH_DOA'].rank(method='min').astype(int)

        BRANCH_OTHER['RANK_OTHER_DOA'] = BRANCH_OTHER['per_OS_claims > 1_MONTH_DOA'].rank(method='min').astype(int)

        BRANCH_MAIN['RANK_MAIN_LDR'] = BRANCH_MAIN['per_OS_claims > 1_MONTH_LDR'].rank(method='min').astype(int)

        BRANCH_HO['RANK_HO_LDR'] = BRANCH_HO['per_OS_claims > 1_MONTH_LDR'].rank(method='min').astype(int)

        BRANCH_OTHER['RANK_OTHER_LDR'] = BRANCH_OTHER['per_OS_claims > 1_MONTH_LDR'].rank(method='min').astype(int)

        BRANCH_MAIN['RANK_MAIN_FI'] = BRANCH_MAIN['per_OS_claims > 3_MONTHS_FIRST_INTIMATION'].rank(method='min').astype(int)

        BRANCH_HO['RANK_HO_FI'] = BRANCH_HO['per_OS_claims > 3_MONTHS_FIRST_INTIMATION'].rank(method='min').astype(int)

        BRANCH_OTHER['RANK_OTHER_FI'] = BRANCH_OTHER['per_OS_claims > 3_MONTHS_FIRST_INTIMATION'].rank(method='min').astype(int)


        # ####################################################RANK_HO_DOA#################################################

        RANK_HO_DOA1 = BRANCH_HO[['REVISED_SERVICING_BRANCH', 'RANK_HO_DOA']].sort_values(by='RANK_HO_DOA',
                                                                                          ascending=True).reset_index(
            drop=True)
        RANK_HO_DOA1.rename(columns = {'REVISED_SERVICING_BRANCH':'BRANCH', 'RANK_HO_DOA':'HO_DOA'}, inplace=True)

        colorscale = [[0, '#272D31'], [.5, '#ffffff'], [1, '#ffffff']]
        font = ['#FCFCFC', '#006400', '#52FF00', '#FFD900', '#FF7700', '#FF1500']

        # Create the table
        # RANK_HO_DOA2 = ff.create_table(
        #     header=dict(values=list(RANK_HO_DOA1.columns)),
        #     cells=dict(values=RANK_HO_DOA1.values),
        #     font_colors=[font],  # Use the provided font colors list
        #     colorscale=colorscale
        # )
        RANK_HO_DOA2 = ff.create_table(np.vstack([RANK_HO_DOA1.columns, RANK_HO_DOA1.values]),
                              colorscale=colorscale)

        RANK_HO_DOA2.update_layout(title_text='RANK ON > THAN 1 MONTH LAT')
        RANK_HO_DOA2.layout.width = 250
        RANK_HO_DOA = plot(RANK_HO_DOA2, output_type='div')
        # RANK_HO_DOA = plot(RANK_HO_DOA2, output_type='div', config = {'staticPlot': True})

##############################################RANK_MAIN_DOA1#######################################################
        RANK_MAIN_DOA1 = BRANCH_MAIN[['REVISED_SERVICING_BRANCH', 'RANK_MAIN_DOA']].sort_values(
            by='RANK_MAIN_DOA', ascending=True).reset_index(drop=True)
        RANK_MAIN_DOA1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_MAIN_DOA': 'MAIN_DOA'}, inplace=True)

        main_doa_font = ['#FCFCFC', '#00EE00', '#008B00', '#FFC500', '#FFB200', '#FF8C00', '#FF3030','#FF5000','#FF3C00', '#FF2900','#FF1500']

        RANK_MAIN_DOA2 = ff.create_table(np.vstack([RANK_MAIN_DOA1.columns, RANK_MAIN_DOA1.values]), font_colors=main_doa_font,
                              colorscale=colorscale)
        RANK_MAIN_DOA2.layout.width = 250
        RANK_MAIN_DOA = plot(RANK_MAIN_DOA2, output_type='div')

        ##############################################RANK_OTHER_DOA#######################################################

        RANK_OTHER_DOA1 = BRANCH_OTHER[['REVISED_SERVICING_BRANCH', 'RANK_OTHER_DOA']].sort_values(
            by='RANK_OTHER_DOA', ascending=True).reset_index(drop=True)
        RANK_OTHER_DOA1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_OTHER_DOA': 'OTHER_DOA'}, inplace=True)

        other_doa_font = ['#FCFCFC', '#00EE00', '#008B00', '#FFC500', '#FFB200', '#FF8C00', '#FF3030','#FF5000','#FF3C00', '#FF2900','#FF1500']


        RANK_OTHER_DOA2 = ff.create_table(np.vstack([RANK_OTHER_DOA1.columns, RANK_OTHER_DOA1.values]))
        RANK_OTHER_DOA2.layout.width = 250
        RANK_OTHER_DOA = plot(RANK_OTHER_DOA2, output_type='div')


        ############################################## RANK_MAIN_LDR ######################################################

        RANK_MAIN_LDR1 = BRANCH_MAIN[['REVISED_SERVICING_BRANCH', 'RANK_MAIN_LDR']].sort_values(
            by='RANK_MAIN_LDR', ascending=True).reset_index(drop=True)
        RANK_MAIN_LDR1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_MAIN_LDR': 'MAIN_LDR'}, inplace=True)


        RANK_MAIN_LDR2 = ff.create_table(np.vstack([RANK_MAIN_LDR1.columns, RANK_MAIN_LDR1.values]))
        RANK_MAIN_LDR2.layout.width = 250
        RANK_MAIN_LDR = plot(RANK_MAIN_LDR2, output_type='div')

        ###### RANK_HO_LDR #####

        RANK_HO_LDR1 = BRANCH_HO[['REVISED_SERVICING_BRANCH', 'RANK_HO_LDR']].sort_values(by='RANK_HO_LDR',
                                                                                          ascending=True).reset_index(
            drop=True)

        RANK_HO_LDR1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_HO_LDR': 'HO_LDR'}, inplace=True)

        RANK_HO_LDR2 = ff.create_table(np.vstack([RANK_HO_LDR1.columns, RANK_HO_LDR1.values]))
        RANK_HO_LDR2.update_layout(title_text='RANK ON > 1 MONTH DOA')
        RANK_HO_LDR2.layout.width = 250
        RANK_HO_LDR = plot(RANK_HO_LDR2, output_type='div')

        ###### RANK_OTHER_LDR ######

        RANK_OTHER_LDR1 = BRANCH_OTHER[['REVISED_SERVICING_BRANCH', 'RANK_OTHER_LDR']].sort_values(
            by='RANK_OTHER_LDR', ascending=True).reset_index(drop=True)
        RANK_OTHER_LDR1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_OTHER_LDR': 'OTHER_LDR'}, inplace=True)

        RANK_OTHER_LDR2 = ff.create_table(np.vstack([RANK_OTHER_LDR1.columns, RANK_OTHER_LDR1.values]))
        RANK_OTHER_LDR2.layout.width = 250
        RANK_OTHER_LDR = plot(RANK_OTHER_LDR2, output_type='div')


        ###### RANK_MAIN_FI ######

        RANK_MAIN_FI1 = BRANCH_MAIN[['REVISED_SERVICING_BRANCH', 'RANK_MAIN_FI']].sort_values(by='RANK_MAIN_FI',
                                                                                              ascending=True).reset_index(
            drop=True)
        RANK_MAIN_FI1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_MAIN_FI': 'MAIN_FI'}, inplace=True)

        RANK_MAIN_FI2 = ff.create_table(np.vstack([RANK_MAIN_FI1.columns, RANK_MAIN_FI1.values]))
        RANK_MAIN_FI2.layout.width = 250
        RANK_MAIN_FI = plot(RANK_MAIN_FI2, output_type='div')


        ###### RANK_HO_FI ######

        RANK_HO_FI1 = BRANCH_HO[['REVISED_SERVICING_BRANCH', 'RANK_HO_FI']].sort_values(by='RANK_HO_FI',
                                                                                        ascending=True).reset_index(
            drop=True)
        RANK_HO_FI1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_HO_FI': 'HO_FI'}, inplace=True)

        RANK_HO_FI2 = ff.create_table(np.vstack([RANK_HO_FI1.columns, RANK_HO_FI1.values]))
        RANK_HO_FI2.update_layout(title='RANK ON > 3 MONTH FIRST INTIMATION')
        RANK_HO_FI2.layout.width = 250
        RANK_HO_FI = plot(RANK_HO_FI2, output_type='div')

        ###### RANK_OTHER_FI ######

        RANK_OTHER_FI1 = BRANCH_OTHER[['REVISED_SERVICING_BRANCH', 'RANK_OTHER_FI']].sort_values(
            by='RANK_OTHER_FI', ascending=True).reset_index(drop=True)
        RANK_OTHER_FI1.rename(columns={'REVISED_SERVICING_BRANCH': 'BRANCH', 'RANK_OTHER_FI': 'OTHER_FI'}, inplace=True)

        RANK_OTHER_FI2 = ff.create_table(np.vstack([RANK_OTHER_FI1.columns, RANK_OTHER_FI1.values]))
        RANK_OTHER_FI2.layout.width = 250
        RANK_OTHER_FI = plot(RANK_OTHER_FI2, output_type='div')

        # # create a Dash app
        # app = dash.Dash(__name__)
        #
        # # create a Dash DataTable
        # table = dash_table.DataTable(data=RANK_MAIN_DOA1.to_dict('records'),
        #                              columns=[{'id': c, 'name': c} for c in RANK_MAIN_DOA1.columns],
        #                              style_data_conditional=[
        #                                  {'if': {'column_id': 'MAIN_DOA', 'filter_query': '{MAIN_DOA} <= 3'},
        #                                   'backgroundColor': 'red', 'color': 'white'},
        #                                  {'if': {'column_id': 'MAIN_DOA',
        #                                          'filter_query': '{MAIN_DOA} > 3 && {MAIN_DOA} <= 7'},
        #                                   'backgroundColor': 'orange', 'color': 'white'},
        #                                  {'if': {'column_id': 'MAIN_DOA', 'filter_query': '{MAIN_DOA} > 7'},
        #                                   'backgroundColor': 'green', 'color': 'white'}
        #                              ])
        #
        # # add the table to the app layout
        # app.layout = table
        #
        # app1 = plot(table, output_type='div')




    mycontext={
            'pred_plot':pred_plot,
            'out_count':out_count,
            'settled_count':settled_count,
            'RANK_HO_DOA':RANK_HO_DOA,
            'RANK_MAIN_DOA':RANK_MAIN_DOA,
            'RANK_OTHER_DOA':RANK_OTHER_DOA,
            'RANK_HO_LDR':RANK_HO_LDR,
            'RANK_MAIN_LDR' :RANK_MAIN_LDR,
            'RANK_OTHER_LDR' :RANK_OTHER_LDR,
            'RANK_HO_FI' :RANK_HO_FI,
            'RANK_MAIN_FI':RANK_MAIN_FI,
            'RANK_OTHER_FI':RANK_OTHER_FI,
        }
    return render(request,'Management/dashboard.html',context=mycontext)




def my_dash_app(request):
    app = DjangoDash('my-dash-app')

    # create a sample DataFrame
    df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie', 'Dave'],
                       'Score': [80, 65, 90, 70]})

    # create a Dash DataTable
    table = dash_table.DataTable(data=df.to_dict('records'),
                                 columns=[{'id': c, 'name': c} for c in df.columns],
                                 style_data_conditional=[
                                     {'if': {'column_id': 'Score', 'filter_query': '{Score} < 70'},
                                      'backgroundColor': 'red', 'color': 'white'},
                                     {'if': {'column_id': 'Score', 'filter_query': '{Score} >= 70 && {Score} < 80'},
                                      'backgroundColor': 'orange', 'color': 'white'},
                                     {'if': {'column_id': 'Score', 'filter_query': '{Score} >= 80'},
                                      'backgroundColor': 'green', 'color': 'white'}
                                 ])

    app.layout = table

    return render(request, 'Management/my_template.html', {'dash_app': app})

def customer_care(request):
    if request.user.is_authenticated:
        return render(request, 'Management/crsclaims.html', )


def branches(request):
    if request.user.is_authenticated:
        oscount = None
        alt_label = None
        alt_data = None
        head_label = None
        head_data = None
        osonemonthdoaact = None
        os_three_month_fiact = None
        per_os_one_month_ldract = None
        no_of_claims_vis_a_vis_outstanding_reasons1 = None
        ic_wise_outstanding_reasons1 = None
        reason_wise_corporate_retail_os = None
        reason_wise_corporate_retail_os1 = None
        branch_wise_adr_count_and_avg_tat1 = None
        os_claims_by_actuallosstype_fig1 = None
        os_claims_by_head1=None
        BRANCH_UPDATED1 = None
        MDI_BRANCHES_OUTSTANDING_STATS1 = None


        form = SelectBranch(request.POST or None)
        if request.method == 'POST':
            branch = request.POST.get('branch')
            branch_qs = Allicdailyoutstanding.objects.filter(revised_servicing_branch=branch).values('servicing_branch','actual_loss_type','revised_servicing_branch','servicing_branch','head','last_document_received','doa','first_intimation_date','sub_head','liablity_reserve_amt','ic_name','corp_retail')
            if len(branch_qs) > 0:
                pd.set_option("styler.format.thousands", ",")
                branch_df = pd.DataFrame(branch_qs.values())
                branch = BranchPortal.objects.all().values()
                branch_master = pd.DataFrame(branch)
                today = date.today()
                EndOfMonth1 = today + MonthEnd(1)
                oscount = len(branch_df.index)
                branch_df['servicing_branch'] = branch_df['servicing_branch'].str.upper()
                # End Of Month
                config = {'responsive': True, 'displaylogo': False}
                # EndOfMonth = now + MonthEnd(1)
                # EndOfMonth1 =EndOfMonth - timedelta(days=1)
                # EndOfMonth1 = pd.to_datetime(EndOfMonth1, format='%y-%m-%d')
                EndOfMonth1 = today + MonthEnd(1)

                # for actual loss
                alt = branch_df['actual_loss_type'].value_counts().reset_index()

                os_claims_by_actuallosstype = [go.Pie(labels=alt['actual_loss_type'],
                                                      values=alt['count'], text=alt['count'],
                                                      marker_colors=px.colors.qualitative.Plotly)
                                               ]
                os_claims_by_actuallosstype_fig = go.Figure(data=os_claims_by_actuallosstype)
                os_claims_by_actuallosstype_fig.update_layout(margin=dict(t=100, b=100, l=100, r=100),
                                                              title='')

                os_claims_by_actuallosstype_fig1 = plot(os_claims_by_actuallosstype_fig, output_type='div', config=config)

                # for graph by head
                byhead = branch_df['head'].value_counts().reset_index()
                os_claims_by_head = [go.Pie(labels=byhead['head'],
                                                      values=byhead['count'], text=byhead['count'],
                                                      marker_colors=px.colors.qualitative.Plotly)
                                               ]
                os_claims_by_head = go.Figure(data=os_claims_by_head)
                os_claims_by_head.update_layout(margin=dict(t=100, b=100, l=100, r=100),
                                                              title='')

                os_claims_by_head1 = plot(os_claims_by_head, output_type='div', config=config)

                # TODAY_TAT_LDR
                branch_df['last_document_received'] = pd.to_datetime(branch_df['last_document_received'],
                                                                     format='%y-%m-%d')

                Last_Document_Recived_date = branch_df['last_document_received']

                TODAY_TAT_LDR = []
                for d in Last_Document_Recived_date:
                    TODAY_TAT_LDR.append(today - d.date())
                branch_df['TODAY_TAT_LDR'] = TODAY_TAT_LDR
                branch_df['TODAY_TAT_LDR'] = branch_df['TODAY_TAT_LDR'].dt.days

                # LDR_BAND
                conditions = [
                    (branch_df['TODAY_TAT_LDR'] <= 0),
                    (branch_df['TODAY_TAT_LDR'] > 0) & (branch_df['TODAY_TAT_LDR'] <= 5),
                    (branch_df['TODAY_TAT_LDR'] > 5) & (branch_df['TODAY_TAT_LDR'] <= 10),
                    (branch_df['TODAY_TAT_LDR'] > 10)
                ]
                # create a list of the values we want to assign for each condition
                values = ['10 Days & ABOVE', '00-05 Days', '06-10 Days', '10 Days & ABOVE']

                # create a new column and use np.select to assign values to it using our lists as arguments
                branch_df['LDR_BAND'] = np.select(conditions, values)

                # TAT_LDR
                TAT_LDR = []
                for b in Last_Document_Recived_date:
                    TAT_LDR.append(EndOfMonth1.date() - b.date() - timedelta(days=1))
                branch_df['TAT_LDR'] = TAT_LDR
                branch_df['TAT_LDR'] = branch_df['TAT_LDR'].dt.days

                # BAND_LDR_TAT

                branch_df.loc[(branch_df['TAT_LDR'] > 30), 'BAND_LDR_TAT'] = 'Above 1 month'
                branch_df.loc[(branch_df['TAT_LDR'] <= 30), 'BAND_LDR_TAT'] = 'Below 1 month'

                # TAT_DOA
                DOA = pd.to_datetime(branch_df['doa'])
                TAT_DOA = []
                for a in DOA:
                    TAT_DOA.append(EndOfMonth1 - a.replace(tzinfo=None) - timedelta(days=1))
                branch_df['TAT_DOA'] = TAT_DOA
                branch_df['TAT_DOA'] = branch_df['TAT_DOA'].dt.days

                # BAND_DOA_TAT
                branch_df.loc[(branch_df['TAT_DOA'] > 30), 'BAND_DOA_TAT'] = 'Above 1 month'
                branch_df.loc[(branch_df['BAND_DOA_TAT'] != 'Above 1 month'), 'BAND_DOA_TAT'] = 'Below 1 month'
                BAND_DOA_TAT = branch_df['BAND_DOA_TAT']

                # TAT_FIRST_INTIMATION
                branch_df['first_intimation_date'] = pd.to_datetime(branch_df['first_intimation_date'].dt.date)
                First_Intimation_Date = pd.to_datetime(branch_df['first_intimation_date'], format='%y-%m-%d')

                TAT_FIRST_INTIMATION = []
                for i in First_Intimation_Date:
                    TAT_FIRST_INTIMATION.append(EndOfMonth1 - i - timedelta(days=1))
                branch_df['TAT_FIRST_INTIMATION'] = TAT_FIRST_INTIMATION
                branch_df['TAT_FIRST_INTIMATION'] = branch_df['TAT_FIRST_INTIMATION'].dt.days

                # YEAR and MONTH_NO

                branch_df['year_1'] = pd.DatetimeIndex(branch_df['doa']).year
                branch_df['month_no'] = pd.DatetimeIndex(branch_df['doa']).month

                # BAND_FIRST_INTIMATION
                branch_df.loc[(branch_df['TAT_FIRST_INTIMATION'] > 90), 'BAND_FIRST_INTIMATION'] = 'Above 3 months'
                branch_df.loc[(branch_df[
                                   'BAND_FIRST_INTIMATION'] != 'Above 3 months'), 'BAND_FIRST_INTIMATION'] = 'Below 3 months'
                BAND_FIRST_INTIMATION = branch_df['BAND_FIRST_INTIMATION']

                # DOA_< 1 MONTH_TILL_CURRENT_MONTH_END
                result = branch_df.assign(count=(branch_df['BAND_DOA_TAT'] == 'Below 1 month')).groupby(
                    'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
                result.rename(columns={'count': 'DOA_<_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
                # doaless1month = result['DOA_<_1_MONTH_TILL_CURRENT_MONTH_END']

                # DOA_> 1 MONTH_TILL_CURRENT_MONTH_END

                result1 = branch_df.assign(count=(branch_df['BAND_DOA_TAT'] == 'Above 1 month')).groupby(
                    'servicing_branch', sort=False, as_index=False).agg({'count': sum})
                result1.rename(columns={'count': 'DOA_>_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
                doagreat1month = result1['DOA_>_1_MONTH_TILL_CURRENT_MONTH_END']

                # LDR_<_1 MONTH_TILL_CURRENT_MONTH_END

                result2 = branch_df.assign(
                    count=(branch_df['BAND_LDR_TAT'] == 'Below 1 month') & (branch_df['revised_servicing_branch']).isin(
                        branch_master['branch_name'])).groupby('revised_servicing_branch', sort=False,
                                                               as_index=False).agg({'count': sum})
                result2.rename(columns={'count': 'LDR_<_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)

                # LDR_>_1 MONTH_TILL_CURRENT_MONTH_END

                result3 = branch_df.assign(count=(branch_df['BAND_LDR_TAT'] == 'Above 1 month')).groupby(
                    'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
                result3.rename(columns={'count': 'LDR_>_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
                ldrgreat1month = result3['LDR_>_1_MONTH_TILL_CURRENT_MONTH_END']

                # FI_DATE_>3MONTH_TILL_CURRENT_MONTH_END

                result4 = branch_df.assign(count=(branch_df['BAND_FIRST_INTIMATION'] == 'Above 3 months')).groupby(
                    'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
                result4.rename(columns={'count': 'FI_DATE_>3MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
                fidategreaterthreemonth = result4['FI_DATE_>3MONTH_TILL_CURRENT_MONTH_END']

                # FI_DATE_<3MONTH_TILL_CURRENT_MONTH_END

                result5 = branch_df.assign(count=(branch_df['BAND_FIRST_INTIMATION'] == 'Below 3 months')).groupby(
                    'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
                result5.rename(columns={'count': 'FI_DATE_<3MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)
                fidatelessthreemonth = result5['FI_DATE_<3MONTH_TILL_CURRENT_MONTH_END']

                # OS CLAIMS

                result6 = pd.DataFrame(branch_df.assign(
                    count=(branch_df['revised_servicing_branch']).isin(branch_master['branch_name'])).groupby(
                    'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum}))
                result6.rename(columns={'count': 'OS_CLAIMS'}, inplace=True)

                # % OS CLAIMS

                result6['per_OS_claims'] = ((result6['OS_CLAIMS'] / sum(result6['OS_CLAIMS'])) * 100).round(2)

                # % OS CLAIMS > 1_MONTH_DOA

                result1['per_OS_claims > 1_MONTH_DOA'] = (
                            (result1['DOA_>_1_MONTH_TILL_CURRENT_MONTH_END'] / result6['OS_CLAIMS']) * 100).round(2)
                osonemonthdoa = result1['per_OS_claims > 1_MONTH_DOA']
                osonemonthdoaact = osonemonthdoa.to_string(index=False)

                # % OS CLAIMS > 1_MONTH_LDR

                result3['per_OS_claims > 1_MONTH_LDR'] = (
                            (result3['LDR_>_1_MONTH_TILL_CURRENT_MONTH_END'] / result6['OS_CLAIMS']) * 100).round(2)

                per_os_one_month_ldr = result3['per_OS_claims > 1_MONTH_LDR']
                per_os_one_month_ldract = per_os_one_month_ldr.to_string(index=False)

                # % OS CLAIMS > 3_MONTHS_FIRST_INTIMATION

                result5['per_OS_claims > 3_MONTHS_FIRST_INTIMATION'] = (
                            (fidategreaterthreemonth / result6['OS_CLAIMS']) * 100).round(2)
                os_three_month_fi = result5['per_OS_claims > 3_MONTHS_FIRST_INTIMATION']
                os_three_month_fiact = os_three_month_fi.to_string(index=False)

                print(result5)

                # Merging

                BRANCH = pd.merge(result6, branch_master, left_on='revised_servicing_branch',
                                  right_on='branch_name', how='inner')

                # Concating Results

                BRANCH_UPDATED0 = pd.concat([BRANCH, result, result1, result2, result3, result4, result5], axis=1)
                print(BRANCH_UPDATED0.columns)
                # Removing Duplicates

                BRANCH_UPDATED = BRANCH_UPDATED0.T.drop_duplicates().T
                # BRANCH_UPDATED1 = BRANCH_UPDATED.to_html()


                BRANCH1 = branch_df.merge(branch_master[['branch_group_type', 'branch_name']], how='left',
                                          left_on='servicing_branch', right_on='branch_name')

                BRANCH_FINAL = pd.merge(BRANCH1, BRANCH_UPDATED, left_on='revised_servicing_branch',
                                        right_on='revised_servicing_branch', how='inner')

                BRANCHES_OUTSTANDING_STATS = BRANCH_UPDATED.groupby(['revised_servicing_branch'])[['per_OS_claims > 3_MONTHS_FIRST_INTIMATION', 'per_OS_claims > 1_MONTH_LDR', 'per_OS_claims > 1_MONTH_DOA']].agg('sum')
                MDI_BRANCHES_OUTSTANDING_STATS = BRANCHES_OUTSTANDING_STATS.reset_index()
                # create the trace for each value column
                trace1 = go.Bar(
                    x=MDI_BRANCHES_OUTSTANDING_STATS['revised_servicing_branch'],
                    y=MDI_BRANCHES_OUTSTANDING_STATS['per_OS_claims > 3_MONTHS_FIRST_INTIMATION'],
                    name='% OS CLAIMS > 3_MONTHS_FIRST_INTIMATION'
                )
                trace2 = go.Bar(
                    x=MDI_BRANCHES_OUTSTANDING_STATS['revised_servicing_branch'],
                    y=MDI_BRANCHES_OUTSTANDING_STATS['per_OS_claims > 1_MONTH_LDR'],
                    name='% OS CLAIMS > 1_MONTH_LDR'
                )
                trace3 = go.Bar(
                    x=MDI_BRANCHES_OUTSTANDING_STATS['revised_servicing_branch'],
                    y=MDI_BRANCHES_OUTSTANDING_STATS['per_OS_claims > 1_MONTH_DOA'],
                    name='% OS CLAIMS > 1_MONTH_DOA'
                )

                # create the layout
                layout = go.Layout(
                    width=550,
                    title='MDI BRANCHES OUTSTANDING STATS',
                    xaxis=dict(title='SERVICING BRANCH'),
                    yaxis=dict(title=''),
                    barmode='group'
                )


                # create the figure and plot it
                MDI_BRANCHES_OUTSTANDING_STATS2 = go.Figure(data=[trace1, trace2, trace3], layout=layout)
                MDI_BRANCHES_OUTSTANDING_STATS1 = plot(MDI_BRANCHES_OUTSTANDING_STATS2,output_type='div', config=config)


                # tat_ldr_avg

                tat_ldr_avg = BRANCH_FINAL.groupby(['sub_head'])['TAT_LDR'].agg('mean').round(2)
                tat_ldr_avg = tat_ldr_avg.reset_index()
                tat_ldr_avg.rename(columns={'TAT_LDR': 'TAT_LDR_AVG'}, inplace=True)

                # liablity_reserve_amt

                liablity_reserve_amt1 = (BRANCH_FINAL.groupby(['sub_head'])['liablity_reserve_amt'].agg('sum'))

                liablity_reserve_amt3 = liablity_reserve_amt1.reset_index()
                liablity_reserve_amt2 = liablity_reserve_amt3['liablity_reserve_amt'].apply(np.ceil)
                liablity_reserve_amt4 = pd.DataFrame(liablity_reserve_amt2)
                liablity_reserve_amt5 = liablity_reserve_amt4.style.format({
                    "liablity_reserve_amt": "{:,d}"})
                liablity_reserve_amt = liablity_reserve_amt5

                # OS_CLAIMS_BY_SubHead

                os_claims_by_sub_head = BRANCH_FINAL.groupby(['sub_head'])['OS_CLAIMS'].agg('count')
                os_claims_by_sub_head = os_claims_by_sub_head.reset_index()

                per_os_claims_by_subhead = BRANCH_FINAL.groupby('sub_head')['OS_CLAIMS'].count().rename(
                    "%_OS_CLAIMS").transform(lambda x: x / x.sum() * 100)

                per_os_claims_by_subhead = per_os_claims_by_subhead.reset_index().round(2)

                # Per_Liablity_Reserve_Amt

                per_liablity_reserve_amt = BRANCH_FINAL.groupby(['sub_head'])['liablity_reserve_amt'].sum().rename(
                    "%_liablity_reserve_amts").transform(lambda x: x / x.sum() * 100)

                per_liablity_reserve_amt1 = per_liablity_reserve_amt.reset_index()
                per_liablity_reserve_amt1 = per_liablity_reserve_amt1['%_liablity_reserve_amts'].apply(np.ceil)

                no_of_claims_vis_a_vis_outstanding_reasons = pd.concat(
                    [os_claims_by_sub_head, tat_ldr_avg, per_os_claims_by_subhead, liablity_reserve_amt2,
                     per_liablity_reserve_amt1], axis=1)
                no_of_claims_vis_a_vis_outstanding_reasons = no_of_claims_vis_a_vis_outstanding_reasons.T.drop_duplicates().T

                no_of_claims_vis_a_vis_outstanding_reasons_fig = go.Figure(
                    data=[go.Table(columnwidth=[150, 70, 90, 90, 110, 130],

                                   header=dict(values=list(no_of_claims_vis_a_vis_outstanding_reasons.columns),
                                               fill_color='#094780',
                                               line_color='darkslategray',
                                               align='left',
                                               font = dict(size = 12, color='white')),
                                   cells=dict(values=no_of_claims_vis_a_vis_outstanding_reasons.T,
                                              fill_color='white',
                                              line_color='darkslategray',
                                              align='left'))
                          ])

                no_of_claims_vis_a_vis_outstanding_reasons_fig.layout.width = 1200
                no_of_claims_vis_a_vis_outstanding_reasons_fig.update_layout(height=len(no_of_claims_vis_a_vis_outstanding_reasons) * 40)
                no_of_claims_vis_a_vis_outstanding_reasons_fig.update_traces(cells_font=dict(size=10))

                no_of_claims_vis_a_vis_outstanding_reasons1 = plot(no_of_claims_vis_a_vis_outstanding_reasons_fig, output_type='div', config=config)

                # ic_wise_outstanding_reasons

                ic_wise_outstanding_reasons = pd.crosstab(branch_df['ic_name'], branch_df['sub_head'])
                ic_wise_outstanding_reasons = ic_wise_outstanding_reasons.reset_index()

                ic_wise_outstanding_reasons_fig = go.Figure(
                    data=[go.Table(columnwidth=[150, 70, 90, 90, 110, 130],

                                   header=dict(values=list(ic_wise_outstanding_reasons.columns),
                                               fill_color='#094780',
                                               line_color='darkslategray',
                                               align='left',
                                               font=dict(size=12, color='white')
                                               ),
                                   cells=dict(values=ic_wise_outstanding_reasons.T,
                                              fill_color='white',
                                              line_color='darkslategray',
                                              align='left'))
                          ])

                ic_wise_outstanding_reasons_fig.layout.width = 1200
                ic_wise_outstanding_reasons_fig.update_layout(height=len(ic_wise_outstanding_reasons) * 80)
                ic_wise_outstanding_reasons_fig.update_traces(cells_font=dict(size=10))

                ic_wise_outstanding_reasons1 = plot(ic_wise_outstanding_reasons_fig, output_type='div', config=config)


                # branch_wise_adr_count_and_avg_tat
                branch_wise_adr_count_and_avg_tat = BRANCH_FINAL.groupby(['revised_servicing_branch'])['TAT_LDR'].agg(
                    'mean')
                branch_wise_adr_count_and_avg_tat2 = branch_wise_adr_count_and_avg_tat.round(2)
                branch_wise_adr_count_and_avg_tat3 = branch_wise_adr_count_and_avg_tat2.reset_index()
                branch_wise_adr_count_and_avg_tat4 = branch_wise_adr_count_and_avg_tat3.rename(
                    columns={'index': 'branch_wise_adr_count_and_avg_tat4'})
                tat_ldr = branch_wise_adr_count_and_avg_tat4['TAT_LDR']
                tat_ldr = tat_ldr[0]

                branch_wise_adr_count_and_avg_tat1 = tat_ldr.tolist()

                # reason_wise_corporate_retail_os
                reason_wise_corporate_retail_os = pd.crosstab(branch_df['sub_head'], branch_df['corp_retail'])
                reason_wise_corporate_retail_os = reason_wise_corporate_retail_os.reset_index()

                reason_wise_corporate_retail_os_fig = go.Figure(
                    data=[go.Table(columnwidth=[40, 20, 20],

                                   header=dict(values=list(reason_wise_corporate_retail_os.columns),
                                               fill_color='#094780',
                                               line_color='darkslategray',
                                               align='left',
                                               font = dict(size = 12, color='white')),
                                   cells=dict(values=reason_wise_corporate_retail_os.T,
                                              fill_color='white',
                                              line_color='darkslategray',
                                              align='left'))
                          ])

                # reason_wise_corporate_retail_os_fig.layout.width = 1200
                reason_wise_corporate_retail_os_fig.update_layout(height=len(reason_wise_corporate_retail_os) * 50)
                reason_wise_corporate_retail_os_fig.update_traces(cells_font=dict(size=10))

                reason_wise_corporate_retail_os1 = plot(reason_wise_corporate_retail_os_fig, output_type='div', config=config)


                # outstanding_reasons = pd.crosstab(branch_df['ic_name'], branch_df['sub_head'])
                # outstanding_reasonsa = outstanding_reasons.to_html()
                #
                # No_of_Claims_Vis_a_vis_Outstanding_Reasons = pd.concat(
                #     [OS_CLAIMS_BY_SubHead, TAT_LDR_AVG, Per_OS_CLAIMS_BY_SubHead, Liablity_Reserve_Amt,
                #      Per_Liablity_Reserve_Amt], axis=1)
                # No_of_Claims_Vis_a_vis_Outstanding_Reasons = No_of_Claims_Vis_a_vis_Outstanding_Reasons.T.drop_duplicates().T
                # No_of_Claims_Vis_a_vis_Outstanding_Reasons



            else:
                    messages.warning(request, "Apparently no values available...")
        mydict = {
            'form': form,
            'oscount': oscount,
            'os_claims_by_actuallosstype_fig1':os_claims_by_actuallosstype_fig1,
            'os_claims_by_head1':os_claims_by_head1,
            'MDI_BRANCHES_OUTSTANDING_STATS1':MDI_BRANCHES_OUTSTANDING_STATS1,
            'no_of_claims_vis_a_vis_outstanding_reasons1': no_of_claims_vis_a_vis_outstanding_reasons1,
            'ic_wise_outstanding_reasons': ic_wise_outstanding_reasons1,
            'reason_wise_corporate_retail_os': reason_wise_corporate_retail_os1,
            'branch_wise_adr_count_and_avg_tat': branch_wise_adr_count_and_avg_tat1,
            'os_three_month_fiact': os_three_month_fiact,
            'per_os_one_month_ldract': per_os_one_month_ldract,
            'osonemonthdoaact': osonemonthdoaact,
            'BRANCH_UPDATED1': BRANCH_UPDATED1,
        }
    return render(request, 'Management/branches.html', context=mydict)


def headoffice(request):
    if request.user.is_authenticated:
        form = SelectHeadOffice(request.POST or None)
        today = date.today()
        oscount_ho = None
        actual_label = None
        actual_data = None
        head_data = None
        head_label = None
        os_claims_by_actuallosstype_fig1 = None
        os_claims_by_head1 = None
        MDI_BRANCHES_OUTSTANDING_STATS1 = None
        os_three_month_fiact = None
        per_os_one_month_ldract = None
        no_of_claims_vis_a_vis_outstanding_reasons1 = None
        ic_wise_outstanding_reasons1 = None
        reason_wise_corporate_retail_os1 = None
        branch_wise_adr_count_and_avg_tat1 = None
        osonemonth_doaact = None
        BRANCH_UPDATED1 = None
        ldr_data = None
        ldr_label = None
        ten_days_above = None
        six_to_ten_days = None
        zero_to_five_days = None
        ic = None

        hodept = BranchPortal.objects.all().values()
        hodept_master = pd.DataFrame(hodept)
        if request.method == 'POST':
            headoffice = request.POST.get('headoffice')
            headoffice_qs = Allicdailyoutstanding.objects.filter(revised_servicing_branch=headoffice).values('servicing_branch','actual_loss_type','revised_servicing_branch','servicing_branch','head','last_document_received','doa','first_intimation_date','sub_head','liablity_reserve_amt','ic_name','corp_retail')
            if len(headoffice_qs) > 0:
                headoffice_df = pd.DataFrame(headoffice_qs.values())
                oscount_ho = len(headoffice_df.index)
                config = {'responsive': True, 'displaylogo': False}

                # End Of Month
                # EndOfMonth = now + MonthEnd(1)
                # EndOfMonth1 = EndOfMonth - timedelta(days=1)
                # EndOfMonth1 = pd.to_datetime(EndOfMonth1, format='%y-%m-%d')
                today = date.today()
                EndOfMonth1 = today + MonthEnd(1)

                # for actual loss
                alt = headoffice_df['actual_loss_type'].value_counts().reset_index()

                os_claims_by_actuallosstype = [go.Pie(labels=alt['actual_loss_type'],
                                                      values=alt['count'], text=alt['count'],
                                                      marker_colors=px.colors.qualitative.Plotly)
                                               ]
                os_claims_by_actuallosstype_fig = go.Figure(data=os_claims_by_actuallosstype)
                os_claims_by_actuallosstype_fig.update_layout(margin=dict(t=100, b=100, l=100, r=100),
                                                              title='')

                os_claims_by_actuallosstype_fig1 = plot(os_claims_by_actuallosstype_fig, output_type='div', config=config)


                # for graph by head
                byhead = headoffice_df['head'].value_counts().reset_index()
                os_claims_by_head = [go.Pie(labels=byhead['head'],
                                                      values=byhead['count'], text=byhead['count'],
                                                      marker_colors=px.colors.qualitative.Plotly)
                                               ]
                os_claims_by_head = go.Figure(data=os_claims_by_head)
                os_claims_by_head.update_layout(margin=dict(t=100, b=100, l=100, r=100),
                                                              title='')

                os_claims_by_head1 = plot(os_claims_by_head, output_type='div', config=config)

                # Today_tat_ LDR

                headoffice_df['last_document_received'] = pd.to_datetime(headoffice_df['last_document_received'],
                                                                         format='%y-%m-%d')
                last_document_recived_date = headoffice_df['last_document_received']

                Today_TAT_LDR = []
                for i in last_document_recived_date:
                    Today_TAT_LDR.append(today - i.date())
                headoffice_df['Today_TAT_LDR'] = Today_TAT_LDR
                headoffice_df['Today_TAT_LDR'] = headoffice_df['Today_TAT_LDR'].dt.days

                # LDR_Band
                conditions = [
                    (headoffice_df['Today_TAT_LDR'] <= 0),
                    (headoffice_df['Today_TAT_LDR'] > 0) & (headoffice_df['Today_TAT_LDR'] <= 5),
                    (headoffice_df['Today_TAT_LDR'] > 5) & (headoffice_df['Today_TAT_LDR'] <= 10),
                    (headoffice_df['Today_TAT_LDR'] > 10)

                ]
                values = ['10 Days & Above', '00-05 Days', '06-10 Days', '10 Days & Above']

                headoffice_df['LDR_BAND'] = np.select(conditions, values)

                ldr = headoffice_df['LDR_BAND'].value_counts()
                by_ldr = pd.DataFrame(ldr).reset_index()
                ldr_index = by_ldr.rename(columns={'index': 'ldr_index'})
                ldr_label = ldr_index['ldr_index'].tolist()
                ldr_data = by_ldr['LDR_BAND'].tolist()

                # IC_WISE_PENDING_CLAIMS_TAT

                ic_wise_pending_claims_tat = pd.crosstab(headoffice_df['ic_name'], headoffice_df['LDR_BAND'])

                ic_wise_pending_claims_tat1 = ic_wise_pending_claims_tat.reset_index()
                ic = ic_wise_pending_claims_tat1['ic_name'].tolist()
                print(ic)
                ten_days_above = ic_wise_pending_claims_tat1['10 Days & Above'].tolist()
                print(ten_days_above)
                zero_to_five_days = ic_wise_pending_claims_tat1['00-05 Days'].tolist()
                print(zero_to_five_days)
                six_to_ten_days = ic_wise_pending_claims_tat1['06-10 Days'].tolist()
                print(six_to_ten_days)

                # TAT_LDR
                TAT_LDR = []
                for a in last_document_recived_date:
                    TAT_LDR.append(EndOfMonth1.date() - a.date() - timedelta(days=1))
                headoffice_df['TAT_LDR'] = TAT_LDR
                headoffice_df['TAT_LDR'] = headoffice_df['TAT_LDR'].dt.days
                # BAND_LDR_TAT

                headoffice_df.loc[(headoffice_df['TAT_LDR'] > 30), 'BAND_LDR_TAT'] = 'Above 1 month'
                headoffice_df.loc[(headoffice_df['TAT_LDR'] <= 30), 'BAND_LDR_TAT'] = 'Below 1 month'

                # TAT_DOA
                DOA = pd.to_datetime(headoffice_df['doa'])
                TAT_DOA = []
                for a in DOA:
                    TAT_DOA.append(EndOfMonth1 - a.replace(tzinfo=None) - timedelta(days=1))
                headoffice_df['TAT_DOA'] = TAT_DOA
                headoffice_df['TAT_DOA'] = headoffice_df['TAT_DOA'].astype('timedelta64[D]').replace([np.inf, -np.inf], 0).astype('Int64')

                # BAND_DOA_TAT
                headoffice_df.loc[(headoffice_df['TAT_DOA'] > 30), 'BAND_DOA_TAT'] = 'Above 1 month'
                headoffice_df.loc[(headoffice_df['BAND_DOA_TAT'] != 'Above 1 month'), 'BAND_DOA_TAT'] = 'Below 1 month'
                BAND_DOA_TAT = headoffice_df['BAND_DOA_TAT']

                # TAT_FIRST_INTIMATION
                headoffice_df['first_intimation_date'] = pd.to_datetime(headoffice_df['first_intimation_date'].dt.date)
                First_Intimation_Date = pd.to_datetime(headoffice_df['first_intimation_date'], format='%y-%m-%d')

                TAT_FIRST_INTIMATION = []
                for i in First_Intimation_Date:
                    TAT_FIRST_INTIMATION.append(EndOfMonth1 - i - timedelta(days=1))
                headoffice_df['TAT_FIRST_INTIMATION'] = TAT_FIRST_INTIMATION
                headoffice_df['TAT_FIRST_INTIMATION'] = headoffice_df['TAT_FIRST_INTIMATION'].astype('timedelta64[D]')

                # YEAR and MONTH_NO

                headoffice_df['year_1'] = pd.DatetimeIndex(headoffice_df['doa']).year
                headoffice_df['month_no'] = pd.DatetimeIndex(headoffice_df['doa']).month

                # BAND_FIRST_INTIMATION
                headoffice_df.loc[
                    (headoffice_df['TAT_FIRST_INTIMATION'] > 90), 'BAND_FIRST_INTIMATION'] = 'Above 3 months'
                headoffice_df.loc[(headoffice_df[
                                       'BAND_FIRST_INTIMATION'] != 'Above 3 months'), 'BAND_FIRST_INTIMATION'] = 'Below 3 months'
                BAND_FIRST_INTIMATION = headoffice_df['BAND_FIRST_INTIMATION']

                # DOA_< 1 MONTH_TILL_CURRENT_MONTH_END
                doaless1_month = headoffice_df.assign(count=(headoffice_df['BAND_DOA_TAT'] == 'Below 1 month')).groupby(
                    'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
                doaless1_month.rename(columns={'count': 'DOA_<_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)

                # DOA_> 1 MONTH_TILL_CURRENT_MONTH_END
                doagreat1_month = headoffice_df.assign(
                    count=(headoffice_df['BAND_DOA_TAT'] == 'Above 1 month')).groupby('revised_servicing_branch',
                                                                                      sort=False, as_index=False).agg(
                    {'count': sum})
                doagreat1_month.rename(columns={'count': 'DOA_>_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)

                # LDR_<_1 MONTH_TILL_CURRENT_MONTH_END
                ldrless1_month = headoffice_df.assign(count=(headoffice_df['BAND_LDR_TAT'] == 'Below 1 month')).groupby(
                    'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
                ldrless1_month.rename(columns={'count': 'LDR_<_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)

                # LDR_>_1 MONTH_TILL_CURRENT_MONTH_END
                ldrgreat1_month = headoffice_df.assign(
                    count=(headoffice_df['BAND_LDR_TAT'] == 'Above 1 month')).groupby('revised_servicing_branch',
                                                                                      sort=False, as_index=False).agg(
                    {'count': sum})
                ldrgreat1_month.rename(columns={'count': 'LDR_>_1_MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)

                # FI_DATE_<3MONTH_TILL_CURRENT_MONTH_END

                fidateless3_month = headoffice_df.assign(
                    count=(headoffice_df['BAND_FIRST_INTIMATION'] == 'Below 3 months')).groupby(
                    'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
                fidateless3_month.rename(columns={'count': 'FI_DATE_<_3MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)

                # FI_DATE_>3MONTH_TILL_CURRENT_MONTH_END

                fidategreat3_month = headoffice_df.assign(
                    count=(headoffice_df['BAND_FIRST_INTIMATION'] == 'Above 3 months')).groupby(
                    'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum})
                fidategreat3_month.rename(columns={'count': 'FI_DATE_>_3MONTH_TILL_CURRENT_MONTH_END'}, inplace=True)

                # OS Claims

                osclaims = pd.DataFrame(headoffice_df.assign(
                    count=(headoffice_df['revised_servicing_branch']).isin(hodept_master['branch_name'])).groupby(
                    'revised_servicing_branch', sort=False, as_index=False).agg({'count': sum}))
                osclaims.rename(columns={'count': 'OS_CLAIMS'}, inplace=True)

                # %OS CLAIMS

                osclaims['per_os_claims'] = ((osclaims['OS_CLAIMS'] / sum(osclaims['OS_CLAIMS'])) * 100).round(2)

                # % OS CLAIMS > 1_MONTH_DOA
                doagreat1_month['per_os_claims > 1_MONTH_DOA'] = (doagreat1_month[
                                                                      'DOA_>_1_MONTH_TILL_CURRENT_MONTH_END'] /
                                                                  osclaims['OS_CLAIMS']) * 100
                osonemonth_doa = doagreat1_month['per_os_claims > 1_MONTH_DOA']
                osonemonth_doaact = osonemonth_doa.to_string(index=False)

                # % OS CLAIMS > 1_MONTH_LDR

                ldrgreat1_month['per_OS_claims > 1_MONTH_LDR'] = ((ldrgreat1_month[
                                                                       'LDR_>_1_MONTH_TILL_CURRENT_MONTH_END'] /
                                                                   osclaims['OS_CLAIMS']) * 100).round(2)

                per_os_one_month_ldr = ldrgreat1_month['per_OS_claims > 1_MONTH_LDR']
                per_os_one_month_ldract = per_os_one_month_ldr.to_string(index=False)

                # % OS CLAIMS > 3_MONTHS_FIRST_INTIMATION

                fidategreat3_month['per_OS_claims > 3_MONTHS_FIRST_INTIMATION'] = ((fidategreat3_month[
                                                                                        'FI_DATE_>_3MONTH_TILL_CURRENT_MONTH_END'] /
                                                                                    osclaims['OS_CLAIMS']) * 100).round(
                    2)
                os_three_month_fi = fidategreat3_month['per_OS_claims > 3_MONTHS_FIRST_INTIMATION']
                os_three_month_fiact = os_three_month_fi.to_string(index=False)

                # Merging
                BRANCH = pd.merge(osclaims, hodept_master, left_on='revised_servicing_branch',
                                  right_on='branch_name', how='inner')

                # Concating Results

                BRANCH_UPDATED = pd.concat(
                    [BRANCH, doaless1_month, doagreat1_month, ldrless1_month, ldrgreat1_month, fidateless3_month,
                     fidategreat3_month], axis=1)

                # Removing Duplicates

                BRANCH_UPDATED = BRANCH_UPDATED.T.drop_duplicates().T
                BRANCH_UPDATED1_fig = go.Figure(
                    data=[go.Table(columnwidth=[40, 20, 20],

                                   header=dict(values=list(BRANCH_UPDATED.columns),
                                               fill_color='#094780',
                                               line_color='darkslategray',
                                               align='left',
                                               font = dict(size = 12, color='white')),
                                   cells=dict(values=BRANCH_UPDATED.T,
                                              fill_color='white',
                                              line_color='darkslategray',
                                              align='left'))
                          ])
                BRANCH_UPDATED1_fig.layout.width = 1200

                BRANCH_UPDATED1 = plot(BRANCH_UPDATED1_fig,output_type='div', config=config)


                BRANCH1 = headoffice_df.merge(hodept_master[['branch_group_type', 'branch_name']], how='left',
                                              left_on='servicing_branch', right_on='branch_name')

                BRANCH_FINAL = pd.merge(BRANCH1, BRANCH_UPDATED, left_on='revised_servicing_branch',
                                        right_on='revised_servicing_branch', how='inner')

                BRANCHES_OUTSTANDING_STATS = BRANCH_UPDATED.groupby(['revised_servicing_branch'])['per_OS_claims > 3_MONTHS_FIRST_INTIMATION', 'per_OS_claims > 1_MONTH_LDR', 'per_os_claims > 1_MONTH_DOA'].agg('sum')
                MDI_BRANCHES_OUTSTANDING_STATS = BRANCHES_OUTSTANDING_STATS.reset_index()
                # create the trace for each value column
                trace1 = go.Bar(
                    x=MDI_BRANCHES_OUTSTANDING_STATS['revised_servicing_branch'],
                    y=MDI_BRANCHES_OUTSTANDING_STATS['per_OS_claims > 3_MONTHS_FIRST_INTIMATION'],
                    name='% OS CLAIMS > 3_MONTHS_FIRST_INTIMATION'
                )
                trace2 = go.Bar(
                    x=MDI_BRANCHES_OUTSTANDING_STATS['revised_servicing_branch'],
                    y=MDI_BRANCHES_OUTSTANDING_STATS['per_OS_claims > 1_MONTH_LDR'],
                    name='% OS CLAIMS > 1_MONTH_LDR'
                )
                trace3 = go.Bar(
                    x=MDI_BRANCHES_OUTSTANDING_STATS['revised_servicing_branch'],
                    y=MDI_BRANCHES_OUTSTANDING_STATS['per_os_claims > 1_MONTH_DOA'],
                    name='% OS CLAIMS > 1_MONTH_DOA'
                )

                # create the layout
                layout = go.Layout(
                    width=550,
                    title='MDI BRANCHES OUTSTANDING STATS',
                    xaxis=dict(title='SERVICING BRANCH'),
                    yaxis=dict(title=''),
                    barmode='group',
                    legend=dict(title_font_family="Times New Roman",
                                font=dict(size=10))
                )


                # create the figure and plot it
                MDI_BRANCHES_OUTSTANDING_STATS2 = go.Figure(data=[trace1, trace2, trace3], layout=layout)
                MDI_BRANCHES_OUTSTANDING_STATS1 = plot(MDI_BRANCHES_OUTSTANDING_STATS2,output_type='div', config=config)






                # tat_ldr_avg

                tat_ldr_avg = BRANCH_FINAL.groupby(['sub_head'])['TAT_LDR'].agg('mean').round(2)
                tat_ldr_avg = tat_ldr_avg.reset_index()
                tat_ldr_avg.rename(columns={'TAT_LDR': 'TAT_LDR_AVG'}, inplace=True)

                # liablity_reserve_amt

                liablity_reserve_amt1 = (BRANCH_FINAL.groupby(['sub_head'])['liablity_reserve_amt'].agg('sum'))

                liablity_reserve_amt3 = liablity_reserve_amt1.reset_index()
                liablity_reserve_amt2 = liablity_reserve_amt3['liablity_reserve_amt'].apply(np.ceil)
                liablity_reserve_amt4 = pd.DataFrame(liablity_reserve_amt2)
                liablity_reserve_amt5 = liablity_reserve_amt4.style.format({
                    "liablity_reserve_amt": "{:,d}"})
                liablity_reserve_amt = liablity_reserve_amt5

                # OS_CLAIMS_BY_SubHead

                os_claims_by_sub_head = BRANCH_FINAL.groupby(['sub_head'])['OS_CLAIMS'].agg('count')
                os_claims_by_sub_head = os_claims_by_sub_head.reset_index()

                per_os_claims_by_subhead = BRANCH_FINAL.groupby('sub_head')['OS_CLAIMS'].count().rename(
                    "per_OS_claims").transform(lambda x: x / x.sum() * 100)

                per_os_claims_by_subhead = per_os_claims_by_subhead.reset_index().round(2)

                # Per_Liablity_Reserve_Amt

                per_liablity_reserve_amt = BRANCH_FINAL.groupby(['sub_head'])['liablity_reserve_amt'].sum().rename(
                    "per_liablity_reserve_amts").transform(lambda x: x / x.sum() * 100)

                per_liablity_reserve_amt1 = per_liablity_reserve_amt.reset_index()
                per_liablity_reserve_amt1 = per_liablity_reserve_amt1['per_liablity_reserve_amts'].apply(np.ceil)

                no_of_claims_vis_a_vis_outstanding_reasons = pd.concat(
                    [os_claims_by_sub_head, tat_ldr_avg, per_os_claims_by_subhead, liablity_reserve_amt2,
                     per_liablity_reserve_amt1], axis=1)
                no_of_claims_vis_a_vis_outstanding_reasons = no_of_claims_vis_a_vis_outstanding_reasons.T.drop_duplicates().T

                print(no_of_claims_vis_a_vis_outstanding_reasons)

                no_of_claims_vis_a_vis_outstanding_reasons_fig = go.Figure(
                    data=[go.Table(columnwidth=[150, 70, 90, 90, 110, 130],

                                   header=dict(values=list(no_of_claims_vis_a_vis_outstanding_reasons.columns),
                                               fill_color='#094780',
                                               line_color='darkslategray',
                                               align='left',
                                               font = dict(size = 12, color='white')),
                                   cells=dict(values=no_of_claims_vis_a_vis_outstanding_reasons.T,
                                              fill_color='white',
                                              line_color='darkslategray',
                                              align='left'))
                          ])

                no_of_claims_vis_a_vis_outstanding_reasons_fig.layout.width = 1200
                no_of_claims_vis_a_vis_outstanding_reasons_fig.update_layout(height=len(no_of_claims_vis_a_vis_outstanding_reasons) * 40)
                no_of_claims_vis_a_vis_outstanding_reasons_fig.update_traces(cells_font=dict(size=10))

                no_of_claims_vis_a_vis_outstanding_reasons1 = plot(no_of_claims_vis_a_vis_outstanding_reasons_fig, output_type='div', config=config)

                # ic_wise_outstanding_reasons

                ic_wise_outstanding_reasons = pd.crosstab(headoffice_df['ic_name'], headoffice_df['sub_head'])
                ic_wise_outstanding_reasons = ic_wise_outstanding_reasons.reset_index()

                ic_wise_outstanding_reasons_fig = go.Figure(
                    data=[go.Table(columnwidth=[150, 70, 90, 90, 110, 130],

                                   header=dict(values=list(ic_wise_outstanding_reasons.columns),
                                               fill_color='#094780',
                                               line_color='darkslategray',
                                               align='left',
                                               font=dict(size=12, color='white')
                                               ),
                                   cells=dict(values=ic_wise_outstanding_reasons.T,
                                              fill_color='white',
                                              line_color='darkslategray',
                                              align='left'))
                          ])

                ic_wise_outstanding_reasons_fig.layout.width = 1200
                ic_wise_outstanding_reasons_fig.update_layout(height=len(ic_wise_outstanding_reasons) * 80)
                ic_wise_outstanding_reasons_fig.update_traces(cells_font=dict(size=10))

                ic_wise_outstanding_reasons1 = plot(ic_wise_outstanding_reasons_fig, output_type='div', config=config)


                # branch_wise_adr_count_and_avg_tat

                branch_wise_adr_count_and_avg_tat = BRANCH_FINAL.groupby(['revised_servicing_branch'])['TAT_LDR'].agg(
                    'mean')
                branch_wise_adr_count_and_avg_tat2 = branch_wise_adr_count_and_avg_tat.round(2)
                branch_wise_adr_count_and_avg_tat3 = branch_wise_adr_count_and_avg_tat2.reset_index()
                branch_wise_adr_count_and_avg_tat4 = branch_wise_adr_count_and_avg_tat3.rename(
                    columns={'index': 'branch_wise_adr_count_and_avg_tat4'})
                tat_ldr = branch_wise_adr_count_and_avg_tat4['TAT_LDR']
                tat_ldr = tat_ldr[0]

                branch_wise_adr_count_and_avg_tat1 = tat_ldr.tolist()

                # reason_wise_corporate_retail_os
                reason_wise_corporate_retail_os = pd.crosstab(headoffice_df['sub_head'], headoffice_df['corp_retail'])
                reason_wise_corporate_retail_os = reason_wise_corporate_retail_os.reset_index()

                reason_wise_corporate_retail_os_fig = go.Figure(
                    data=[go.Table(columnwidth=[40, 20, 20],

                                   header=dict(values=list(reason_wise_corporate_retail_os.columns),
                                               fill_color='#094780',
                                               line_color='darkslategray',
                                               align='left',
                                               font = dict(size = 12, color='white')),
                                   cells=dict(values=reason_wise_corporate_retail_os.T,
                                              fill_color='white',
                                              line_color='darkslategray',
                                              align='left'))
                          ])

                # reason_wise_corporate_retail_os_fig.layout.width = 1200
                reason_wise_corporate_retail_os_fig.update_layout(height=len(reason_wise_corporate_retail_os) * 50)
                reason_wise_corporate_retail_os_fig.update_traces(cells_font=dict(size=10))

                reason_wise_corporate_retail_os1 = plot(reason_wise_corporate_retail_os_fig, output_type='div', config=config)


            else:
                messages.warning(request, "Apparently no values available...")

        mydict = {
            'form': form,
            'oscont_ho': oscount_ho,
            'os_claims_by_actuallosstype_fig1':os_claims_by_actuallosstype_fig1,
            'os_claims_by_head1':os_claims_by_head1,
            'MDI_BRANCHES_OUTSTANDING_STATS1':MDI_BRANCHES_OUTSTANDING_STATS1,
            'BRANCH_UPDATED1': BRANCH_UPDATED1,
            'no_of_claims_vis_a_vis_outstanding_reasons1': no_of_claims_vis_a_vis_outstanding_reasons1,
            'ic_wise_outstanding_reasons': ic_wise_outstanding_reasons1,

            'reason_wise_corporate_retail_os': reason_wise_corporate_retail_os1,
            'branch_wise_adr_count_and_avg_tat': branch_wise_adr_count_and_avg_tat1,
            'os_three_month_fiact': os_three_month_fiact,
            'per_os_one_month_ldract': per_os_one_month_ldract,
            'osonemonth_doaact': osonemonth_doaact,
            'ldr_data': ldr_data,
            'ldr_label': ldr_label,
            'ten_days_above': ten_days_above,
            'zero_to_five_days': zero_to_five_days,
            'six_to_ten_days': six_to_ten_days,
            'ic': ic,
    }
    return render(request, 'Management/headoffice.html', context=mydict)




def corporate(request):
    if request.user.is_authenticated:
        insured_name1 = None
        premium = None
        lives = None
        paid_claim_count = None
        out_claim_count = None
        crs_claim_count = None
        lives_by_ip_relation_code_fig_1 = None
        age_band_relationship_wise_insured_lives_1 = None
        age_band_relationship_wise_insured_lives_fig2 = None
        age_band_relationship_wise_insured_lives_fig1 = None
        ALL_CLAIMS_BY_CLAIM_STATUS_fig = None
        lives_by_age_band_fig1 = None
        summary1 = None
        relation_age_band_wise_paid_amt_fig_1 = None
        relation_age_band_wise_paid_amt_fig_2 = None
        outstanding_claim_analysis_fig = None
        outstanding_claim_analysis_tb = None
        claim_status_fig1 = None
        rejected_claim_breakup_fig = None
        rejected_claim_breakup_tb = None
        treatment_type_wise_analysis_fig = None
        treatment_type_wise_analysis_tb = None
        cashless_vs_reimbersement_claim_count_paid_fig = None
        cashless_vs_reimbersement_claim_amt_paid_fig = None
        cashless_vs_reimbersement_lodge_amount_fig = None
        Summary_plot = None
        Summary_plot2 = None
        ipd_opd_summary_fig = None
        top10_ailment_wise_analysis_fig1 = None
        top10_ailment_wise_analysis_fig2 = None
        top10_hospital_wise_paidamt_fig1 = None
        top10_hospital_wise_paidamt_fig2 = None
        Age_Band_IR = None
        SI_Band_IR = None
        Amount_Band_IR = None
        paidamt_band_analysis1 = None
        paidamt_band_analysis_fig = None
        ipd_opd_wise_claims_fig = None
        disease_wise_buffer_amount_fig2 = None
        buffer_amt_vs_liab_amt = None
        customer_touch_point_fig2 = None
        customer_touch_point_fig1 = None
        customer_call_analysis_fig2 = None
        customer_call_analysis_fig1 = None
        ReasonForCall_analysis_fig2 = None
        ReasonForCall_analysis_fig1 = None
        griveance_analysis_fig2 = None
        griveance_analysis_fig1 = None
        form = SelectCorporate(request.POST or None)
        if request.method == 'POST':
            pol_no = request.POST.get('polno')

            paid_qs_corp = ClPaidDet2.objects.prefetch_related('pol_no').filter(pol_no=pol_no).values('pol_no','sla_heading','insuredname', 'relation','ccn', 'lodgetype','status','actuallosstype','actual_lodge_amt','settledamt','liablityamt','consider_count','ipd_opd', 'age_band_rev','new_disease_category', 'lodgedate','dod', 'lodgeamt','treatmenttype', 'sumins', 'amount_band','diseasecategory', 'hospitlname', 'utilizationband', 'buffer_amt', 'payment_tat', 'end_to_end_tat', 'final_processing_tat_settlment_tat','adr_raise_date_lodge_date', 'adr_recive_adr_raise_date')


            #paid_crop_df = ClPaidDet2[['insuredname', 'relation', 'lodgetype','status','actuallosstype','actual_lodge_amt','settledamt','liablityamt','consider_count','ipd_opd', 'age_band_rev','new_disease_category', 'lodgedate','dod', 'lodgeamt','treatmenttype', 'sumins', 'amount_band','diseasecategory', 'hospitlname', 'utilizationband', 'buffer_amt', 'payment_tat', 'end_to_end_tat', 'final_processing_tat_settlment_tat','adr_raise_date_lodge_date', 'adr_recive_adr_raise_date']]



            crs_qs = ClCrsDet2.objects.prefetch_related('pol_no').filter(pol_no=pol_no).values('status','substatus','ccn','lodgedate','lodgetype','cl_lod_amt','doa','dod','sla_heading','consider_count','actuallosstype','age_band_rev','ipd_opd','sumins','actual_lodge_amt')
            out_qs = ClOutDet2.objects.prefetch_related('pol_no').filter(pol_no=pol_no).values('status','extra','head','sla_heading','consider_count','age_band_rev','liablityamt','actuallosstype','ccn','lodgetype','lodgedate','doa','dod','ipd_opd','sumins','actual_lodge_amt','actuallodgeamt','sla_heading_updated')
            customercare_qs = Customercare.objects.filter(pol_no=pol_no)
            relation_qs = RelationMaster.objects.all()
            # pol_no1 = tuple(pol_no)


            Driver = 'SQL Server'
            Server = r'MDINETPROJECTS\Analytics'
            Database = 'Enrollment'
            UID = 'mdianalytics'
            PWD = 'mdianalytics@123'
            Database_Connection = f'mssql://{UID}:{PWD}@{Server}/{Database}?driver={Driver}'
            connection = pyodbc.connect(driver='{SQL Server}', host=Server, database=Database,
                                        user=UID, password=PWD)
            portal_premium_qs = pd.read_sql_query(f"""SELECT
                            A.pol_no,A.name_of_insured, A.ic_name, ip_relation_code, ip_age, broker_name, risk_from_date,risk_expiry_date,
                            CASE WHEN RISK_EXPIRY_DATE<=GETDATE() THEN '365'
                            ELSE CASE WHEN  RISK_EXPIRY_DATE>GETDATE() THEN DATEDIFF(DAY,RISK_FROM_DATE,GETDATE())+1 ELSE 0 END
                            END AS [POLICYRUNDAY],
                            CONVERT(VARCHAR(10),GETDATE(),103)+' (TIME: '+SUBSTRING(CAST(GETDATE()AS VARCHAR(30)),13,7)+')' AS REPORT_DATE,
                            ISNULL(A.NET_PREMIUM,0) AS NET_PREMIUM,
                            ISNULL(B.PREMIUM_ENDORSEMENT,0) AS PREMIUM_ENDORSEMENT,
                            CASE WHEN ISNULL(RISK_EXPIRY_DATE,'')<=ISNULL(GETDATE(),'') THEN  ( ISNULL(A.NET_PREMIUM,0)+ISNULL(B.PREMIUM_ENDORSEMENT,0))
                            ELSE ((ISNULL(A.NET_PREMIUM,0)+ISNULL(B.PREMIUM_ENDORSEMENT,0))/365)*DATEDIFF(DAY,RISK_FROM_DATE,GETDATE())+1 END AS [EARNED_PERMIUM],
                            ISNULL(A.NO_OF_EMPLOYEES_COVERED,0) AS NO_OF_EMPLOYEES_COVERED,
                            ISNULL(A.NO_OF_DEPENDANTS_COVERED,0) AS NO_OF_DEPENDANTS_COVERED,
            	            NO_OF_EMPLOYEES_COVERED + NO_OF_DEPENDANTS_COVERED AS lives,NET_PREMIUM + CAST(PREMIUM_ENDORSEMENT AS int) As Premium
                            FROM (SELECT IC_NAME,TEMP.POL_NO, IP_Relation_Code, IP_Age, Broker_Name,NAME_OF_INSURED,

                                MIN(TEMP.RISK_FROM_DATE) AS RISK_FROM_DATE,
                                MAX(TEMP.RISK_EXPIRY_DATE) AS RISK_EXPIRY_DATE,MAX(TEMP.NET_PREMIUM) AS NET_PREMIUM,
                                SUM(NO_OF_EMPLOYEES_COVERED) AS NO_OF_EMPLOYEES_COVERED, SUM(NO_OF_DEPENDANTS_COVERED) AS NO_OF_DEPENDANTS_COVERED
                                FROM
                                (SELECT IC_NAME,POL_NO, IP_Relation_Code, IP_Age, Broker_Name, NAME_OF_INSURED,
                                    MIN(RISK_FROM_DATE) AS RISK_FROM_DATE,
                                    MAX(RISK_EXPIRY_DATE) AS RISK_EXPIRY_DATE,
                                    MAX(PREMIUM) AS NET_PREMIUM,
                                    SUM(CASE WHEN ((IP_RELATION_CODE = 'SELF' OR IP_RELATION_CODE = 'EMPLOYEE')
                                    AND ISNULL(IP_CANCEL_DATE,'')='' ) THEN 1 ELSE 0 END) AS NO_OF_EMPLOYEES_COVERED,
                                    SUM(CASE WHEN ISNULL(IP_CANCEL_DATE,'')=''  THEN 1 ELSE 0 END ) -
                                    SUM(CASE WHEN (IP_RELATION_CODE IN ('SELF','EMPLOYEE')
                                    AND ISNULL(IP_CANCEL_DATE,'')='')  THEN 1 ELSE 0 END)  AS NO_OF_DEPENDANTS_COVERED
                                    FROM ENROLLMENT_MASTER WHERE Pol_No = '{pol_no}' AND Pol_Status = 'ENFORCED'
                                    GROUP BY POL_NO,IC_NAME, Broker_Name,NAME_OF_INSURED, IP_Relation_Code, IP_Age)
                                    AS TEMP GROUP BY TEMP.POL_NO,IC_NAME, Broker_Name,NAME_OF_INSURED, IP_Relation_Code, IP_Age)
                                    AS A LEFT OUTER JOIN(SELECT POL_NO,(SUM(ISNULL(AMTADD,0)) - SUM(ISNULL(AMTRED,0)) ) AS PREMIUM_ENDORSEMENT FROM DBO.[ENDORS]
                                    WHERE Pol_No = '{pol_no}'
                                    GROUP BY POL_NO) AS B ON A.Pol_No = B.Pol_No """, connection)

            insured_name1 = portal_premium_qs['name_of_insured'].unique().tolist()
            # portal_premium_qs = PremiumLivesCorporate.objects.filter(Q(pol_no=pol_no))
            if ((len(paid_qs_corp) > 0) and (len(out_qs) > 0) and (len(crs_qs) > 0)):
                paid_df = pd.DataFrame(paid_qs_corp)
                crs_df = pd.DataFrame(crs_qs)
                out_df = pd.DataFrame(out_qs)
                portal_premium_df = pd.DataFrame(portal_premium_qs)
                relation_df = pd.DataFrame(relation_qs.values())

                paid_df['settledamt'] = paid_df['settledamt'].astype("int")

                # format_comma_separated
                def format_comma_separated(number):
                    return "{:,}".format(number)

                out_df = out_df[out_df['sla_heading_updated'] == out_df['sla_heading_updated'].max()]
                print(out_df['sla_heading_updated'].unique())

                config = {'responsive': True, 'displaylogo': False}
                # portal_premium_df = pd.DataFrame(portal_premium_qs.values())
                # enroll_df = pd.DataFrame(enroll_qs.values())
                # endors_qs = pd.DataFrame(enroll_qs.values())
                customercare_df = pd.DataFrame(customercare_qs.values())
                paid_df['status'] = paid_df['status'].str.strip()
                out_df['status'] = out_df['status'].str.strip()
                crs_df['status'] = crs_df['status'].str.strip()
                out_df['extra'] = out_df['extra'].str.strip()
                out_df['head'] = out_df['head'].str.strip()
                crs_df['substatus'] = crs_df['substatus'].str.strip()
                portal_premium_df = pd.merge(portal_premium_df, relation_df, how='left', left_on='ip_relation_code',
                                             right_on='relation').drop(columns=['id', 'relation'])


                done = []
                not_done = []
                for col in portal_premium_df.columns:
                    try:
                        portal_premium_df[col] = portal_premium_df[col].str.strip()
                        done.append(col)
                    except:
                        not_done.append(col)

                print(portal_premium_df['Premium'])
                # portal_premium_df['Premium'] = portal_premium_df['NET_PREMIUM'].fillna(0) + portal_premium_df[
                #     'PREMIUM_ENDORSEMENT'].fillna(0)
                relation_df['relation'] = relation_df['relation'].str.strip()
                # Count for all Paid, Crs, Outstanding ,lives, premium
                paid_df['concat'] = paid_df['ccn'] + "-" + paid_df['lodgetype'] + "-" +paid_df['lodgedate'].astype('str')
                out_df['concat'] = out_df['ccn'] + "-" + out_df['lodgetype'] + "-" +out_df['lodgedate'].astype('str')
                crs_df['concat'] = crs_df['ccn'] + "-" + crs_df['lodgetype'] + "-" +crs_df['lodgedate'].astype('str')


                paid_claim_count = paid_df['concat'].nunique()
                out_claim_count = out_df['concat'].nunique()
                crs_claim_count = crs_df['concat'].nunique()

                portal_premium_df['Premium'].replace(np.NaN,0, inplace=True)
                # Check if there are any valid values in the 'premium' column
                if portal_premium_df['Premium'].notna().any():
                    # Calculate the mean premium as an integer
                    premium = int(np.round(portal_premium_df['Premium'].mean()))
                else:
                    # Set a default value or handle the case when there are no valid values
                    premium = 0  # Replace with your desired default value or handling logic

                print('premium', premium)
                lives = round((portal_premium_df['lives'].sum()), 0)

                # Set the locale to Indian English
                locale.setlocale(locale.LC_NUMERIC, 'en_IN')



                # Relationship Wise lives
                relation_wise_lives_covered = portal_premium_df.groupby(['std_relation'])['lives'].agg(
                    'sum').reset_index()

                # # Format the 'lives' column with Indian style comma separation
                # relation_wise_lives_covered['lives'] = relation_wise_lives_covered['lives'].apply(format_comma_separated)

                data = [go.Pie(labels=relation_wise_lives_covered['std_relation'],
                               values=relation_wise_lives_covered['lives'], text=relation_wise_lives_covered['lives'],
                               hole=.5, marker_colors=px.colors.qualitative.Plotly)
                        ]
                lives_by_ip_relation_code_fig = go.Figure(data=data)
                lives_by_ip_relation_code_fig.update_layout(margin=dict(t=80, b=80, l=80, r=80),
                                                            plot_bgcolor="rgba(0,0,0,0)",
                                                            title='')

                lives_by_ip_relation_code_fig_1 = plot(lives_by_ip_relation_code_fig, output_type='div', config=config)

                portal_premium_df['ip_age'] = portal_premium_df['ip_age'].astype(int)

                conditions = [
                    (portal_premium_df['ip_age'] <= 10),
                    (portal_premium_df['ip_age'] >= 11) & (portal_premium_df['ip_age'] <= 20),
                    (portal_premium_df['ip_age'] >= 21) & (portal_premium_df['ip_age'] <= 30),
                    (portal_premium_df['ip_age'] >= 31) & (portal_premium_df['ip_age'] <= 40),
                    (portal_premium_df['ip_age'] >= 41) & (portal_premium_df['ip_age'] <= 50),
                    (portal_premium_df['ip_age'] >= 51) & (portal_premium_df['ip_age'] <= 60),
                    (portal_premium_df['ip_age'] >= 61) & (portal_premium_df['ip_age'] <= 70),
                    (portal_premium_df['ip_age'] >= 71) & (portal_premium_df['ip_age'] <= 80),
                    (portal_premium_df['ip_age'] > 80)
                ]

                # create a list of the values we want to assign for each condition
                values = ['00-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', 'Above 80']

                # create a new column and use np.select to assign values to it using our lists as arguments
                portal_premium_df['age_band'] = np.select(conditions, values)

                crs_df['status'] = np.where(
                    (crs_df['status'].str.strip() == 'Closed') | (crs_df['status'].str.strip() == 'Repudiated'), 'CRS',
                    crs_df['status'])

                import random

                clrs = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black',
                        'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
                        'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
                        'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki', 'darkmagenta',
                        'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
                        'darkslateblue', 'darkslategray', 'darkslategrey',
                        'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue',
                        'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold',
                        'goldenrod', 'gray', 'grey', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred',
                        'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon',
                        'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgrey',
                        'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
                        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta',
                        'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple',
                        'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred',
                        'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
                        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
                        'palevioletred', 'papayawhip',
                        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue',
                        'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue',
                        'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan',
                        'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow',
                        'yellowgreen']

                clrs = random.sample(clrs, 20)

                # AgeBand wise relationship and lives
                age_band_wise_lives_covered = portal_premium_df.groupby(['age_band'])['lives'].agg('sum').reset_index()
                data = [go.Pie(labels=age_band_wise_lives_covered['age_band'],
                               values=age_band_wise_lives_covered['lives'], text=age_band_wise_lives_covered['lives'],
                               hole=.5, marker_colors=px.colors.qualitative.Plotly)
                        ]
                lives_by_age_band_fig = go.Figure(data=data)
                lives_by_age_band_fig.update_layout(margin=dict(t=100, b=100, l=100, r=100),
                                                    plot_bgcolor="rgba(0,0,0,0)",
                                                    title='')
                lives_by_age_band_fig.update_traces(rotation=90)
                lives_by_age_band_fig.update_layout(title_x=0.5)

                lives_by_age_band_fig1 = plot(lives_by_age_band_fig, output_type='div', config=config)

                age_band_relationship_wise_insured_lives = portal_premium_df.groupby(['age_band', 'std_relation'])[
                    'lives'].agg('sum').reset_index()
                age_band_relationship_wise_insured_lives = age_band_relationship_wise_insured_lives[
                    (age_band_relationship_wise_insured_lives['std_relation'] != 'Parent in Law')]

                age_band_relationship_wise_insured_lives_1 = pd.pivot_table(age_band_relationship_wise_insured_lives,
                                                                            values='lives', index=['age_band'],
                                                                            columns=['std_relation'],
                                                                            aggfunc='sum').replace(np.nan, 0)

                age_band_relationship_wise_insured_lives_1_table = age_band_relationship_wise_insured_lives_1.reset_index()

                table_layout = go.Layout(
                    margin=dict(l=0, r=0, t=0, b=0))



                # figure for plotly
                age_band_relationship_wise_insured_lives_fig1 = go.Figure(data=[go.Table(
                    header=dict(values=age_band_relationship_wise_insured_lives_1_table.columns,
                                fill_color='#094780',
                                line_color='darkslategray',
                                align='left',
                                font=dict(size=10, color='white')),
                    cells=dict(values=age_band_relationship_wise_insured_lives_1_table.T,
                               fill_color='white',
                               line_color='darkslategray',
                               align='left'))

                ], layout=table_layout)

                age_band_relationship_wise_insured_lives_fig2 = px.bar(age_band_relationship_wise_insured_lives,
                                                                       x="std_relation", y="lives", text_auto=True,
                                                                       color_discrete_sequence=px.colors.sequential.RdBu,
                                                                       color="age_band", barmode='stack')

                age_band_relationship_wise_insured_lives_fig2.update_layout(title='')

                age_band_relationship_wise_insured_lives_fig2.update_layout(
                    xaxis_title=None,
                    yaxis_title="No. of Lives"
                )

                age_band_relationship_wise_insured_lives_fig2.update_layout(title_x=0.5, plot_bgcolor = 'white')

                age_band_relationship_wise_insured_lives_fig2 = plot(age_band_relationship_wise_insured_lives_fig2,
                                                                     output_type='div', config=config)

                age_band_relationship_wise_insured_lives_fig1 = plot(age_band_relationship_wise_insured_lives_fig1,
                                                                     output_type='div', config=config)

                # claims status
                paid_sub_status = paid_df['status'].value_counts().reset_index()
                paid_sub_status = pd.DataFrame(paid_sub_status.values, columns=['status', 'claim_count'])

                out_sub_status = out_df['head'].str.strip()[
                    out_df['consider_count'] == 1].value_counts().reset_index()
                out_sub_status = pd.DataFrame(out_sub_status.values, columns=['status', 'claim_count'])

                crs_sub_status = crs_df['substatus'][crs_df['consider_count'] == 1].value_counts().reset_index()
                crs_sub_status = pd.DataFrame(crs_sub_status.values, columns=['status', 'claim_count'])

                claim_status = pd.concat([paid_sub_status, out_sub_status, crs_sub_status], axis=0)
                claim_status = claim_status.sort_values('claim_count',ascending=True)  # Sort by 'claim_count' in descending order

                claim_status_data = [
                    go.Bar(
                        y=claim_status['status'],  # Use 'status' as the y-axis values
                        x=claim_status['claim_count'],  # Use 'claim_count' as the x-axis values
                        text=claim_status['claim_count'],
                        orientation='h',  # Set the orientation to horizontal
                        marker=dict(color=px.colors.qualitative.Plotly)  # Set the marker color
                    )
                ]

                claim_status_fig = go.Figure(data=claim_status_data)
                claim_status_fig.update_layout(
                    margin=dict(t=100, b=100, l=100, r=100),
                    title='<b><b>',  # Add your desired title here
                    title_x=0.5
                )


                claim_status_fig1 = plot(claim_status_fig, output_type='div', config=config)

                # All Claims By Claim Status

                paid_count = claim_status[claim_status['status'] == 'Paid']
                out_count = out_df['status'][out_df['consider_count'] == 1].value_counts().reset_index()
                out_count = pd.DataFrame(out_count.values, columns=['status', 'claim_count'])
                crs_count = crs_df['status'][crs_df['consider_count'] == 1].value_counts().reset_index()
                crs_count = pd.DataFrame(crs_count.values, columns=['status', 'claim_count'])

                ALL_CLAIMS_BY_CLAIM_STATUS = pd.concat([paid_count, out_count, crs_count])

                data = [
                    go.Pie(labels=ALL_CLAIMS_BY_CLAIM_STATUS['status'],
                           values=ALL_CLAIMS_BY_CLAIM_STATUS['claim_count'],
                           text=ALL_CLAIMS_BY_CLAIM_STATUS['claim_count'],
                           hole=.5, marker_colors=px.colors.qualitative.Plotly)
                ]
                ALL_CLAIMS_BY_CLAIM_STATUS_fig = go.Figure(data=data)
                ALL_CLAIMS_BY_CLAIM_STATUS_fig.update_layout(margin=dict(t=100, b=100, l=100, r=100),
                                                             title='')

                ALL_CLAIMS_BY_CLAIM_STATUS_fig.update_layout(title_x=0.5,plot_bgcolor = 'white')

                ALL_CLAIMS_BY_CLAIM_STATUS_fig = plot(ALL_CLAIMS_BY_CLAIM_STATUS_fig, output_type='div')

                # All Claims Lodge Amt by Claim Status

                paid_lodge_amt = paid_df.groupby(['status']).agg({'actual_lodge_amt': sum}).reset_index()
                out_lodge_amt = out_df.groupby(['status']).agg({'actual_lodge_amt': sum}).reset_index()
                crs_lodge_amt = crs_df.groupby(['status']).agg({'actual_lodge_amt': sum}).reset_index()
                ALL_CLAIMS_LODGE_AMT_BY_CLAIM_STATUS = pd.concat([paid_lodge_amt, out_lodge_amt, crs_lodge_amt])

                # AGE BAND & RELATION WISE PAID CLAIM ANALYSIS

                Paid_Data_Relation = pd.merge(paid_df, relation_df, how='left', left_on='relation',
                                              right_on='relation').drop(columns=['relation'])



                relation_age_band_wise_paid_amt = pd.pivot_table(Paid_Data_Relation, index='age_band_rev',
                                                                 columns=('std_relation'), values=('settledamt'),
                                                                 aggfunc='sum', margins='True', margins_name='Total')

                relation_age_band_wise_paid_amt.drop(["Total"], inplace=True)

                relation_age_band_wise_paid_amt['Total'] = relation_age_band_wise_paid_amt['Total'].astype(int)

                relation_age_band_wise_paid_amt['Total%'] = round(
                    (relation_age_band_wise_paid_amt['Total'] / relation_age_band_wise_paid_amt['Total'].sum()) * 100,
                    2)
                relation_age_band_wise_paid_amt.loc['Total'] = relation_age_band_wise_paid_amt.sum(numeric_only=True,
                                                                                                   axis=0)
                relation_age_band_wise_paid_amt.loc['Total'] = round(relation_age_band_wise_paid_amt.loc['Total'],2)

                relation_age_band_wise_paid_amt_1 = Paid_Data_Relation.groupby(['age_band_rev', 'std_relation'])[
                    'settledamt'].agg('sum').reset_index()
                relation_age_band_wise_paid_amt_1.rename(columns={'age_band_rev': 'age_band'}, inplace=True)

                relation_age_band_wise_paid_amt_tb = relation_age_band_wise_paid_amt.reset_index()
                relation_age_band_wise_paid_amt_tb.fillna(0, inplace=True)

                relation_age_band_wise_paid_amt_tb.rename(columns={'age_band_rev': 'age_band'}, inplace=True)

                relation_age_band_wise_paid_amt_fig1 = go.Figure(data=[go.Table(
                    header=dict(values=relation_age_band_wise_paid_amt_tb.columns,
                                fill_color='#094780',
                                line_color='darkslategray',
                                align='left',
                                font=dict(size=12, color='white')),
                    cells=dict(values=relation_age_band_wise_paid_amt_tb.T,
                               fill_color='white',
                               line_color='darkslategray',
                               align='left'))
                ], layout=table_layout)

                relation_age_band_wise_paid_amt_fig2 = px.bar(relation_age_band_wise_paid_amt_1, x="std_relation",
                                                              y="settledamt", text='settledamt',
                                                              color_discrete_sequence=px.colors.sequential.RdBu,
                                                              color="age_band", barmode='stack')

                relation_age_band_wise_paid_amt_fig2.update_layout(title='<b><b>')
                relation_age_band_wise_paid_amt_fig2.update_layout(
                    xaxis_title=None,
                    yaxis_title="Paid Amt"
                )

                relation_age_band_wise_paid_amt_fig2.update_layout(title_x=0.5, plot_bgcolor = 'white')

                relation_age_band_wise_paid_amt_fig_1 = plot(relation_age_band_wise_paid_amt_fig1, output_type='div',
                                                             config=config)

                relation_age_band_wise_paid_amt_fig_2 = plot(relation_age_band_wise_paid_amt_fig2, output_type='div',
                                                             config=config)

                # Outstanding Claims Analysis

                out_df['substatus'] = np.where((out_df['head'] == "Under CI") | (out_df['head'] == "Under AL"),
                                               out_df['head'], out_df['extra'])

                out_claim = out_df['substatus'][out_df['consider_count'] == 1].value_counts().reset_index()
                out_claim = pd.DataFrame(out_claim.values, columns=['Claim_Status', 'OS_Count'])

                out_claim1 = out_df['substatus'][out_df['consider_count'] == 1].value_counts().reset_index()
                out_claim1 = pd.DataFrame(out_claim1.values, columns=['Claim_Status', 'OS_Count'])
                out_claim1['OS_Percentage'] = out_claim1['OS_Count'] / out_claim1['OS_Count'].sum() * 100
                out_claim1['OS_Percentage'] = out_claim1['OS_Percentage'].astype('float').round(2)
                out_claim1.loc['Total'] = pd.Series(out_claim1[['OS_Count', 'OS_Percentage']].sum(),
                                                    index=['OS_Count', 'OS_Percentage'])
                out_claim1.reset_index(drop=True, inplace=True)
                out_claim1['Claim_Status'].replace(np.NaN, 'Total', inplace=True)
                out_claim1['OS_Percentage'] = out_claim1['OS_Percentage'].round(2)
                out_claim1.rename(columns={'OS_Percentage': 'OS_Per'}, inplace=True)

                outstanding_claim_analysis_fig1 = px.bar(out_claim, x="Claim_Status", y="OS_Count",
                                                         color="Claim_Status", title="",
                                                         text_auto=True,
                                                         color_discrete_sequence=px.colors.sequential.RdBu)
                outstanding_claim_analysis_fig1.update_layout(xaxis={'visible': False, 'showticklabels': False})

                outstanding_claim_analysis_fig2 = go.Figure(data=[go.Table(
                    columnwidth=[70, 30, 30],
                    header=dict(values=list(out_claim1.columns),
                                fill_color='#094780',
                                line_color='darkslategray',
                                align='left',
                                font=dict(size=12, color='white')),
                    cells=dict(values=[out_claim1.Claim_Status, out_claim1.OS_Count, out_claim1.OS_Per],
                               fill_color='white',
                               line_color='darkslategray',
                               align='left'))
                ])

                outstanding_claim_analysis_fig1.update_layout(title_x=0.5,plot_bgcolor = 'white')

                outstanding_claim_analysis_fig = plot(outstanding_claim_analysis_fig1, output_type='div', config=config)

                outstanding_claim_analysis_tb = plot(outstanding_claim_analysis_fig2, output_type='div', config=config)

                # Rejected Claims Analysis
                rejected_claim = crs_df['substatus'][crs_df['consider_count'] == 1].value_counts().reset_index()
                rejected_claim = pd.DataFrame(rejected_claim.values, columns=['claim_status', 'claim_count'])

                rejected_claim_breakup_fig1 = px.bar(rejected_claim, x="claim_status", y="claim_count",
                                                     color="claim_status", title="",
                                                     text_auto=True, color_discrete_sequence=px.colors.sequential.RdBu)
                rejected_claim_breakup_fig1.update_layout(xaxis={'visible': False, 'showticklabels': False})

                rejected_claim_breakup_fig2 = go.Figure(data=[go.Table(
                    header=dict(values=list(rejected_claim.columns),
                                fill_color='#094780',
                                line_color='darkslategray',
                                align='left',
                                font=dict(size=12, color='white')),
                    cells=dict(values=[rejected_claim.claim_status, rejected_claim.claim_count],
                               fill_color='white',
                               line_color='darkslategray',
                               align='left'))
                ])

                rejected_claim_breakup_fig1.update_layout(title_x=0.5,plot_bgcolor='white')

                rejected_claim_breakup_fig = plot(rejected_claim_breakup_fig1, output_type='div')

                rejected_claim_breakup_tb = plot(rejected_claim_breakup_fig2, output_type='div')

                paid_df['settledamt'] = paid_df['settledamt'].astype('int')

                # Treatment Type wise Analysis

                treatment_type_wise_analysis = paid_df.groupby(['treatmenttype'])[['consider_count', 'settledamt']].agg('sum').reset_index()



                treatment_type_wise_analysis1 = \
                treatment_type_wise_analysis.sort_values('settledamt', axis=0, ascending=False)[['treatmenttype', 'consider_count', 'settledamt']]
                treatment_type_wise_analysis1['Avg_Paid_Amount'] = (
                        treatment_type_wise_analysis1['settledamt'] / treatment_type_wise_analysis1[
                    'consider_count']).round()

                trace1 = go.Scatter(
                    mode='lines+markers',
                    x=treatment_type_wise_analysis['treatmenttype'],
                    y=treatment_type_wise_analysis['consider_count'],
                    text=treatment_type_wise_analysis["consider_count"],
                    name="Paid Claims",
                    marker_color='crimson'
                )

                trace2 = go.Bar(
                    x=treatment_type_wise_analysis['treatmenttype'],
                    y=treatment_type_wise_analysis['settledamt'], text=treatment_type_wise_analysis["settledamt"],
                    name="Paid Amount",
                    yaxis='y2',
                    marker_color='blue',
                    marker_line_width=1.5,
                    marker_line_color='rgb(8,48,107)',
                    opacity=0.5
                )

                data = [trace1, trace2]

                layout = go.Layout(
                    title_text='',
                    yaxis=dict(
                        range=[0, 600],
                        autorange=True,  # Allow autoscaling
                        side='right'
                    ),
                    yaxis2=dict(
                        overlaying='y',
                        anchor='y3',
                        rangemode='tozero'
                    ),
                    plot_bgcolor='white',
                    autosize=True
                )
                fig = go.Figure(data=data, layout=layout)
                # iplot(fig, filename='multiple-axes-double')
                treatment_type_wise_analysis_fig = plot(fig, output_type='div')

                ##Table
                treatment_type_wise_analysis1 = go.Figure(data=[go.Table(
                    columnwidth=[50,50,50],
                    header=dict(values=treatment_type_wise_analysis1.columns,
                                fill_color='#094780',
                                line_color='darkslategray',
                                align='left',
                                font=dict(size=12, color='white')),
                    cells=dict(
                        values=[treatment_type_wise_analysis1.treatmenttype,
                                treatment_type_wise_analysis1.consider_count,
                                treatment_type_wise_analysis1.settledamt,
                                treatment_type_wise_analysis1.Avg_Paid_Amount],
                        fill_color='white',
                        line_color='darkslategray',
                        align='left'))])

                treatment_type_wise_analysis1.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0)
                )


                treatment_type_wise_analysis_tb = plot(treatment_type_wise_analysis1, output_type='div')

                # paid_ipd_opd_liablityamt_count
                paid_ipd_opd_liablityamt = paid_df.groupby(['status', 'ipd_opd']).agg(
                    {'settledamt': sum}).reset_index()

                paid_ipd_opd_claim_count = paid_df[['status', 'ipd_opd']][
                    paid_df['consider_count'] == 1].value_counts().reset_index()
                paid_ipd_opd_claim_count = pd.DataFrame(paid_ipd_opd_claim_count.values,
                                                        columns=['status', 'ipd_opd', 'claim_count'])
                paid_ipd_opd_claim_count['claim_count'] = paid_ipd_opd_claim_count['claim_count'].astype('int64')

                paid_ipd_opd_liablityamt_count = paid_ipd_opd_liablityamt
                paid_ipd_opd_liablityamt_count['claim_count'] = paid_ipd_opd_claim_count['claim_count']

                # cashless_vs_reimbersment_claim_count
                # Paid
                cashless_vs_reimbersement_claim_count_paid = paid_df.groupby(['status', 'actuallosstype']).agg(
                    {'consider_count': sum}).reset_index()

                df = cashless_vs_reimbersement_claim_count_paid.copy()
                data = [
                    go.Pie(labels=df['actuallosstype'],
                           values=df['consider_count'], text=df['consider_count'],
                           marker_colors=px.colors.qualitative.Plotly)
                ]
                cashless_vs_reimbersement_claim_count_paid1 = go.Figure(data=data)
                cashless_vs_reimbersement_claim_count_paid1.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                                                                          title='<b>Cashless vs Reimbersement Paid Claim Count<b>')
                cashless_vs_reimbersement_claim_count_paid1.update_layout(title_x=0.5)

                cashless_vs_reimbersement_claim_count_paid_fig = plot(cashless_vs_reimbersement_claim_count_paid1,
                                                                      output_type='div', config=config)

                # Out
                cashless_vs_reimbersement_claim_count_out = out_df.groupby(['status', 'actuallosstype']).agg(
                    {'consider_count': sum}).reset_index()

                # PAID AMOUNT ANALYSIS

                cashless_vs_reimbersement_claim_amt_paid = paid_df.groupby(['status', 'actuallosstype']).agg(
                    {'settledamt': sum}).reset_index()
                cashless_vs_reimbersement_claim_amt_paid_df = cashless_vs_reimbersement_claim_amt_paid.copy()

                cashless_vs_reimbersement_claim_amt_paid_fig1 = [
                    go.Pie(labels=cashless_vs_reimbersement_claim_amt_paid_df['actuallosstype'],
                           values=cashless_vs_reimbersement_claim_amt_paid_df['settledamt'],
                           text=cashless_vs_reimbersement_claim_amt_paid_df['settledamt'],
                           marker_colors=px.colors.qualitative.Plotly)
                ]
                cashless_vs_reimbersement_claim_amt_paid_fig = go.Figure(
                    data=cashless_vs_reimbersement_claim_amt_paid_fig1)
                cashless_vs_reimbersement_claim_amt_paid_fig.update_layout(margin=dict(t=100, b=100, l=100, r=100),
                                                                           title='<b>Cashless vs Reimbersement Paid Amount<b>')
                cashless_vs_reimbersement_claim_amt_paid_fig.update_layout(title_x=0.5)

                cashless_vs_reimbersement_claim_amt_paid_fig = plot(cashless_vs_reimbersement_claim_amt_paid_fig,
                                                                    output_type='div')

                ## CASHLESS UTILIZATION

                cashless_consider = paid_df[paid_df['consider_count'] == 1]

                cashless_utilization = cashless_consider.groupby(
                    ['ipd_opd', 'actuallosstype']).size().reset_index().rename(columns={0: 'Count'})

                cashless_utilization1 = cashless_utilization.pivot_table(index=['ipd_opd'],
                                                                         columns='actuallosstype', values='Count',
                                                                         aggfunc='sum')

                cashless_utilization1.reset_index(inplace=True)
                cashless_utilization1.replace(np.NaN, 0, inplace=True)
                cashless_utilization2 = cashless_utilization1.iloc[:, 1:].apply(lambda x: x / x.sum() * 100,
                                                                                axis=1).round(2)

                cashless_utilization3 = pd.concat([cashless_utilization1[['ipd_opd']], cashless_utilization2], axis=1)

                cashless_utilization_data = [
                    go.Bar(x=cashless_utilization3['ipd_opd'], y=cashless_utilization3['Cash Less'], name='Cash Less',
                           text=cashless_utilization3["Cash Less"]),
                    go.Bar(x=cashless_utilization3['ipd_opd'], y=cashless_utilization3['Non Cash Less'],
                           name='Non Cash Less', text=cashless_utilization3["Non Cash Less"])
                ]

                cashless_utilization_layout = go.Layout(
                    title='CASHLESS UTILIZATION',
                    yaxis=dict(title='Percentage')
                )

                cashless_utilization_fig = go.Figure(data=cashless_utilization_data, layout=cashless_utilization_layout)
                cashless_utilization_fig = plot(cashless_utilization_fig, output_type='div')

                # df1 = cashless_vs_reimbersement_claim_count_out.copy()
                # data = [
                #     go.Pie(labels=df1['actuallosstype'],
                #            values=df1['consider_count'], text=df1['consider_count'],
                #            hole=.5, marker_colors=px.colors.qualitative.Plotly)
                # ]
                # cashless_vs_reimbersement_claim_count_out1 = go.Figure(data=data)
                # cashless_vs_reimbersement_claim_count_out1.update_layout(margin=dict(t=100, b=100, l=100, r=100),
                #                                                           title='<b>Cashless vs Reimbersement Paid Claim Count<b>')
                # cashless_vs_reimbersement_claim_count_out1.update_layout(title_x=0.5)
                #
                # cashless_vs_reimbersement_claim_count_out_fig = plot(cashless_vs_reimbersement_claim_count_out1,
                #                                                   output_type='div', config=config)

                # CRS
                cashless_vs_reimbersement_claim_count_crs = crs_df.groupby(['status', 'actuallosstype']).agg(
                    {'consider_count': sum}).reset_index()
                # cashless_vs_reimbersement_claim_count_crs = cashless_vs_reimbersement_claim_count_crs.to_html()

                cashless_vs_reimbersement_claim_count_1 = pd.concat(
                    [cashless_vs_reimbersement_claim_count_paid, cashless_vs_reimbersement_claim_count_crs,
                     cashless_vs_reimbersement_claim_count_out])
                cashless_vs_reimbersement_claim_count = cashless_vs_reimbersement_claim_count_1.groupby(
                    ['actuallosstype']).agg({'consider_count': sum}).reset_index()

                # cashless_vs_reimbersment_lodge_amount
                # Paid

                cashless_vs_reimbersement_lodge_amount_paid = paid_df.groupby(['status', 'actuallosstype']).agg(
                    {'actual_lodge_amt': sum}).reset_index()

                # out
                cashless_vs_reimbersement_lodge_amount_out = out_df.groupby(['status', 'actuallosstype']).agg(
                    {'actual_lodge_amt': sum}).reset_index()

                # crs
                cashless_vs_reimbersement_lodge_amount_crs = crs_df.groupby(['status', 'actuallosstype']).agg(
                    {'actual_lodge_amt': sum}).reset_index()

                cashless_vs_reimbersement_lodge_amount_1 = pd.concat(
                    [cashless_vs_reimbersement_lodge_amount_paid, cashless_vs_reimbersement_lodge_amount_crs,
                     cashless_vs_reimbersement_lodge_amount_out])
                cashless_vs_reimbersement_lodge_amount = cashless_vs_reimbersement_lodge_amount_1.groupby(
                    ['actuallosstype']).agg({'actual_lodge_amt': sum}).reset_index()

                cashless_vs_reimbersement_lodge_amount_data = [
                    go.Pie(labels=cashless_vs_reimbersement_lodge_amount['actuallosstype'],
                           values=cashless_vs_reimbersement_lodge_amount['actual_lodge_amt'],
                           text=cashless_vs_reimbersement_lodge_amount['actual_lodge_amt'],
                            marker_colors=px.colors.qualitative.Plotly)
                ]
                cashless_vs_reimbersement_lodge_amount_fig = go.Figure(data=cashless_vs_reimbersement_lodge_amount_data)
                cashless_vs_reimbersement_lodge_amount_fig.update_layout(margin=dict(t=100, b=100, l=100, r=100),
                                                                         title='<b>Cashless vs Reimbersement Lodge Amt<b>')
                cashless_vs_reimbersement_lodge_amount_fig.update_layout(title_x=0.5)

                cashless_vs_reimbersement_lodge_amount_fig = plot(cashless_vs_reimbersement_lodge_amount_fig,
                                                                  output_type='div')

                out_df['settledamt'] = 0
                crs_df['settledamt'] = 0
                crs_df['liablityamt'] = 0

                cols = ['status', 'actuallosstype', 'actual_lodge_amt', 'settledamt', 'consider_count', 'ipd_opd']

                dump = pd.concat([paid_df[cols], crs_df[cols], out_df[cols]]).reset_index(drop=True)
                dump['actual_lodge_amt'] = dump['actual_lodge_amt'].fillna(0)
                dump['consider_count'] = dump['consider_count'].fillna(0)
                # cashless_vs_reimbursement_average_amount

                cond1 = dump['consider_count'] == 1
                dump['avg_lodge_amt'] = round(
                    dump[cond1]['actual_lodge_amt'].astype(int) / dump[cond1]['consider_count'].astype(int))

                cashless_vs_reimbursement_average_lodge_amount = pd.pivot_table(dump, values='avg_lodge_amt',
                                                                                index=['actuallosstype']
                                                                                ).replace(np.nan, 0)
                cashless_vs_reimbursement_average_lodge_amount.round().astype('int')

                cond2 = paid_df['consider_count'] == 1
                paid_df['avg_paid_amt'] = round(
                    paid_df[cond2]['settledamt'].astype(int) / paid_df[cond2]['consider_count'].astype(int))

                cashless_vs_reimbursement_average_paid_amount = pd.pivot_table(paid_df, values='avg_paid_amt',
                                                                               index=['actuallosstype']
                                                                               ).replace(np.nan, 0)
                cashless_vs_reimbursement_average_paid_amount.round().astype('int')


                # cashless_vs_reimbursement_average_amount_ipd
                ipd_filtered_lodge = dump.loc[(dump['consider_count'] == 1) & (dump['ipd_opd'] == 'IPD')]

                ipd_filtered_lodge['avg_lodge_amt_ipd'] = round(
                    ipd_filtered_lodge['actual_lodge_amt'].astype(int) / ipd_filtered_lodge['consider_count'].astype(
                        int))

                ipd_filtered_paid = dump.loc[
                    (dump['status'] == 'Paid') & (dump['consider_count'] == 1) & (dump['ipd_opd'] == 'IPD')]

                cashless_vs_reimbursement_average_lodge_amount_ipd = pd.pivot_table(ipd_filtered_lodge,
                                                                                    values='avg_lodge_amt_ipd',
                                                                                    index=['actuallosstype']
                                                                                    ).replace(np.nan, 0)
                cashless_vs_reimbursement_average_lodge_amount_ipd.round().astype('int')

                ipd_filtered_paid['avg_paid_amt_ipd'] = ipd_filtered_paid['settledamt'] / ipd_filtered_paid[
                    'consider_count']
                ipd_filtered_paid['avg_paid_amt_ipd'] = ipd_filtered_paid['avg_paid_amt_ipd'].astype('int')

                cashless_vs_reimbursement_average_paid_amount_ipd = pd.pivot_table(ipd_filtered_paid,
                                                                                   values='avg_paid_amt_ipd',
                                                                                   index=['actuallosstype']
                                                                                   ).replace(np.nan, 0)

                cashless_vs_reimbursement_average_paid_amount_ipd.round().astype('int')
                # Total Paid No by Age Band
                paid_df['age_band_rev'].value_counts().rename_axis('age_band').reset_index(name='Paid Count')
                # Total Paid Amount by Age Band
                paid_df.groupby(['age_band_rev'])['settledamt'].agg('sum').reset_index(name='Paid Amount')

                # Incidence Rate by Age Band
                age_band_consider_count = pd.concat(
                    [paid_df[['age_band_rev', 'consider_count']], out_df[['age_band_rev', 'consider_count']],
                     crs_df[['age_band_rev', 'consider_count']]])

                age_band_ir = age_band_consider_count.groupby('age_band_rev')['consider_count'].sum().reset_index()
                age_band_ir1 = pd.merge(age_band_ir, age_band_wise_lives_covered, how='inner', left_on='age_band_rev',
                                        right_on='age_band')
                age_band_ir1['IR'] = age_band_ir1['consider_count'] / age_band_ir1['lives'] * 100
                age_band_ir1['IR'] = age_band_ir1['IR'].round(2)

                Age_Band_IR = px.bar(age_band_ir1.sort_values('age_band_rev', ascending=False), x="age_band_rev",
                                     y="IR", color="age_band_rev",
                                     title="<b><b>", text="IR",
                                     color_discrete_sequence=px.colors.sequential.RdBu)
                # SI_Band_IR.update_layout( xaxis={'visible': False, 'showticklabels': False})

                Age_Band_IR.update_layout(title_x=0.5)
                Age_Band_IR = plot(Age_Band_IR, output_type='div')

                no_of_claims_for_paid = pd.DataFrame(
                    paid_df.assign(count=(paid_df['status'] == 'Paid')).groupby(['pol_no', 'insuredname'],
                                                                                sort=False, as_index=False)[
                        'consider_count'].agg({'count': sum}))
                no_of_claims_for_paid.rename(columns={'count': 'no_of_claims_for_paid'}, inplace=True)
                # no_of_claims_for_paid = no_of_claims_for_paid[(no_of_claims_for_paid['Status'] == 'Paid')]
                # Cashless Paid Amt by Age Band
                paid_df.groupby(['age_band_rev'])['settledamt'].agg('sum').reset_index(name='Paid Amount')

                cashless_paid_amt_age_band = \
                    paid_df[paid_df['actuallosstype'] == 'Cash Less'].groupby(['age_band_rev'], sort=False,
                                                                              as_index=False)['settledamt'].agg(
                        {'count': sum})
                cashless_paid_amt_age_band.rename(columns={'count': 'cashless_paid_amt_age_band'}, inplace=True)
                # no_of_claims_for_paid = no_of_claims_for_paid[(no_of_claims_for_paid['Status'] == 'Paid')]

                non_cashless_paid_amt_age_band = \
                    paid_df[paid_df['actuallosstype'] == 'Non Cash Less'].groupby(['age_band_rev'], sort=False,
                                                                                  as_index=False)['settledamt'].agg(
                        {'count': sum})
                non_cashless_paid_amt_age_band.rename(columns={'count': 'non_cashless_paid_amt_age_band'}, inplace=True)
                # no_of_claims_for_paid = no_of_claims_for_paid[(no_of_claims_for_paid['Status'] == 'Paid')]

                # Top Ailment Wise Analysis

                d = paid_df[paid_df['new_disease_category'] != 'Other']
                top_ailment_wise_analysis = d.groupby(['new_disease_category'], sort=False, as_index=False)[
                ['settledamt', 'consider_count']].agg(sum)

                top10_ailment_wise_analysis = \
                top_ailment_wise_analysis.sort_values('settledamt', axis=0, ascending=False)[
                    ['new_disease_category', 'settledamt', 'consider_count']].head(10)
                top10_ailment_wise_analysis.rename(
                    columns={'new_disease_category': 'Disease_catogary', 'settledamt': 'PaidAmt',
                             'consider_count': 'PaidClaims'}, inplace=True)

                # --------------------------------------Fig-----------------------------------------------
                top10_ailment_wise_analysis_fig2 = go.Figure(data=[go.Table(
                    header=dict(values=list(top10_ailment_wise_analysis.columns),
                                fill_color='#094780',
                                line_color='darkslategray',
                                align='left',
                                font=dict(size=12, color='white')),
                    cells=dict(
                        values=[top10_ailment_wise_analysis.Disease_catogary, top10_ailment_wise_analysis.PaidAmt,
                                top10_ailment_wise_analysis.PaidClaims],
                        fill_color='white',
                        line_color='darkslategray',
                        align='left'))
                ])

                top10_ailment_wise_analysis_fig1 = go.Figure()
                # fig.add_trace(go.Bar(
                #     x=top_ailment_wise_analysis['DiseaseCategory'],
                #     y=top_ailment_wise_analysis['SettledAmt'],
                #     name='Setlled amt',
                #     marker_color='indianred'
                # ))
                top10_ailment_wise_analysis_fig1.add_trace(go.Bar(
                    x=top10_ailment_wise_analysis['PaidAmt'],
                    y=top10_ailment_wise_analysis['Disease_catogary'],
                    text=top10_ailment_wise_analysis['PaidAmt'],
                    name='Paid Amount',
                    marker_color='lightsalmon', orientation='h'
                ))
                top10_ailment_wise_analysis_fig1.update_traces(textposition='auto')
                top10_ailment_wise_analysis_fig1.update_layout(yaxis=dict(autorange="reversed"))
                top10_ailment_wise_analysis_fig1.update_layout(title='<b><b>')
                top10_ailment_wise_analysis_fig1.update_layout(title_x=0.5)

                top10_ailment_wise_analysis_fig1.update_layout(barmode='group', xaxis_tickangle=-90)
                top10_ailment_wise_analysis_fig1 = plot(top10_ailment_wise_analysis_fig1, output_type='div')
                top10_ailment_wise_analysis_fig2 = plot(top10_ailment_wise_analysis_fig2, output_type='div')

                # Top 10 Hospital wise Paid Amount -IPD Only

                paid_df['consider_count'].replace(0, np.nan, inplace=True)
                paid_df['claim_count'] = paid_df[paid_df['ipd_opd'] == 'IPD'].groupby('hospitlname')[
                    'consider_count'].transform('count')
                top10hospital_paidamt_ipd = paid_df.sort_values('settledamt', axis=0, ascending=False)[
                    ['hospitlname', 'settledamt', 'claim_count']].head(10)

                # Top 10 Hospital wise Paid Amount

                top_hospital_paidamt = paid_df.groupby(['hospitlname'], sort=False, as_index=False)[[
                    'settledamt', 'consider_count']].sum()
                top10hospital_paidamt = top_hospital_paidamt.sort_values('settledamt', axis=0, ascending=False)[
                    ['hospitlname', 'settledamt', 'consider_count']].head(10)
                top10hospital_paidamt.rename(
                    columns={'hospitlname': 'HospitalName', 'settledamt': 'PaidAmt', 'consider_count': 'PaidClaims'},
                    inplace=True)
                # define a regular expression pattern to match the text before "hospital"
                import re
                pattern = r'^(.*?)\bHospital\b'

                # define a function to extract the text before "hospital" using regex
                def extract_text(text):
                    match = re.search(pattern, text)
                    if match:
                        return match.group(0).strip()
                    else:
                        return text

                # apply the function to the 'text' column to extract the text before "hospital"
                top10hospital_paidamt['HospitalName'] = top10hospital_paidamt['HospitalName'].apply(extract_text)
                # top10hospital_paidamt['HospitalName1'] = top10hospital_paidamt['HospitalName'].apply(
                #     lambda x: "<br>".join(textwrap.wrap(x, width=30)) if len(x) > 4 else x)
                pattern = r'\s*\([^)]*\)'

                # Use the replace() method to remove text in brackets from the 'text' column
                top10hospital_paidamt['HospitalName'] = top10hospital_paidamt['HospitalName'].replace(pattern, '',
                                                                                                      regex=True)

                top10_hospital_wise_paidamt_fig2 = go.Figure(data=[go.Table(
                    header=dict(values=list(top10hospital_paidamt.columns),
                                fill_color='#094780',
                                line_color='darkslategray',
                                align='left',
                                font=dict(size=12, color='white')),
                    cells=dict(values=[top10hospital_paidamt.HospitalName, top10hospital_paidamt.PaidAmt,
                                       top10hospital_paidamt.PaidAmt],
                               fill_color='white',
                               line_color='darkslategray',
                               align='left'))
                ])

                top10_hospital_wise_paidamt_fig1 = go.Figure()
                # fig.add_trace(go.Bar(
                #     x=top_ailment_wise_analysis['DiseaseCategory'],
                #     y=top_ailment_wise_analysis['SettledAmt'],
                #     name='Setlled amt',
                #     marker_color='indianred'
                # ))
                top10_hospital_wise_paidamt_fig1.add_trace(go.Bar(
                    x=top10hospital_paidamt['PaidAmt'],
                    y=top10hospital_paidamt['HospitalName'],
                    text=top10hospital_paidamt['PaidAmt'],
                    name='Paid Amount',
                    marker_color='skyblue', orientation='h'
                ))
                top10_hospital_wise_paidamt_fig1.update_traces(textposition='auto')
                top10_hospital_wise_paidamt_fig1.update_layout(yaxis=dict(autorange="reversed"))
                top10_hospital_wise_paidamt_fig1.update_layout(title='<b><b>')
                top10_hospital_wise_paidamt_fig1.update_layout(title_x=0.5)

                # fig.update_layout(barmode='group', xaxis_tickangle=-45)
                top10_hospital_wise_paidamt_fig1 = plot(top10_hospital_wise_paidamt_fig1, output_type='div')
                top10_hospital_wise_paidamt_fig2 = plot(top10_hospital_wise_paidamt_fig2, output_type='div')

                # Amt Band Wise Paid Amt
                conditions = [
                    (paid_df['settledamt'] >= 0) & (paid_df['settledamt'] <= 10000),
                    (paid_df['settledamt'] >= 10001) & (paid_df['settledamt'] <= 20000),
                    (paid_df['settledamt'] >= 20001) & (paid_df['settledamt'] <= 30000),
                    (paid_df['settledamt'] >= 30001) & (paid_df['settledamt'] <= 50000),
                    (paid_df['settledamt'] >= 50001) & (paid_df['settledamt'] <= 75000),
                    (paid_df['settledamt'] >= 75001) & (paid_df['settledamt'] <= 100000),
                    (paid_df['settledamt'] >= 100001) & (paid_df['settledamt'] <= 150000),
                    (paid_df['settledamt'] >= 150001) & (paid_df['settledamt'] <= 200000),
                    (paid_df['settledamt'] >= 200001) & (paid_df['settledamt'] <= 300000),
                    (paid_df['settledamt'] >= 300001) & (paid_df['settledamt'] <= 1000000000),
                ]

                # create a list of the values we want to assign for each condition
                values = ['0-10000', '10001-20000', '20001-30000', '30001-50000', '50001-75000', '75001-100000',
                          '100001-150000', '150001-200000', '200001-300000', 'Above 300000']

                # create a new column and use np.select to assign values to it using our lists as arguments
                paid_df['amt_band'] = np.select(conditions, values)

                amt_band_wise_paid_amt = paid_df[paid_df['consider_count'] == 1].groupby(["amt_band"])[
                    'settledamt'].sum().reset_index()

                paid_amt_band_cashless = (paid_df[paid_df['consider_count'] == 1])
                paid_amt_band_cashless = (
                    paid_amt_band_cashless[paid_amt_band_cashless['actuallosstype'] == 'Cash Less'])
                paid_amt_band_cashless = paid_amt_band_cashless.groupby(["amt_band"])['settledamt'].sum().reset_index()

                paid_amt_band_non_cashless = (paid_df[paid_df['consider_count'] == 1])
                paid_amt_band_non_cashless = (
                    paid_amt_band_non_cashless[paid_amt_band_non_cashless['actuallosstype'] == 'Non Cash Less'])
                paid_amt_band_non_cashless = paid_amt_band_non_cashless.groupby(["amt_band"])[
                    'settledamt'].sum().reset_index()

                total_insured_person = portal_premium_df['lives'].sum()

                pd.options.display.float_format = '₹{:.2f}'.format
                total_premium = portal_premium_df['Premium'].unique().sum()
                print(total_premium)
                total_premium_rupees = format_currency(total_premium, 'INR', locale='en_IN')

                total_earned_premium = portal_premium_df['EARNED_PERMIUM'].unique().sum()
                total_earned_premium = round(total_earned_premium,2)
                total_earned_premium_rupees = format_currency(total_earned_premium, 'INR', locale='en_IN')

                policy_coverage_completion_days = portal_premium_df['POLICYRUNDAY'].unique().sum()

                incurred_amount = paid_df['liablityamt'].sum() + out_df['liablityamt'].sum()
                incurred_amount_rupees = format_currency(incurred_amount, 'INR', locale='en_IN')

                # Calculate the sum of 'lives'
                total_lives = portal_premium_df['lives'].sum()

                # Check if the sum of 'lives' is non-zero
                if total_lives != 0:
                    # Calculate the average premium per life
                    avg_premium_per_life = (portal_premium_df['Premium'].mean() / total_lives).round()
                else:
                    # Set a default value or handle the case when the sum of 'lives' is zero
                    avg_premium_per_life = 0


                avg_premium_per_life_rupees = format_currency(avg_premium_per_life, 'INR', locale='en_IN')

                # Check if the sum of 'lives' is non-zero
                if total_lives != 0:
                    # Perform the division and round the result
                    avg_incurred_amt_per_life = round(incurred_amount / total_lives)
                else:
                    # Set a default value or handle the case when the sum of 'lives' is zero
                    avg_incurred_amt_per_life = 0


                avg_incurred_amt_per_life_rupees = format_currency(avg_incurred_amt_per_life, 'INR', locale='en_IN')

                incurred_amount = float(incurred_amount)
                if total_premium != 0:
                    claims_ratio_per = round(incurred_amount / total_premium * 100)
                else:
                    claims_ratio_per = 0
                claims_ratio_percentage = str(claims_ratio_per) + '%'
                # claims_ratio_percentage = claims_ratio_per.astype(str) + '%'
                if total_earned_premium != 0:
                    claims_ratio_on_earned_premium = (incurred_amount / total_earned_premium * 100).round()
                else:
                    claims_ratio_on_earned_premium = 0

                claims_ratio_on_earned_premium_percentage = str(claims_ratio_on_earned_premium) + '%'

                cashless_out_lodge_amt = out_df.loc[(out_df['actuallosstype'].str.strip() == 'Non Cash Less') & (
                        out_df['lodgetype'].str.strip() != 'Deductions Payment')]

                cashless_paid_liablity_amt = paid_df.loc[(paid_df['actuallosstype'].str.strip() == 'Non Cash Less')]

                current_incurred_non_cashless_amt = cashless_paid_liablity_amt['liablityamt'].sum() + \
                                                    cashless_out_lodge_amt['actual_lodge_amt'].sum()

                non_cashless_paid = paid_df['actuallosstype'].str.strip() == 'Non Cash Less'
                non_cashless_out = out_df['actuallosstype'].str.strip() == 'Non Cash Less'
                non_cashless_crs = crs_df['actuallosstype'].str.strip() == 'Non Cash Less'

                data = pd.concat([paid_df[non_cashless_paid][['lodgedate', 'dod']],
                                  out_df[non_cashless_out][['lodgedate', 'dod']],
                                  crs_df[non_cashless_crs][['lodgedate', 'dod']]])

                gap = (data['lodgedate'] - data['dod']).apply(lambda x: x.days)
                gap = gap.apply(lambda x: x if x > 0 else 0)
                gap = gap.mean()

                incurred_but_not_reported_amount_till_date = (
                        float(current_incurred_non_cashless_amt) * gap / policy_coverage_completion_days).round()

                ibnr = (incurred_but_not_reported_amount_till_date / incurred_amount * 100).round()
                ibnr_percentage = ibnr.astype(str) + '%'

                ir = (paid_count['claim_count'].astype(int).sum() + out_count['claim_count'].astype(int).sum() +
                      crs_count['claim_count'].astype(int).sum()) / portal_premium_df['lives'].astype(int).sum() * 100
                incident_rate = ir.round(2)
                incident_rate_percenatge = incident_rate.astype(str) + '%'

                prorated_ir = (incident_rate * 365 / policy_coverage_completion_days).round(2)
                prorated_ir_percenatge = prorated_ir.astype(str) + '%'

                if policy_coverage_completion_days != 0:
                    total_likely_incurred_amount_at_policy_expiry = (
                                incurred_amount * 365 / policy_coverage_completion_days).round()
                else:
                    # Handle the case when policy_coverage_completion_days is zero
                    # For example, set the total_likely_incurred_amount_at_policy_expiry to a default value or raise an exception.
                    total_likely_incurred_amount_at_policy_expiry = 0  # Set to a default value, but you can choose an appropriate alternative action.

                IBNR_Amount_at_policy_Expiry = total_likely_incurred_amount_at_policy_expiry * ibnr

                total_likely_outgo_for_policy = total_likely_incurred_amount_at_policy_expiry + IBNR_Amount_at_policy_Expiry
                total_likely_outgo_for_policy_rupees = format_currency(total_likely_outgo_for_policy, 'INR',
                                                                       locale='en_IN')

                if total_insured_person != 0:
                    avg_incurred_amt_per_life = (incurred_amount / total_insured_person).round()
                else:
                    avg_incurred_amt_per_life = 0


                Lodge_Amount = ALL_CLAIMS_LODGE_AMT_BY_CLAIM_STATUS['actual_lodge_amt'].sum()

                Reported_claims = paid_df['consider_count'].sum() + crs_df['consider_count'].sum() + out_df[
                    'consider_count'].sum()

                Settled_claims = paid_df['consider_count'].sum() + crs_df['consider_count'].sum()

                Settlement_Ratio = round((Settled_claims / Reported_claims) * 100, 2)
                Settlement_Ratio_Per = Settlement_Ratio.astype(str) + '%'

                Paid_Amount = paid_df['settledamt'].sum()

                a1 = {'total_insured_person': total_insured_person, 'total_premium': total_premium_rupees,
                      'total_earned_premium': total_earned_premium_rupees,
                      'policy_coverage_completion_days': policy_coverage_completion_days,
                      'incurred_amount': incurred_amount_rupees, 'avg_premium_per_life': avg_premium_per_life_rupees,
                      'claims_ratio_per': claims_ratio_percentage,
                      'claims_ratio_on_earned_premium': claims_ratio_on_earned_premium_percentage,
                      'ibnr': ibnr_percentage, 'incident_rate': incident_rate_percenatge,
                      'prorated_ir': prorated_ir_percenatge,
                      'total_likely_outgo_for_policy': total_likely_outgo_for_policy_rupees,
                      'total_likely_incurred_amount_at_policy_expiry': total_likely_incurred_amount_at_policy_expiry,
                      'avg_incurred_amt_per_life': avg_incurred_amt_per_life}

                # Consolidated Summery
                pd.options.display.float_format = '{:.2f}'.format
                summary = pd.DataFrame(zip(a1.keys(), (a1.values())), columns=['Parameters', 'Values'])

                summary2 = go.Figure(data=[go.Table(#columnwidth=[90, 50],
                                                    header=dict(values=list(summary.columns),
                                                                fill_color='#094780',
                                                                line_color='darkslategray',
                                                                align='left',
                                                                font=dict(size=12, color='white')),
                                                    cells=dict(values=[summary.Parameters, summary.Values],
                                                               fill_color='white',
                                                               line_color='darkslategray',
                                                               align='left'))
                                           ], layout=table_layout)

                summary2.layout.width = 1000

                summary2.update_layout(height=len(summary) * 30)

                summary1 = plot(summary2, output_type='div')

                # Summary Bar Plot

                numbers = {'Premium': total_premium,
                           'Earned Premium': total_earned_premium, 'Lodge Amount': Lodge_Amount,
                           'Incurred Amount': incurred_amount, 'Paid Amount': Paid_Amount}
                numbers = pd.DataFrame(zip(numbers.keys(), (numbers.values())), columns=['Parameters', 'Values'])

                percentage = {'IR(%)': incident_rate,
                              'Prorated Incident Rate(%)': prorated_ir, 'ICR(%)': claims_ratio_per,
                              'ECR(%)': claims_ratio_on_earned_premium, 'Settlement Ratio(%)': Settlement_Ratio}

                percentage = pd.DataFrame(zip(percentage.keys(), (percentage.values())),
                                          columns=['Parameters', 'Values'])

                Summary_plot = px.bar(numbers, x="Parameters", y="Values",
                                      color="Parameters", title="", text="Values",
                                      color_discrete_sequence=px.colors.sequential.RdBu)
                Summary_plot.update_layout(xaxis={'visible': True, 'showticklabels': True})
                Summary_plot.update_layout(title_x=0.5)

                Summary_plot = plot(Summary_plot, output_type='div')

                Summary_plot2 = px.bar(percentage, x="Parameters", y="Values",
                                       color="Parameters", title="",
                                       text=[f'{i}%' for i in percentage['Values']],
                                       color_discrete_sequence=px.colors.sequential.RdBu)
                Summary_plot2.update_layout(xaxis={'visible': True, 'showticklabels': True})
                Summary_plot2.update_layout(title_x=0.5)

                Summary_plot2 = plot(Summary_plot2, output_type='div')

                # IPD_OPD WISE SUMMARY

                crs_ipd_summary = (crs_df[crs_df['ipd_opd'] == 'IPD'])
                paid_ipd_summary = (paid_df[paid_df['ipd_opd'] == 'IPD'])
                out_ipd_summary = (out_df[out_df['ipd_opd'] == 'IPD'])

                crs_opd_summary = (crs_df[crs_df['ipd_opd'] == 'OPD'])
                paid_opd_summary = (paid_df[paid_df['ipd_opd'] == 'OPD'])
                out_opd_summary = (out_df[out_df['ipd_opd'] == 'OPD'])

                Reported_claims_ipd_summary = paid_ipd_summary['consider_count'].sum() + crs_ipd_summary[
                    'consider_count'].sum() + out_ipd_summary['consider_count'].sum()

                Reported_claims_opd_summary = paid_opd_summary['consider_count'].sum() + crs_opd_summary[
                    'consider_count'].sum() + out_opd_summary['consider_count'].sum()

                Reported_amt_ipd_summary = \
                    paid_ipd_summary[(paid_ipd_summary['lodgetype'].str.strip() != "Deductions Payment")][
                        'lodgeamt'].agg(
                        {'count': sum}) + \
                    crs_ipd_summary[(crs_ipd_summary['lodgetype'].str.strip() != "Deductions Payment")][
                        'cl_lod_amt'].agg(
                        {'count': sum}) + \
                    out_ipd_summary[(out_ipd_summary['lodgetype'].str.strip() != "Deductions Payment")][
                        'actuallodgeamt'].agg(
                        {'count': sum})
                Reported_amt_ipd_summary = Reported_amt_ipd_summary[0]
                Reported_amt_ipd = format_currency(Reported_amt_ipd_summary, 'INR', locale='en_IN')

                Reported_amt_opd_summary = \
                    paid_opd_summary[(paid_opd_summary['lodgetype'].str.strip() != "Deductions Payment")][
                        'lodgeamt'].agg(
                        {'count': sum}) + \
                    crs_opd_summary[(crs_opd_summary['lodgetype'].str.strip() != "Deductions Payment")][
                        'cl_lod_amt'].agg(
                        {'count': sum}) + \
                    out_opd_summary[(out_opd_summary['lodgetype'].str.strip() != "Deductions Payment")][
                        'actuallodgeamt'].agg(
                        {'count': sum})
                Reported_amt_opd_summary = Reported_amt_opd_summary[0]
                Reported_amt_opd = format_currency(Reported_amt_opd_summary, 'INR', locale='en_IN')

                settled_claims_ipd_summary = paid_ipd_summary['consider_count'].sum() + crs_ipd_summary[
                    'consider_count'].sum()

                settled_claims_opd_summary = paid_opd_summary['consider_count'].sum() + crs_opd_summary[
                    'consider_count'].sum()

                incurred_amt_ipd_summary = paid_ipd_summary['liablityamt'].sum() + out_ipd_summary['liablityamt'].sum()
                incurred_amt_ipd = format_currency(incurred_amt_ipd_summary, 'INR', locale='en_IN')

                incurred_amt_opd_summary = paid_opd_summary['liablityamt'].sum() + out_opd_summary['liablityamt'].sum()
                incurred_amt_opd = format_currency(incurred_amt_opd_summary, 'INR', locale='en_IN')

                paid_claims_ipd_summary = paid_ipd_summary['consider_count'].sum()

                paid_claims_opd_summary = paid_opd_summary['consider_count'].sum()

                paid_amt_ipd_summary = paid_ipd_summary['liablityamt'].sum()
                paid_amt_ipd = format_currency(paid_amt_ipd_summary, 'INR', locale='en_IN')

                paid_amt_opd_summary = paid_opd_summary['liablityamt'].sum()
                paid_amt_opd = format_currency(paid_amt_opd_summary, 'INR', locale='en_IN')


                lives_ipd_opd = total_insured_person

                IR_ipd_summary = ((Reported_claims_ipd_summary / lives_ipd_opd) * 100).round(2)
                IR_ipd = IR_ipd_summary.astype(str) + '%'

                IR_opd_summary = ((Reported_claims_opd_summary / lives_ipd_opd) * 100).round(2)
                IR_opd = IR_opd_summary.astype(str) + '%'

                prorated_ir_ipd_summary = ((IR_ipd_summary * 365) / policy_coverage_completion_days).round(2)
                prorated_ir_ipd = prorated_ir_ipd_summary.astype(str) + '%'

                prorated_ir_opd_summary = ((IR_opd_summary * 365) / policy_coverage_completion_days).round(2)
                prorated_ir_opd = prorated_ir_opd_summary.astype(str) + '%'

                if total_premium != 0:
                    ICR_ipd_summary = ((int(incurred_amt_ipd_summary) / total_premium) * 100).round(2)
                else:
                    ICR_ipd_summary = 0


                ICR_ipd = str(ICR_ipd_summary) + '%'

                ICR_opd_summary = ((int(incurred_amt_opd_summary) / total_premium) * 100).round(2)
                ICR_opd = str(ICR_opd_summary) + '%'

                ECR_ipd_summary = ((int(incurred_amt_ipd_summary) / total_earned_premium) * 100).round(2)
                ECR_ipd = str(ECR_ipd_summary) + '%'

                ECR_opd_summary = ((int(incurred_amt_opd_summary) / total_earned_premium) * 100).round(2)
                ECR_opd = str(ECR_opd_summary) + '%'

                Settlement_ratio_ipd_summary = ((settled_claims_ipd_summary / Reported_claims_ipd_summary) * 100).round(
                    2)
                Settlement_ratio_ipd = str(Settlement_ratio_ipd_summary) + '%'

                Settlement_ratio_opd_summary = ((settled_claims_opd_summary / Reported_claims_opd_summary) * 100).round(
                    2)
                Settlement_ratio_opd = str(Settlement_ratio_opd_summary) + '%'

                ipd_summary_formatted = {'Reported_claims': Reported_claims_ipd_summary,
                                         'Reported_amt': Reported_amt_ipd,
                                         'settled_claims': settled_claims_ipd_summary, 'incurred_amt': incurred_amt_ipd,
                                         'paid_claims': paid_claims_ipd_summary, 'paid_amt': paid_amt_ipd,
                                         'IR': IR_ipd, 'prorated_ir': prorated_ir_ipd, 'ICR': ICR_ipd,
                                         'ECR': ECR_ipd, 'Settlement_ratio': Settlement_ratio_ipd}

                opd_summary_formatted = {'Reported_claims': Reported_claims_opd_summary,
                                         'Reported_amt': Reported_amt_opd,
                                         'settled_claims': settled_claims_opd_summary, 'incurred_amt': incurred_amt_opd,
                                         'paid_claims': paid_claims_opd_summary, 'paid_amt': paid_amt_opd,
                                         'IR': IR_opd, 'prorated_ir': prorated_ir_opd, 'ICR': ICR_opd,
                                         'ECR': ECR_opd, 'Settlement_ratio': Settlement_ratio_ipd}
                ipd_summary_formatted = pd.DataFrame(
                    zip(ipd_summary_formatted.keys(), (ipd_summary_formatted.values())), columns=['Parameters', 'IPD'])
                opd_summary_formatted = pd.DataFrame(
                    zip(opd_summary_formatted.keys(), (opd_summary_formatted.values())), columns=['Parameters', 'OPD'])

                pd.options.display.float_format = '{:.2f}'.format
                ipd_opd_summary = pd.merge(ipd_summary_formatted, opd_summary_formatted, how='inner', on='Parameters')

                ipd_opd_summary_fig = go.Figure(data=[go.Table(
                    header=dict(values=list(ipd_opd_summary.columns),
                                fill_color='#094780',
                                line_color='darkslategray',
                                # align='left',
                                font=dict(size=12, color='white')),
                    cells=dict(values=[ipd_opd_summary.Parameters, ipd_opd_summary.IPD, ipd_opd_summary.OPD],
                               fill_color='white',
                               line_color='darkslategray',
                               # align='left'
                               ))
                ])

                ipd_opd_summary_fig = plot(ipd_opd_summary_fig, output_type='div')

                ###############   INCIDENCE RATE BY SUM INSURED BAND ####################

                conditions1 = [(paid_df['sumins'] >= 0) & (paid_df['sumins'] <= 100000),
                               (paid_df['sumins'] >= 100000) & (paid_df['sumins'] <= 300000),
                               (paid_df['sumins'] >= 300000) & (paid_df['sumins'] <= 500000),
                               (paid_df['sumins'] >= 500000) & (paid_df['sumins'] <= 700000),
                               (paid_df['sumins'] >= 700000) & (paid_df['sumins'] <= 1000000),
                               (paid_df['sumins'] >= 1000000) & (paid_df['sumins'] <= 1500000)]

                conditions2 = [(crs_df['sumins'] >= 0) & (crs_df['sumins'] <= 100000),
                               (crs_df['sumins'] >= 100000) & (crs_df['sumins'] <= 300000),
                               (crs_df['sumins'] >= 300000) & (crs_df['sumins'] <= 500000),
                               (crs_df['sumins'] >= 500000) & (crs_df['sumins'] <= 700000),
                               (crs_df['sumins'] >= 700000) & (crs_df['sumins'] <= 1000000),
                               (crs_df['sumins'] >= 1000000) & (crs_df['sumins'] <= 1500000)]

                conditions3 = [(out_df['sumins'] >= 0) & (out_df['sumins'] <= 100000),
                               (out_df['sumins'] >= 100000) & (out_df['sumins'] <= 300000),
                               (out_df['sumins'] >= 300000) & (out_df['sumins'] <= 500000),
                               (out_df['sumins'] >= 500000) & (out_df['sumins'] <= 700000),
                               (out_df['sumins'] >= 700000) & (out_df['sumins'] <= 1000000),
                               (out_df['sumins'] >= 1000000) & (out_df['sumins'] <= 1500000)]

                # create a list of the values we want to assign for each condition
                values = ['00-01 Lakh', '01-03 Lakh', '03-05 Lakh', '05-07 Lakh', '07-10 Lakh', '10-15 Lakh']

                # create a new column and use np.select to assign values to it using our lists as arguments
                paid_df['Si_band'] = np.select(conditions1, values)

                crs_df['Si_band'] = np.select(conditions2, values)

                out_df['Si_band'] = np.select(conditions3, values)

                claim_paid_si_band = paid_df.groupby(['Si_band'], sort=False, as_index=False)['consider_count'].agg(
                    {'count': sum})
                claim_paid_si_band.rename(columns={'count': 'claim_paid_si_band'}, inplace=True)

                claim_crs_si_band = crs_df.groupby(['Si_band'], sort=False, as_index=False)['consider_count'].agg(
                    {'count': sum})
                claim_crs_si_band.rename(columns={'count': 'claim_crs_si_band'}, inplace=True)

                claim_out_si_band = out_df.groupby(['Si_band'], sort=False, as_index=False)['consider_count'].agg(
                    {'count': sum})
                claim_out_si_band.rename(columns={'count': 'claim_out_si_band'}, inplace=True)

                si_bandwise_incidence_rate1 = pd.merge(claim_crs_si_band, claim_paid_si_band, how='right',
                                                       on=['Si_band'])
                si_bandwise_incidence_rate = pd.merge(si_bandwise_incidence_rate1, claim_out_si_band, how='left',
                                                      on=['Si_band'])

                si_bandwise_incidence_rate['reported_claims'] = si_bandwise_incidence_rate['claim_crs_si_band'].fillna(
                    0) + si_bandwise_incidence_rate['claim_paid_si_band'].fillna(0) + si_bandwise_incidence_rate[
                                                                    'claim_out_si_band'].fillna(0)
                si_bandwise_incidence_rate['Lives'] = total_insured_person
                si_bandwise_incidence_rate['IR'] = si_bandwise_incidence_rate['reported_claims'] / \
                                                   si_bandwise_incidence_rate['Lives'] * 100

                # si_bandwise_incidence_rate.loc['Total']= si_bandwise_incidence_rate.sum(numeric_only=True, axis=0)
                si_bandwise_incidence_rate['IR'] = si_bandwise_incidence_rate['IR'].round(2)

                si_bandwise_incidence_rate.sort_values('Si_band', ascending=True, inplace=True)

                # Fig
                SI_Band_IR = px.bar(si_bandwise_incidence_rate, x="Si_band", y="IR",
                                    color="Si_band", title="<b><b>", text="IR",
                                    color_discrete_sequence=px.colors.sequential.RdBu)
                # SI_Band_IR.update_layout( xaxis={'visible': False, 'showticklabels': False})
                SI_Band_IR.update_layout(title_x=0.5)

                SI_Band_IR = plot(SI_Band_IR, output_type='div')

                # INCIDENCE RATE BY AMOUNT BAND

                claim_paid_amount_band = paid_df.groupby(['amount_band'], sort=False, as_index=False)[
                    'consider_count'].agg({'count': sum})
                claim_paid_amount_band.rename(columns={'count': 'claim_paid_amount_band'}, inplace=True)
                amount_bandwise_incidence_rate = claim_paid_amount_band.copy()

                amount_bandwise_incidence_rate['Lives'] = total_insured_person
                amount_bandwise_incidence_rate['IR'] = amount_bandwise_incidence_rate['claim_paid_amount_band'] / \
                                                       amount_bandwise_incidence_rate['Lives'] * 100
                # amount_bandwise_incidence_rate.loc['Total']= amount_bandwise_incidence_rate.sum(numeric_only=True, axis=0)

                amount_bandwise_incidence_rate['IR'] = amount_bandwise_incidence_rate['IR'].round(2)
                amount_bandwise_incidence_rate.sort_values('amount_band', ascending=True, inplace=True)

                Amount_Band_IR = px.bar(amount_bandwise_incidence_rate, x="amount_band", y="IR",
                                        color="amount_band", title="<b><b>", text="IR",
                                        color_discrete_sequence=px.colors.sequential.RdBu)
                # SI_Band_IR.update_layout( xaxis={'visible': False, 'showticklabels': False})
                Amount_Band_IR.update_layout(title_x=0.5)
                Amount_Band_IR = plot(Amount_Band_IR, output_type='div')

                # PAID AMOUNT BAND ANALYSIS

                paidamt_band_analysis = paid_df.groupby(['amount_band'])[['consider_count', 'settledamt']].agg(
                    'sum').reset_index()
                paidamt_band_analysis = paidamt_band_analysis.sort_values('settledamt', axis=0, ascending=False)[
                    ['amount_band', 'consider_count', 'settledamt']]
                paidamt_band_analysis['Avg_Paid_Amount'] = (
                        paidamt_band_analysis['settledamt'] / paidamt_band_analysis['consider_count']).round()
                paidamt_band_analysis.rename(columns={'consider_count': 'PaidClaims', 'settledamt': 'PaidAmt'},
                                             inplace=True)

                trace1 = go.Scatter(
                    mode='lines+markers',
                    x=paidamt_band_analysis['amount_band'],
                    y=paidamt_band_analysis['PaidClaims'], text=paidamt_band_analysis["PaidClaims"],
                    name="Paid Claims",
                    marker_color='crimson'
                )

                trace2 = go.Bar(
                    x=paidamt_band_analysis['amount_band'],
                    y=paidamt_band_analysis['PaidAmt'], text=paidamt_band_analysis["PaidAmt"],
                    name="Paid Amount",
                    yaxis='y2',
                    marker_color='blue',
                    marker_line_width=1.5,
                    marker_line_color='rgb(8,48,107)',
                    opacity=0.5
                )

                data = [trace1, trace2]

                layout = go.Layout(
                    title_text='<b><b>',
                    yaxis=dict(
                        range=[0, 200],
                        autorange=True,  # Allow autoscaling
                        side='right'
                    ),
                    yaxis2=dict(
                        overlaying='y',
                        anchor='y3',
                    )
                )
                paidamt_band_analysis_fig1 = go.Figure(data=data, layout=layout)
                paidamt_band_analysis_fig = plot(paidamt_band_analysis_fig1, output_type='div')

                ##Table
                paidamt_band_analysis1 = go.Figure(data=[go.Table(
                    header=dict(values=paidamt_band_analysis.columns,
                                fill_color='#094780',
                                line_color='darkslategray',
                                align='left',
                                font=dict(size=12, color='white')),
                    cells=dict(values=[paidamt_band_analysis.amount_band, paidamt_band_analysis.PaidClaims,
                                       paidamt_band_analysis.PaidAmt, paidamt_band_analysis.Avg_Paid_Amount],
                               fill_color='white',
                               line_color='darkslategray',
                               align='left'))])

                paidamt_band_analysis1 = plot(paidamt_band_analysis1, output_type='div')

                # STATUS WISE ANALYSIS

                ipd_opd_wise_paid_claims = paid_df.groupby(['status', 'ipd_opd'])[
                    'consider_count'].sum().reset_index()

                ipd_opd_wise_out_claims = out_df.groupby(['status', 'ipd_opd'])['consider_count'].sum().reset_index()

                ipd_opd_wise_crs_claims = crs_df.groupby(['status', 'ipd_opd'])['consider_count'].sum().reset_index()

                ipd_opd_wise_claims = pd.concat(
                    [ipd_opd_wise_paid_claims, ipd_opd_wise_out_claims, ipd_opd_wise_crs_claims])
                ipd_opd_wise_claims['consider_count'] = ipd_opd_wise_claims['consider_count'].astype(int)

                ipd_opd_wise_claims_fig = px.bar(ipd_opd_wise_claims, x="ipd_opd", y="consider_count",
                                                 color="status", title="<b><b>",
                                                 text='consider_count',
                                                 color_discrete_sequence=px.colors.sequential.RdBu, barmode='group',
                                                 color_discrete_map={'Paid': 'green', 'Outstanding': '#D9B300',
                                                                     'CRS': 'red'})
                ipd_opd_wise_claims_fig.update_xaxes(title='')
                ipd_opd_wise_claims_fig.update_yaxes(title='No. of Claims')
                ipd_opd_wise_claims_fig.update_layout(title_x=0.5)
                ipd_opd_wise_claims_fig = plot(ipd_opd_wise_claims_fig, output_type='div')

                # DISEASE WISE ANALYSIS BY BUFFER AMOUNT

                disease_wise_buffer_amount = paid_df.groupby(['new_disease_category']).agg(
                    {'buffer_amt': sum}).reset_index()
                disease_wise_buffer_amount2 = disease_wise_buffer_amount[disease_wise_buffer_amount['buffer_amt'] != 0]

                disease_wise_buffer_amount_fig2 = px.bar(disease_wise_buffer_amount2, x='new_disease_category',
                                                         y="buffer_amt", text='buffer_amt',
                                                         color_discrete_sequence=px.colors.sequential.RdBu,
                                                         color='new_disease_category', barmode='stack')

                disease_wise_buffer_amount_fig2.update_layout(title='<b><b>')

                disease_wise_buffer_amount_fig2.update_layout(title_x=0.5)
                disease_wise_buffer_amount_fig2.update_xaxes(title='')

                disease_wise_buffer_amount_fig2 = plot(disease_wise_buffer_amount_fig2, output_type='div')

                # Buffer Amt vs Liablity Amt

                Buffer_Amt = paid_df['buffer_amt'].sum()
                Liab_Amt = paid_df['liablityamt'].sum()

                labels = ['Amt to Release', 'Liablity Amt']
                values = [Buffer_Amt, Liab_Amt]

                data = [go.Pie(labels=labels, values=values, text=values)]
                layout = go.Layout(title='')
                buffer_amt_vs_liab_amt = go.Figure(data=data, layout=layout)

                buffer_amt_vs_liab_amt = plot(buffer_amt_vs_liab_amt, output_type='div')

                ## CUSTOMER SERVICE ANALYSIS (CC)

                customer_touch_point1 = customercare_df['sourcefield'].value_counts()

                customer_touch_point2 = customercare_df['sourcefield'].value_counts(normalize=0.1) * 100

                customer_touch_point = pd.concat([customer_touch_point1, customer_touch_point2], axis=1)

                customer_touch_point.columns = ['No Of Interactions', 'No Of Interactions%']
                customer_touch_point = customer_touch_point.reset_index()

                customer_touch_point.columns = ['SourceField', 'No Of Interactions', 'No Of Interactions%']

                # figure for plotly
                customer_touch_point_fig1 = go.Figure(data=[go.Table(
                    header=dict(values=customer_touch_point.columns,
                                fill_color='#094780',
                                line_color='darkslategray',
                                align='left',
                                font=dict(size=12, color='white')),
                    cells=dict(values=[customer_touch_point['SourceField'],
                                       customer_touch_point['No Of Interactions'],
                                       customer_touch_point['No Of Interactions%'].round()],
                               fill_color='white',
                               line_color='darkslategray',
                               align='left'))])

                customer_touch_point_fig2 = px.bar(customer_touch_point, x='SourceField', y="No Of Interactions",
                                                   text='No Of Interactions',
                                                   color_discrete_sequence=px.colors.sequential.RdBu,
                                                   color='SourceField', barmode='stack')

                customer_touch_point_fig2.update_layout(title='<b><b>')

                customer_touch_point_fig2.update_layout(title_x=0.5)

                customer_touch_point_fig2 = plot(customer_touch_point_fig2, output_type='div')

                customer_touch_point_fig1 = plot(customer_touch_point_fig1, output_type='div')

                ## Customer Call Analysis

                customer_call_analysis_point1 = customercare_df['call_type'].value_counts()

                customer_call_analysis_point2 = customercare_df['call_type'].value_counts(normalize=0.1) * 100

                customer_call_analysis_point = pd.concat([customer_call_analysis_point1, customer_call_analysis_point2],
                                                         axis=1)
                customer_call_analysis_point.columns = ['No Of Calls', 'No Of Calls%']
                customer_call_analysis_point = customer_call_analysis_point.reset_index()

                customer_call_analysis_point.columns = ['Call_Type', 'No Of Calls', 'No Of Calls%']

                # figure for plotly
                customer_call_analysis_fig1 = go.Figure(data=[go.Table(
                    header=dict(values=customer_call_analysis_point.columns,
                                fill_color='#094780',
                                line_color='darkslategray',
                                align='left',
                                font=dict(size=12, color='white')),
                    cells=dict(values=[customer_call_analysis_point['Call_Type'],
                                       customer_call_analysis_point['No Of Calls'],
                                       customer_call_analysis_point['No Of Calls%'].round()],
                               fill_color='white',
                               line_color='darkslategray',
                               align='left'))])

                customer_call_analysis_fig2 = px.bar(customer_call_analysis_point, x='Call_Type', y="No Of Calls",
                                                     text='No Of Calls',
                                                     color_discrete_sequence=px.colors.sequential.RdBu,
                                                     color='Call_Type', barmode='stack')

                customer_call_analysis_fig2.update_layout(title='<b><b>')

                customer_call_analysis_fig2.update_layout(title_x=0.5)
                customer_call_analysis_fig2.update_xaxes(title='')

                customer_call_analysis_fig2 = plot(customer_call_analysis_fig2, output_type='div')

                customer_call_analysis_fig1 = plot(customer_call_analysis_fig1, output_type='div')

                # Reason for call Analysis

                customercare_df['call_reason_count'] = customercare_df.groupby('reasonforcall')[
                    'reasonforcall'].transform(
                    'count')
                customercare_df['call_reason_count'] = \
                    customercare_df.drop_duplicates(['reasonforcall', 'call_reason_count'])['call_reason_count']
                ReasonForCall_analysis = customercare_df[['reasonforcall', 'call_reason_count']].sort_values(
                    'call_reason_count', ascending=False).head(10)
                ReasonForCall_analysis['Total%'] = ReasonForCall_analysis['call_reason_count'] / ReasonForCall_analysis[
                    'call_reason_count'].sum() * 100
                ReasonForCall_analysis.columns = ['ReasonForCall', 'No Of Calls', 'No Of Calls%']

                # figure for plotly
                ReasonForCall_analysis_fig1 = go.Figure(data=[go.Table(
                    header=dict(values=ReasonForCall_analysis.columns,
                                fill_color='#094780',
                                line_color='darkslategray',
                                align='left',
                                font=dict(size=12, color='white')),
                    cells=dict(values=[ReasonForCall_analysis['ReasonForCall'],
                                       ReasonForCall_analysis['No Of Calls'],
                                       ReasonForCall_analysis['No Of Calls%'].round()],
                               fill_color='white',
                               line_color='darkslategray',
                               align='left'))])

                ReasonForCall_analysis_fig2 = px.bar(ReasonForCall_analysis, x='ReasonForCall', y="No Of Calls",
                                                     text='No Of Calls',
                                                     color_discrete_sequence=px.colors.sequential.RdBu,
                                                     color='ReasonForCall', barmode='stack')

                ReasonForCall_analysis_fig2.update_layout(title='<b><b>')

                ReasonForCall_analysis_fig2.update_layout(title_x=0.5)
                ReasonForCall_analysis_fig2.update_xaxes(title='')

                ReasonForCall_analysis_fig2 = plot(ReasonForCall_analysis_fig2, output_type='div')

                ReasonForCall_analysis_fig1 = plot(ReasonForCall_analysis_fig1, output_type='div')

                # GRIEVANCE ANALYSIS

                griveance = customercare_df[(customercare_df['call_type'] == 'Grievance')]

                griveance['response_given_count'] = griveance.groupby('call_type')['call_type'].transform('count')
                griveance['response_given_count'] = griveance.drop_duplicates(['call_type', 'response_given_count'])[
                    'response_given_count']

                griveance['response_given_count1'] = griveance.groupby('reasonforcall')['reasonforcall'].transform(
                    'count')
                griveance['response_given_count1'] = \
                    griveance.drop_duplicates(['reasonforcall', 'response_given_count1'])['response_given_count1']

                griveance_analysis = griveance[['reasonforcall', 'response_given_count1']].sort_values(
                    'response_given_count1', ascending=False).drop_duplicates().head(10)

                griveance_analysis['Total%'] = griveance_analysis['response_given_count1'] / griveance_analysis[
                    'response_given_count1'].sum() * 100
                griveance_analysis.columns = ['ReasonForCall', 'No Of Calls', 'Griveances%']

                # figure for plotly
                griveance_analysis_fig1 = go.Figure(data=[go.Table(
                    header=dict(values=griveance_analysis.columns,
                                fill_color='#094780',
                                line_color='darkslategray',
                                align='left',
                                font=dict(size=12, color='white')),
                    cells=dict(values=[griveance_analysis['ReasonForCall'],
                                       griveance_analysis['No Of Calls'],
                                       griveance_analysis['Griveances%'].round()],
                               fill_color='white',
                               line_color='darkslategray',
                               align='left'))])

                griveance_analysis_fig2 = px.bar(griveance_analysis, x='ReasonForCall', y="No Of Calls",
                                                 text='No Of Calls', color_discrete_sequence=px.colors.sequential.RdBu,
                                                 color='ReasonForCall', barmode='stack')

                griveance_analysis_fig2.update_layout(title='<b><b>')

                griveance_analysis_fig2.update_layout(title_x=0.5)
                griveance_analysis_fig2.update_xaxes(title='')

                griveance_analysis_fig2 = plot(griveance_analysis_fig2, output_type='div')

                griveance_analysis_fig1 = plot(griveance_analysis_fig1, output_type='div')


            else:
                messages.warning(request, "Apparently no values available...")

        mydict = {
            'form': form,
            'insured_name1':insured_name1,
            'paid_claim_count': paid_claim_count,
            'out_claim_count': out_claim_count,
            'crs_claim_count': crs_claim_count,
            'premium': premium,
            'lives': lives,
            'lives_by_ip_relation_code_fig_1': lives_by_ip_relation_code_fig_1,
            'age_band_relationship_wise_insured_lives_fig2': age_band_relationship_wise_insured_lives_fig2,
            'age_band_relationship_wise_insured_lives_fig1': age_band_relationship_wise_insured_lives_fig1,
            'claim_status_fig1': claim_status_fig1,
            'ALL_CLAIMS_BY_CLAIM_STATUS_fig': ALL_CLAIMS_BY_CLAIM_STATUS_fig,
            'lives_by_age_band_fig1': lives_by_age_band_fig1,
            'relation_age_band_wise_paid_amt_fig_1': relation_age_band_wise_paid_amt_fig_1,
            'relation_age_band_wise_paid_amt_fig_2': relation_age_band_wise_paid_amt_fig_2,
            'outstanding_claim_analysis_fig': outstanding_claim_analysis_fig,
            'outstanding_claim_analysis_tb': outstanding_claim_analysis_tb,
            'rejected_claim_breakup_fig': rejected_claim_breakup_fig,
            'rejected_claim_breakup_tb': rejected_claim_breakup_tb,
            'treatment_type_wise_analysis_fig': treatment_type_wise_analysis_fig,
            'treatment_type_wise_analysis_tb': treatment_type_wise_analysis_tb,
            'cashless_vs_reimbersement_claim_count_paid_fig': cashless_vs_reimbersement_claim_count_paid_fig,
            'cashless_vs_reimbersement_claim_amt_paid_fig': cashless_vs_reimbersement_claim_amt_paid_fig,
            'cashless_vs_reimbersement_lodge_amount_fig': cashless_vs_reimbersement_lodge_amount_fig,
            'summary1': summary1,
            'Summary_plot': Summary_plot,
            'Summary_plot2': Summary_plot2,
            'ipd_opd_summary_fig': ipd_opd_summary_fig,
            'top10_ailment_wise_analysis_fig1': top10_ailment_wise_analysis_fig1,
            'top10_ailment_wise_analysis_fig2': top10_ailment_wise_analysis_fig2,
            'top10_hospital_wise_paidamt_fig1': top10_hospital_wise_paidamt_fig1,
            'top10_hospital_wise_paidamt_fig2': top10_hospital_wise_paidamt_fig2,
            'Age_Band_IR': Age_Band_IR,
            'SI_Band_IR': SI_Band_IR,
            'Amount_Band_IR': Amount_Band_IR,
            'paidamt_band_analysis_fig': paidamt_band_analysis_fig,
            'paidamt_band_analysis1': paidamt_band_analysis1,
            'ipd_opd_wise_claims_fig': ipd_opd_wise_claims_fig,
            'disease_wise_buffer_amount_fig2': disease_wise_buffer_amount_fig2,
            'buffer_amt_vs_liab_amt': buffer_amt_vs_liab_amt,
            'customer_touch_point_fig2': customer_touch_point_fig2,
            'customer_touch_point_fig1': customer_touch_point_fig1,
            'customer_call_analysis_fig2': customer_call_analysis_fig2,
            'customer_call_analysis_fig1': customer_call_analysis_fig1,
            'ReasonForCall_analysis_fig2': ReasonForCall_analysis_fig2,
            'ReasonForCall_analysis_fig1': ReasonForCall_analysis_fig1,
            'griveance_analysis_fig2': griveance_analysis_fig2,
            'griveance_analysis_fig1': griveance_analysis_fig1,
        }

        return render(request, 'Management/corporate.html', context=mydict)




class Corporate1(View):
    def get(self,request,*args,**kwargs):


        pol_no = request.GET.get('polno')
        pol_no = '910000/34/22/04/00000132'

        # get data from multiple tables
        paid_data, crs_data, out_data = self.get_table_data(pol_no)

        # paid_data = self.get_table_data( pol_no)
        # crs_data = self.get_table_data( pol_no)
        # out_data = self.get_table_data( pol_no)
        portal_premium = self.premium_lives(pol_no)

        # create pandas data frames from each table's data
        paid_df = pd.DataFrame(paid_data)
        print(paid_df.columns)
        crs_df = pd.DataFrame(crs_data)

        out_df = pd.DataFrame(out_data)
        portal_premium_df = pd.DataFrame(portal_premium)
        relation_qs = RelationMaster.objects.all()
        relation_df = pd.DataFrame(relation_qs.values())

        portal_premium_df = pd.merge(portal_premium_df, relation_df, how='left', left_on='ip_relation_code',
                                     right_on='relation').drop(columns=['id', 'relation'])
        done = []
        not_done = []
        for col in portal_premium_df.columns:
            try:
                portal_premium_df[col] = portal_premium_df[col].str.strip()
                done.append(col)
            except:
                not_done.append(col)
        portal_premium_df['Premium'] = portal_premium_df['NET_PREMIUM'].fillna(0) + portal_premium_df[
            'PREMIUM_ENDORSEMENT'].fillna(0)
        relation_df['relation'] = relation_df['relation'].str.strip()
        # Count for all Paid, Crs, Outstanding ,lives, premium
        paid_claim_count = paid_df['sla_heading'].count()
        out_claim_count = out_df['sla_heading'].count()
        crs_claim_count = crs_df['sla_heading'].count()
        premium = int(round((portal_premium_df['Premium'].mean()), 0))
        lives = round((portal_premium_df['lives'].sum()), 0)


        # Relationship Wise lives
        relation_wise_lives_covered = portal_premium_df.groupby(['std_relation'])['lives'].agg(
            'sum').reset_index()

        relation_wise_lives_covered_data = [go.Pie(labels=relation_wise_lives_covered['std_relation'],
                       values=relation_wise_lives_covered['lives'], text=relation_wise_lives_covered['lives'],
                       hole=.5, marker_colors=px.colors.qualitative.Plotly)
                ]
        lives_by_ip_relation_code_fig = go.Figure(data=relation_wise_lives_covered_data)


        # lives_by_ip_relation_code_fig_1 = plot(lives_by_ip_relation_code_fig, output_type='div', config=config)


        # # create data visualizations using Plotly
        # pie_chart1 = self.create_pie_chart(df1, 'column1', 'column2')

        customercare_qs = Customercare.objects.filter(pol_no=pol_no)
        relation_qs = RelationMaster.objects.all()

        # create data visualizations using Plotly
        lives_by_ip_relation_code_fig_1 = self.create_pie_chart(relation_wise_lives_covered_data,lives_by_ip_relation_code_fig)


        mydict ={
            'lives_by_ip_relation_code_fig_1':lives_by_ip_relation_code_fig_1,
        }

        return render(request, 'Management/corporate1.html', context=mydict)

    def get_table_data(self,pol_no):
        # filter data from table based on user input
        paid_data = ClPaidDet2.objects.filter(pol_no=pol_no).values('pol_no', 'sla_heading', 'insuredname',
                                                                       'relation', 'lodgetype', 'status',
                                                                       'actuallosstype', 'actual_lodge_amt',
                                                                       'settledamt', 'liablityamt', 'consider_count',
                                                                       'ipd_opd', 'age_band_rev',
                                                                       'new_disease_category', 'lodgedate', 'dod',
                                                                       'lodgeamt', 'treatmenttype', 'sumins',
                                                                       'amount_band', 'diseasecategory', 'hospitlname',
                                                                       'utilizationband', 'buffer_amt', 'payment_tat',
                                                                       'end_to_end_tat',
                                                                       'final_processing_tat_settlment_tat',
                                                                       'adr_raise_date_lodge_date',
                                                                       'adr_recive_adr_raise_date')

        crs_data = ClCrsDet2.objects.filter(pol_no=pol_no).values('status', 'substatus', 'lodgedate', 'lodgetype',
                                                                'cl_lod_amt', 'doa', 'dod', 'sla_heading',
                                                                'consider_count', 'actuallosstype', 'age_band_rev',
                                                                'ipd_opd', 'sumins', 'actual_lodge_amt')

        out_data = ClOutDet2.objects.filter(pol_no=pol_no).values('status', 'extra', 'head', 'sla_heading',
                                                                'consider_count', 'age_band_rev', 'liablityamt',
                                                                'actuallosstype', 'lodgetype', 'lodgedate', 'doa',
                                                                'dod', 'ipd_opd', 'sumins', 'actual_lodge_amt',
                                                                'actuallodgeamt')


        return paid_data,crs_data,out_data

    def create_pie_chart(self,data,fig1):
        layout = go.Layout()
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(margin=dict(t=100, b=100, l=100, r=100),
                                                    plot_bgcolor="rgba(0,0,0,0)",
                                                    title='')
        fig = plot(fig1, output_type='div', config=config)
        return fig

    def create_bar_chart(self, df, x_column, y_column):
        # create bar chart using Plotly
        fig = px.bar(df, x=x_column, y=y_column)

        return fig

    def create_donut_chart(self, df, x_column, y_column):
        # create donut chart using Plotly
        fig = px.pie(df, values=y_column, names=x_column, hole=.5)

        return fig

    def premium_lives(self,pol_no):

        portal_premium_qs = pd.read_sql_query(f"""SELECT
                                    A.pol_no,A.name_of_insured, A.ic_name, ip_relation_code, ip_age, broker_name, risk_from_date,risk_expiry_date,
                                    CASE WHEN RISK_EXPIRY_DATE<=GETDATE() THEN '365'
                                    ELSE CASE WHEN  RISK_EXPIRY_DATE>GETDATE() THEN DATEDIFF(DAY,RISK_FROM_DATE,GETDATE())+1 ELSE 0 END
                                    END AS [POLICYRUNDAY],
                                    CONVERT(VARCHAR(10),GETDATE(),103)+' (TIME: '+SUBSTRING(CAST(GETDATE()AS VARCHAR(30)),13,7)+')' AS REPORT_DATE,
                                    ISNULL(A.NET_PREMIUM,0) AS NET_PREMIUM,
                                    ISNULL(B.PREMIUM_ENDORSEMENT,0) AS PREMIUM_ENDORSEMENT,
                                    CASE WHEN ISNULL(RISK_EXPIRY_DATE,'')<=ISNULL(GETDATE(),'') THEN  ( ISNULL(A.NET_PREMIUM,0)+ISNULL(B.PREMIUM_ENDORSEMENT,0))
                                    ELSE ((ISNULL(A.NET_PREMIUM,0)+ISNULL(B.PREMIUM_ENDORSEMENT,0))/365)*DATEDIFF(DAY,RISK_FROM_DATE,GETDATE())+1 END AS [EARNED_PERMIUM],
                                    ISNULL(A.NO_OF_EMPLOYEES_COVERED,0) AS NO_OF_EMPLOYEES_COVERED,
                                    ISNULL(A.NO_OF_DEPENDANTS_COVERED,0) AS NO_OF_DEPENDANTS_COVERED,
                    	            NO_OF_EMPLOYEES_COVERED + NO_OF_DEPENDANTS_COVERED AS lives,NET_PREMIUM + CAST(PREMIUM_ENDORSEMENT AS int) As premium
                                    FROM (SELECT IC_NAME,TEMP.POL_NO, IP_Relation_Code, IP_Age, Broker_Name,NAME_OF_INSURED,

                                        MIN(TEMP.RISK_FROM_DATE) AS RISK_FROM_DATE,
                                        MAX(TEMP.RISK_EXPIRY_DATE) AS RISK_EXPIRY_DATE,MAX(TEMP.NET_PREMIUM) AS NET_PREMIUM,
                                        SUM(NO_OF_EMPLOYEES_COVERED) AS NO_OF_EMPLOYEES_COVERED, SUM(NO_OF_DEPENDANTS_COVERED) AS NO_OF_DEPENDANTS_COVERED
                                        FROM
                                        (SELECT IC_NAME,POL_NO, IP_Relation_Code, IP_Age, Broker_Name, NAME_OF_INSURED,
                                            MIN(RISK_FROM_DATE) AS RISK_FROM_DATE,
                                            MAX(RISK_EXPIRY_DATE) AS RISK_EXPIRY_DATE,
                                            MAX(PREMIUM) AS NET_PREMIUM,
                                            SUM(CASE WHEN ((IP_RELATION_CODE = 'SELF' OR IP_RELATION_CODE = 'EMPLOYEE')
                                            AND ISNULL(IP_CANCEL_DATE,'')='' ) THEN 1 ELSE 0 END) AS NO_OF_EMPLOYEES_COVERED,
                                            SUM(CASE WHEN ISNULL(IP_CANCEL_DATE,'')=''  THEN 1 ELSE 0 END ) -
                                            SUM(CASE WHEN (IP_RELATION_CODE IN ('SELF','EMPLOYEE')
                                            AND ISNULL(IP_CANCEL_DATE,'')='')  THEN 1 ELSE 0 END)  AS NO_OF_DEPENDANTS_COVERED
                                            FROM ENROLLMENT_MASTER WHERE Pol_No = '{pol_no}' AND Pol_Status = 'ENFORCED'
                                            GROUP BY POL_NO,IC_NAME, Broker_Name,NAME_OF_INSURED, IP_Relation_Code, IP_Age)
                                            AS TEMP GROUP BY TEMP.POL_NO,IC_NAME, Broker_Name,NAME_OF_INSURED, IP_Relation_Code, IP_Age)
                                            AS A LEFT OUTER JOIN(SELECT POL_NO,(SUM(ISNULL(AMTADD,0)) - SUM(ISNULL(AMTRED,0)) ) AS PREMIUM_ENDORSEMENT FROM DBO.[ENDORS]
                                            WHERE Pol_No = '{pol_no}'
                                            GROUP BY POL_NO) AS B ON A.Pol_No = B.Pol_No """, connection)

        return portal_premium_qs


def get_icname():
    icname = ClPaidDet2.objects.values_list('icname',flat=True).distinct()
    return icname


def insurance_company(request):
    if request.user.is_authenticated:
        # icnames = get_icname()
        altualloss_data = None
        altualloss_label = None
        FY21_22_Claim_Status = None
        ir = None
        icr = None
        premium = None
        lives = None
        paid_count = None
        out_count = None
        crs_count = None
        Covid_Claims = None
        Non_Covid_Claims = None
        master_stats1 = None

        if request.method == 'POST':

            insurance_company = request.POST.get('insurance_company')
            FromDate = request.POST.get('FromDate')
            ToDate = request.POST.get('ToDate')

            # beg_previous1 = datetime.strptime(FromDate, '%YYYY-%MM')
            # close_previous = datetime.strptime(ToDate, '%YYYY-%')
            beg_previous = pd.to_datetime(FromDate)
            if beg_previous.month != 1:
                try:
                    beg_previous = datetime(beg_previous.year, (beg_previous.month - 1), beg_previous.day)
                except:
                    try:
                        beg_previous = datetime(beg_previous.year, (beg_previous.month - 1), (beg_previous.day - 1))
                    except:
                        try:
                            beg_previous = datetime(beg_previous.year, (beg_previous.month - 1), (beg_previous.day - 2))
                        except:
                            beg_previous = datetime(beg_previous.year, (beg_previous.month - 1), (beg_previous.day - 3))

            else:
                beg_previous = datetime((beg_previous.year - 1), 12, beg_previous.day)


            # print(close_previous)

            openos_qs = ClOutDet2.objects.filter(Q(ic_name=insurance_company) & Q(sla_heading_updated__range=[beg_previous,FromDate]))
            paid_qs = ClPaidDet2.objects.filter(Q(icname=insurance_company) & Q(sla_heading_updated__range=[FromDate, ToDate]))
            out_qs = ClOutDet2.objects.filter(Q(ic_name=insurance_company) & Q(sla_heading_updated__range=[FromDate,ToDate]))
            crs_qs = ClCrsDet2.objects.filter(Q(ic_name=insurance_company) & Q(sla_heading_updated__range=[FromDate,ToDate]))
            portal_premium_qs = EFinance.objects.filter(Q(insurer=insurance_company) & Q(sla_heading_updated__range=[FromDate,ToDate]))
            if ((len(paid_qs) > 0) and (len(out_qs) > 0) and (len(crs_qs) > 0) and (len(openos_qs) > 0) and (len(portal_premium_qs) > 0)):

                paid_df = pd.DataFrame(paid_qs.values())
                out_df = pd.DataFrame(out_qs.values())
                crs_df = pd.DataFrame(crs_qs.values())
                portal_premium_df1 = pd.DataFrame(portal_premium_qs.values())
                portal_premium_df = portal_premium_df1.drop_duplicates(subset=['pol_no','premium'],keep='last').reset_index(drop = True)

                beg_os_df = pd.DataFrame(openos_qs.values())
                print(portal_premium_df)
                ## Count for all Paid, Crs, Outstanding
                paid_count = paid_df['sla_heading_updated'].count()
                out_count = out_df['sla_heading_updated'].count()
                crs_count = crs_df['sla_heading_updated'].count()

                premium = round((portal_premium_df['premium'].mean()),0)
                lives = round((portal_premium_df['lives'].sum()),0)
                paid_df['ro_name'] = paid_df['ro_name'].str.strip().str.upper()
                out_df['ro_name'] = out_df['ro_name'].str.strip().str.upper()
                crs_df['ro_name'] = crs_df['ro_name'].str.strip().str.upper()
                portal_premium_df['ro_name'] = portal_premium_df['ro_name'].str.strip().str.upper()
                beg_os_df['ro_name'] = beg_os_df['ro_name'].str.strip().str.upper()
                paid_df['icname'] = paid_df['icname'].str.strip().str.upper()
                out_df['ic_name'] = out_df['ic_name'].str.strip().str.upper()
                crs_df['ic_name'] = crs_df['ic_name'].str.strip().str.upper()
                beg_os_df['ic_name'] = crs_df['ic_name'].str.strip().str.upper()
                paid_df['policy_plan_type'] = paid_df['policy_plan_type'].str.strip()
                out_df['policy_plan_type'] = out_df['policy_plan_type'].str.strip()
                crs_df['policy_plan_type'] = crs_df['policy_plan_type'].str.strip()
                beg_os_df['policy_plan_type'] = crs_df['policy_plan_type'].str.strip()
                paid_df['clienttype'] = paid_df['clienttype'].str.strip()
                out_df['clienttype'] = out_df['clienttype'].str.strip()
                crs_df['clienttype'] = crs_df['clienttype'].str.strip()
                beg_os_df['clienttype'] = crs_df['clienttype'].str.strip()
                paid_df['lodgetype'] = paid_df['lodgetype'].str.strip()
                out_df['lodgetype'] = out_df['lodgetype'].str.strip()
                crs_df['lodgetype'] = crs_df['lodgetype'].str.strip()
                beg_os_df['lodgetype'] = crs_df['lodgetype'].str.strip()
                paid_df['report_plan'] = paid_df['report_plan'].str.strip()
                out_df['report_plan'] = out_df['report_plan'].str.strip()
                crs_df['report_plan'] = crs_df['report_plan'].str.strip()
                beg_os_df['report_plan'] = crs_df['report_plan'].str.strip()




                paid_df['ro_name_month_year'] = paid_df['icname'] + '-' + paid_df['ro_name'] + '-' + paid_df['sla_heading_updated']

                out_df['ro_name_month_year'] = out_df['ic_name'] + '-' + out_df['ro_name'] + '-' + out_df['sla_heading_updated']

                crs_df['ro_name_month_year'] = crs_df['ic_name'] + '-' + crs_df['ro_name'] + '-' + crs_df['sla_heading_updated']

                beg_os_df['ro_name_month_year'] = beg_os_df['ic_name'] + '-' + beg_os_df['ro_name'] + '-' + beg_os_df['sla_heading_updated']


                portal_premium_df['ic_ro_name_month_year'] = portal_premium_df['ic_ro_name_month_year'].str.upper()




                ### Cashless and NonCashless
                altualloss = paid_df['actuallosstype'].value_counts()
                actual_loss = pd.DataFrame(altualloss).reset_index()
                alt_index = actual_loss.rename(columns={'index': 'alt_index'})
                altualloss_label = alt_index['alt_index'].tolist()
                altualloss_data = actual_loss['actuallosstype'].tolist()


                ### Covid NoN- Covid



                # Claim Status

                Covid_Paid = sum((paid_df['status'] == 'Paid') & (paid_df['covid_non_covid'] == 'Covid') & (
                            paid_df['consider_count'] == 1))

                Covid_Outstanding = sum((out_df['status'] == 'Outstanding') & (
                            out_df['covid_non_covid'] == 'Covid') & (out_df['consider_count'] == 1))

                Covid_Rejected = sum((crs_df['substatus'] == 'Claim Repudiation') & (
                            crs_df['covid_non_covid'] == 'Covid') & (crs_df['consider_count'] == 1))

                Covid_Closed = sum((crs_df['substatus'] == 'Claim Intimation Closed') & (
                            crs_df['covid_non_covid'] == 'Covid') & (crs_df['consider_count'] == 1))



                Non_Covid_Paid = sum((paid_df['status'] == 'Paid') & (paid_df['covid_non_covid'] == 'Non Covid') & (
                            paid_df['consider_count'] == 1))

                Non_Covid_Outstanding = sum((out_df['status'] == 'Outstanding') & (
                            out_df['covid_non_covid'] == 'Non Covid') & (out_df['consider_count'] == 1))

                Non_Covid_Rejected = sum((crs_df['substatus'] == 'Claim Repudiation') & (
                            crs_df['covid_non_covid'] == 'Non Covid') & (crs_df['consider_count'] == 1))

                Non_Covid_Closed = sum((crs_df['substatus'] == 'Claim Intimation Closed') & (
                            crs_df['covid_non_covid'] == 'Non Covid') & (crs_df['consider_count'] == 1))

                FY21_22_Claim_Status = pd.DataFrame([[Covid_Paid, Covid_Outstanding, Covid_Rejected, Covid_Closed],
                                                     [Non_Covid_Paid, Non_Covid_Outstanding, Non_Covid_Rejected,
                                                      Non_Covid_Closed]],
                                                    columns=['Paid', 'Outstanding', 'Rejected', 'Closed'],
                                                    index=['Covid', 'Non_Covid'])


                print(FY21_22_Claim_Status)

                # Covid_Non Covid_Claims

                # Covid_Claims = sum(((paid_df['covid_non_covid'] == 'Covid') & (paid_df['consider_count'] == 1)) ||  ((crs_df['covid_non_covid'] == 'Covid') & (crs_df['consider_count'] == 1))
                #                     ||((out_df['covid_non_covid'] == 'Covid') & (out_df['consider_count'] == 1)))
                #
                #
                # Non_Covid_Claims = sum(((paid_df['covid_non_covid'] == 'Non Covid') & (paid_df['consider_count'] == 1)) &  ((crs_df['covid_non_covid'] == 'Non Covid') & (crs_df['consider_count'] == 1))
                #                    & ((out_df['covid_non_covid'] == 'Non Covid') & (out_df['consider_count'] == 1)))

                Covid_Claims = Covid_Paid + Covid_Outstanding + Covid_Rejected + Covid_Closed
                Non_Covid_Claims = Non_Covid_Paid + Non_Covid_Outstanding + Non_Covid_Rejected + Non_Covid_Closed
                print(Covid_Claims)
                print(Non_Covid_Claims)



        mydict = {}
        return render(request, 'Management/insurance_company.html',context=mydict)

def getbrokerdistinctpol():
    pol_no = EnrollmentMaster.objects.values_list('pol_no', flat=True).distinct()
    return pol_no


def getbrokerdistinctname():
    brokername = BrokerName.objects.all()
    return brokername


def broker(request):
    if request.user.is_authenticated:
        broker_name = BrokerName.objects.all()

        if request.method == 'POST':
            brokername = request.POST.get('brokername')
            pol_numbers = EnrollmentMaster.objects.filter(Q(broker_name__contains=brokername)
                                                          & Q(premium__gt=100)
                                                          & Q(pol_status='ENFORCED')).values_list('pol_no', flat=True).distinct()


            Pol_N0 = tuple(pol_numbers)

            Policy_Master = pd.read_sql_query(f"""
            select distinct Pol_No,  name_of_insured,IC_Name,Risk_From_Date, Risk_Expiry_Date
                                 FROM [Enrollment].[dbo].[Enrollment_Master]
                                 where pol_no IN {Pol_N0}
                 """, connection)

            if len(pol_numbers) > 0:
                paid = pd.read_sql_query(f""" SELECT pol_no FROM [Claims_SLA].[dbo].[CL_PAID_DET2]                       
                WHERE Pol_No in {Pol_N0} """, connection)

                crs = pd.read_sql_query(f""" SELECT pol_no FROM [Claims_SLA].[dbo].[CL_CRS_DET2] 
                WHERE Pol_No in {Pol_N0}""", connection)

                out = pd.read_sql_query(
                    f""" SELECT pol_no FROM [Claims_SLA].[dbo].[CL_OUT_DET2] WHERE Pol_No in {Pol_N0} AND SLA_HEADING = 'Dec-2022'""",
                    connection)

                customer_care = pd.read_sql_query(f""" SELECT pol_no FROM [Claims_SLA].[dbo].[CustomerCare]  WHERE Pol_No in {Pol_N0} """,
                                                  connection)
                # paid_qs = ClPaidDet2.objects.filter(pol_no__in=Subquery(pol_numbers))
                # crs_qs = ClCrsDet2.objects.filter(pol_no__in=Subquery(pol_numbers))
                # out_qs = ClOutDet2.objects.filter(pol_no__in=Subquery(pol_numbers))
                # customercare_qs = Customercare.objects.filter(pol_no__in=Subquery(pol_numbers))
                # relation_qs = RelationMaster.objects.all()
                #
                # print(paid_qs)
                #
                # paid_df = pd.DataFrame(paid_qs)
                # out_df = pd.DataFrame(out_qs)
                # crs_df = pd.DataFrame(crs_qs)
                # customercare_df = pd.DataFrame(customercare_qs)
                # relation_df = pd.DataFrame(relation_qs.values())





        mydict = {
                'broker_name': broker_name,

            }
        return render(request, 'Management/broker.html', context=mydict)


class BrokerView(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'chart_form.html', {})
    # your code for handling GET requests

    def post(self, request, *args, **kwargs):
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')
        product_type = request.POST.get('product_type')

        # filter data based on user inputs
        table1_data = Table1.objects.filter(Q(date__gte=start_date) & Q(date__lte=end_date))
        table2_data = Table2.objects.filter(product_type=product_type)

        # create charts based on filtered data
        chart1 = self.create_chart(table1_data)
        chart2 = self.create_chart(table2_data)
        return render(request, 'chart_results.html', {'chart1': chart1, 'chart2': chart2})



def paid(request):
    if request.user.is_authenticated:

        table_layout = go.Layout(
            margin=dict(l=0, r=0, t=20, b=0),
            # padding=dict(l=0, r=0, t=0, b=0)
        )
        today = date.today()

        def add_percentage(df):
            # Apply percentage symbol to numeric columns
            numeric_cols = df.select_dtypes(include=[float, int]).columns
            df[numeric_cols] = df[numeric_cols].applymap(lambda x: f"{x}%")
            return df

        daily_settled = AllicSettledCasesOfYesterday.objects.values('pol_no','corp_retail','lodgetype','utr_updatedate','last_document_received','lodgedate','br_cd_dor','ho_cd_dor','inwdt','gmc_category','actuallosstype','prs_date','gmc_category','psu_pvt')
        settled_data = pd.DataFrame(daily_settled)


        settled_data['LDRTAT'] = settled_data['utr_updatedate'] - settled_data['last_document_received']
        settled_data['LDRTAT'] = settled_data['LDRTAT'].dt.days

        # EndToEndTAT

        date_list = settled_data[["lodgedate", "br_cd_dor", "ho_cd_dor", "inwdt"]].apply(lambda x: x.min(), axis=1)
        settled_data['EndToEndTAT'] = settled_data["utr_updatedate"] - date_list
        settled_data['EndToEndTAT'] = settled_data['EndToEndTAT'].dt.days

        conditions = [
            (settled_data['corp_retail'].str.strip() == 'Retail'),
            (settled_data['gmc_category'].str.strip() == 'PSU')
        ]

        # create a list of the values we want to assign for each condition
        values = ['RETAIL', 'PSU GMC']

        # create a new column and use np.select to assign values to it using our lists as arguments
        settled_data['CLIENT_TYPE'] = np.select(conditions, values, default='PVT GMC')

        # RI/Cashless

        conditions = [
            (settled_data['lodgetype'] == 'Cash Less'),
            (settled_data['lodgetype'] == 'AL Issued'),
            (settled_data['lodgetype'] == 'RAL Lodged'),
            ((settled_data['lodgetype'] == 'Reconsideration') & (
                    settled_data['actuallosstype'] == 'Cash Less')),
        ]

        # create a list of the values we want to assign for each condition
        values = ['Cash Less', 'Cash Less', 'Cash Less', 'Cash Less']

        # create a new column and use np.select to assign values to it using our lists as arguments
        settled_data['RI/Cashless'] = np.select(conditions, values, default='Reimbursement')
        # create a new column and use np.select to assign values to it using our lists as arguments

        settled_data['prs_date'] = pd.to_datetime(settled_data['prs_date'], format='%y-%m-%d')
        settled_data['last_document_received'] = pd.to_datetime(settled_data['last_document_received'],format='%y-%m-%d')

        # Processing_TAT
        settled_data['last_document_received'] = settled_data['last_document_received'].dt.tz_convert(settled_data['prs_date'].dt.tz)

        settled_data['Processing_TAT'] = settled_data['prs_date'] - settled_data['last_document_received']
        settled_data['Processing_TAT'] = settled_data['Processing_TAT'].dt.days

        # EndToEnd_Setted_BAND

        conditions = [
            (settled_data['EndToEndTAT'] > 0) & (settled_data['EndToEndTAT'] <= 3),
            (settled_data['EndToEndTAT'] > 3) & (settled_data['EndToEndTAT'] <= 7),
            (settled_data['EndToEndTAT'] > 7) & (settled_data['EndToEndTAT'] <= 10),
            (settled_data['EndToEndTAT'] > 10) & (settled_data['EndToEndTAT'] <= 15),
            (settled_data['EndToEndTAT'] > 15) & (settled_data['EndToEndTAT'] <= 21),
            (settled_data['EndToEndTAT'] > 21) & (settled_data['EndToEndTAT'] <= 30),
            (settled_data['EndToEndTAT'] > 30) & (settled_data['EndToEndTAT'] <= 45),
            (settled_data['EndToEndTAT'] > 45) & (settled_data['EndToEndTAT'] <= 60),
            (settled_data['EndToEndTAT'] > 60) & (settled_data['EndToEndTAT'] <= 90),
            (settled_data['EndToEndTAT'] > 90) & (settled_data['EndToEndTAT'] <= 120)
        ]

        # create a list of the values we want to assign for each condition
        values = ['00-03 Days', '04-07 Days', '08-10 Days', '11-15 Days', '16-21 Days', '22-30 Days', '31-45 Days',
                  '46-60 Days', '61-90 Days', '91-120 Days']

        # create a new column and use np.select to assign values to it using our lists as arguments
        settled_data['EndToEnd_Setted_BAND'] = np.select(conditions, values, default='ABOVE 121 DAYS')


        # ProcessingTAT_Setted_BAND
        conditions = [
            (settled_data['Processing_TAT'] <= 1),
            (settled_data['Processing_TAT'] > 1) & (settled_data['Processing_TAT'] <= 3),
            (settled_data['Processing_TAT'] > 3) & (settled_data['Processing_TAT'] <= 7),
            (settled_data['Processing_TAT'] > 7) & (settled_data['Processing_TAT'] <= 10),
            (settled_data['Processing_TAT'] > 10) & (settled_data['Processing_TAT'] <= 15),
            (settled_data['Processing_TAT'] > 15) & (settled_data['Processing_TAT'] <= 21),
            (settled_data['Processing_TAT'] > 21) & (settled_data['Processing_TAT'] <= 30),
            (settled_data['Processing_TAT'] > 30) & (settled_data['Processing_TAT'] <= 45),
            (settled_data['Processing_TAT'] > 45) & (settled_data['Processing_TAT'] <= 60),
            (settled_data['Processing_TAT'] > 60) & (settled_data['Processing_TAT'] <= 90),
            (settled_data['Processing_TAT'] > 90) & (settled_data['Processing_TAT'] <= 120)
        ]

        # create a list of the values we want to assign for each condition
        values = ['00-01 Days', '00-03 Days', '04-07 Days', '08-10 Days', '11-15 Days', '16-21 Days', '22-30 Days',
                  '31-45 Days', '46-60 Days', '61-90 Days', '91-120 Days']

        # create a new column and use np.select to assign values to it using our lists as arguments
        settled_data['ProcessingTAT_Setted_BAND'] = np.select(conditions, values, default='ABOVE 121 DAYS')

        # GMC & Retail Settled Claims End to End TAT

        gmc_retail_settled__EndTAT = settled_data.pivot_table(index=['CLIENT_TYPE', 'RI/Cashless'],
                                                              columns='EndToEnd_Setted_BAND', values='pol_no',
                                                              aggfunc='count', margins=True, margins_name='Total')
        gmc_retail_settled__EndTAT = gmc_retail_settled__EndTAT.reset_index()
        gmc_retail_settled__EndTAT.replace(np.NaN, 0, inplace=True)


        # gmc_retail_settled__EndTAT_fig

        gmc_retail_settled__EndTAT_fig = go.Figure(
            data=[go.Table(columnwidth=[100] + [50] * (len(gmc_retail_settled__EndTAT) - 1),
                           header=dict(values=gmc_retail_settled__EndTAT.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=gmc_retail_settled__EndTAT.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))], layout=table_layout)

        gmc_retail_settled__EndTAT_fig.update_layout(height=len(gmc_retail_settled__EndTAT) * 50)

        gmc_retail_settled__EndTAT_fig_1 = plot(gmc_retail_settled__EndTAT_fig, output_type='div', config=config)

        gmc_retail_settled__EndTAT_per = settled_data.pivot_table(index=['CLIENT_TYPE', 'RI/Cashless'],
                                                                  columns='EndToEnd_Setted_BAND', values='pol_no',
                                                                  aggfunc='count')
        gmc_retail_settled__EndTAT_per = gmc_retail_settled__EndTAT_per.reset_index()
        gmc_retail_settled__EndTAT_per.replace(np.NaN, 0, inplace=True)

        gmc_retail_settled__EndTAT_1 = gmc_retail_settled__EndTAT_per.iloc[:, 2:].apply(lambda x: x / x.sum() * 100,
                                                                                        axis=1).round(2)
        gmc_retail_settled__EndTAT_2 = pd.concat(
            [gmc_retail_settled__EndTAT_per[['CLIENT_TYPE', 'RI/Cashless']], gmc_retail_settled__EndTAT_1], axis=1)

        gmc_retail_settled__EndTAT_2.loc[:,'Total'] = gmc_retail_settled__EndTAT_2.sum(numeric_only=True,axis=1).round(2)

        # # Apply percentage symbol to numeric columns
        # numeric_cols = gmc_retail_settled__EndTAT_2.select_dtypes(include=[float, int]).columns
        # gmc_retail_settled__EndTAT_2[numeric_cols] = gmc_retail_settled__EndTAT_2[numeric_cols].applymap(lambda x: f"{x}%")

        gmc_retail_settled__EndTAT_2 = add_percentage(gmc_retail_settled__EndTAT_2)

        gmc_retail_settled__EndTAT_per_fig = go.Figure(
            data=[go.Table(columnwidth=[100] + [50] * (len(gmc_retail_settled__EndTAT_2) - 1),
                           header=dict(values=gmc_retail_settled__EndTAT_2.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=gmc_retail_settled__EndTAT_2.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))], layout=table_layout)

        gmc_retail_settled__EndTAT_per_fig.update_layout(height=len(gmc_retail_settled__EndTAT_2) * 50)

        gmc_retail_settled__EndTAT_per_fig_1 = plot(gmc_retail_settled__EndTAT_per_fig, output_type='div', config=config)

        # LDR_Settled_BAND
        conditions = [
            (settled_data['LDRTAT'] > 0) & (settled_data['LDRTAT'] <= 3),
            (settled_data['LDRTAT'] > 3) & (settled_data['LDRTAT'] <= 7),
            (settled_data['LDRTAT'] > 7) & (settled_data['LDRTAT'] <= 10),
            (settled_data['LDRTAT'] > 10) & (settled_data['LDRTAT'] <= 15),
            (settled_data['LDRTAT'] > 15) & (settled_data['LDRTAT'] <= 21),
            (settled_data['LDRTAT'] > 21) & (settled_data['LDRTAT'] <= 30),
            (settled_data['LDRTAT'] > 30) & (settled_data['LDRTAT'] <= 45),
            (settled_data['LDRTAT'] > 45) & (settled_data['LDRTAT'] <= 60),
            (settled_data['LDRTAT'] > 60) & (settled_data['LDRTAT'] <= 90),
            (settled_data['LDRTAT'] > 90) & (settled_data['LDRTAT'] <= 120)
        ]

        # create a list of the values we want to assign for each condition
        values = ['00-03 Days', '04-07 Days', '08-10 Days', '11-15 Days', '16-21 Days', '22-30 Days', '31-45 Days',
                  '46-60 Days', '61-90 Days', '91-120 Days']

        # create a new column and use np.select to assign values to it using our lists as arguments
        settled_data['LDR_Settled_BAND'] = np.select(conditions, values, default='ABOVE 121 DAYS')

        # GMC & Retail Settled TAT from Last Document Received Date

        gmc_retail_settled_LDRTAT = settled_data.pivot_table(index=['CLIENT_TYPE', 'RI/Cashless'],
                                                              columns='LDR_Settled_BAND', values='pol_no',
                                                              aggfunc='count', margins=True, margins_name='Total')
        gmc_retail_settled_LDRTAT = gmc_retail_settled_LDRTAT.reset_index()
        gmc_retail_settled_LDRTAT.replace(np.NaN, 0, inplace=True)


        # gmc_retail_settled__EndTAT_fig

        gmc_retail_settled_LDRTAT_fig = go.Figure(
            data=[go.Table(columnwidth=[100] + [50] * (len(gmc_retail_settled_LDRTAT) - 1),
                           header=dict(values=gmc_retail_settled_LDRTAT.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=gmc_retail_settled_LDRTAT.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))], layout=table_layout)

        gmc_retail_settled_LDRTAT_fig.update_layout(height=len(gmc_retail_settled_LDRTAT) * 50)

        gmc_retail_settled_LDRTAT_fig_1 = plot(gmc_retail_settled_LDRTAT_fig, output_type='div', config=config)

        gmc_retail_settled_LDRTAT_per = settled_data.pivot_table(index=['CLIENT_TYPE', 'RI/Cashless'],
                                                                  columns='LDR_Settled_BAND', values='pol_no',
                                                                  aggfunc='count')
        gmc_retail_settled_LDRTAT_per = gmc_retail_settled_LDRTAT_per.reset_index()
        gmc_retail_settled_LDRTAT_per.replace(np.NaN, 0, inplace=True)

        gmc_retail_settled_LDRTAT_per_1 = gmc_retail_settled_LDRTAT_per.iloc[:, 2:].apply(lambda x: x / x.sum() * 100,
                                                                                        axis=1).round(2)
        gmc_retail_settled_LDRTAT_per_2 = pd.concat(
            [gmc_retail_settled_LDRTAT_per[['CLIENT_TYPE', 'RI/Cashless']], gmc_retail_settled_LDRTAT_per_1], axis=1)

        gmc_retail_settled_LDRTAT_per_2.loc[:, 'Total'] = gmc_retail_settled_LDRTAT_per_2.sum(numeric_only=True,axis=1).round(2)

        gmc_retail_settled_LDRTAT_per_2 = add_percentage(gmc_retail_settled_LDRTAT_per_2)

        gmc_retail_settled_LDRTAT_per_fig = go.Figure(
            data=[go.Table(columnwidth=[100] + [50] * (len(gmc_retail_settled_LDRTAT_per_2) - 1),
                           header=dict(values=gmc_retail_settled_LDRTAT_per_2.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=gmc_retail_settled_LDRTAT_per_2.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))], layout=table_layout)

        gmc_retail_settled_LDRTAT_per_fig.update_layout(height=len(gmc_retail_settled_LDRTAT_per_2) * 50)

        gmc_retail_settled_LDRTAT_per_fig_1 = plot(gmc_retail_settled_LDRTAT_per_fig, output_type='div', config=config)



        # GMC & Retail Settled Claims Processing TAT

        gmc_retail_settled_ProcessingTAT = settled_data.pivot_table(index=['CLIENT_TYPE', 'RI/Cashless'],
                                                              columns='ProcessingTAT_Setted_BAND', values='pol_no',
                                                              aggfunc='count', margins=True, margins_name='Total')
        gmc_retail_settled_ProcessingTAT = gmc_retail_settled_ProcessingTAT.reset_index()
        gmc_retail_settled_ProcessingTAT.replace(np.NaN, 0, inplace=True)


        # gmc_retail_settled__EndTAT_fig

        gmc_retail_settled_ProcessingTAT_fig = go.Figure(
            data=[go.Table(columnwidth=[100] + [50] * (len(gmc_retail_settled_ProcessingTAT) - 1),
                           header=dict(values=gmc_retail_settled_ProcessingTAT.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=gmc_retail_settled_ProcessingTAT.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))], layout=table_layout)

        gmc_retail_settled_ProcessingTAT_fig.update_layout(height=len(gmc_retail_settled_ProcessingTAT) * 50)

        gmc_retail_settled_ProcessingTAT_fig_1 = plot(gmc_retail_settled_ProcessingTAT_fig, output_type='div', config=config)



        gmc_retail_settled_ProcessingTAT_per = settled_data.pivot_table(index=['CLIENT_TYPE', 'RI/Cashless'],
                                                                  columns='ProcessingTAT_Setted_BAND', values='pol_no',
                                                                  aggfunc='count')
        gmc_retail_settled_ProcessingTAT_per = gmc_retail_settled_ProcessingTAT_per.reset_index()
        gmc_retail_settled_ProcessingTAT_per.replace(np.NaN, 0, inplace=True)

        gmc_retail_settled_ProcessingTAT_per_1 = gmc_retail_settled_ProcessingTAT_per.iloc[:, 2:].apply(lambda x: x / x.sum() * 100,
                                                                                        axis=1).round(2)
        gmc_retail_settled_ProcessingTAT_per_2 = pd.concat(
            [gmc_retail_settled_ProcessingTAT_per[['CLIENT_TYPE', 'RI/Cashless']], gmc_retail_settled_ProcessingTAT_per_1], axis=1)

        gmc_retail_settled_ProcessingTAT_per_2.loc[:, 'Total'] = gmc_retail_settled_ProcessingTAT_per_2.sum(numeric_only=True,axis=1).round(2)

        gmc_retail_settled_ProcessingTAT_per_2 = add_percentage(gmc_retail_settled_ProcessingTAT_per_2)


        gmc_retail_settled_ProcessingTAT_per_fig = go.Figure(
            data=[go.Table(columnwidth=[100] + [50] * (len(gmc_retail_settled_ProcessingTAT_per_2) - 1),
                           header=dict(values=gmc_retail_settled_ProcessingTAT_per_2.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=gmc_retail_settled_ProcessingTAT_per_2.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))], layout=table_layout)

        gmc_retail_settled_ProcessingTAT_per_fig.update_layout(height=len(gmc_retail_settled_ProcessingTAT_per_2) * 50)

        gmc_retail_settled_ProcessingTAT_per_fig_1 = plot(gmc_retail_settled_ProcessingTAT_per_fig, output_type='div', config=config)





        mydict = {'gmc_retail_settled__EndTAT_fig_1':gmc_retail_settled__EndTAT_fig_1,
                  'gmc_retail_settled__EndTAT_per_fig_1':gmc_retail_settled__EndTAT_per_fig_1,
                  'gmc_retail_settled_LDRTAT_fig_1':gmc_retail_settled_LDRTAT_fig_1,
                  'gmc_retail_settled_LDRTAT_per_fig_1':gmc_retail_settled_LDRTAT_per_fig_1,
                  'gmc_retail_settled_ProcessingTAT_fig_1':gmc_retail_settled_ProcessingTAT_fig_1,
                  'gmc_retail_settled_ProcessingTAT_per_fig_1':gmc_retail_settled_ProcessingTAT_per_fig_1,
                  }



        return render(request, 'Management/paidclaims.html',context=mydict )


def crs(request):
    if request.user.is_authenticated:
        return render(request, 'Management/crsclaims.html', )


def outstanding(request):
    if request.user.is_authenticated:

        table_layout = go.Layout(
            margin=dict(l=0, r=0, t=20, b=0),
            # padding=dict(l=0, r=0, t=0, b=0)
        )

        today = date.today()

        daily_out = Allicdailyoutstanding.objects.values('sla_heading','corp_retail', 'lodge_type','sub_head',
                                                                    'last_document_received', 'lodge_date', 'gmc_category',
                                                                    'actual_loss_type', 'prs_date', 'gmc_category',
                                                                    'psu_pvt')
        daily_out_df = pd.DataFrame(daily_out)

        daily_out_df['psu_pvt'].replace('NA', 'PVT IC', inplace=True)
        daily_out_df['psu_pvt'].replace('PSU', 'PSU IC', inplace=True)
        daily_out_df['gmc_category'].fillna("PVT GMC", inplace=True)
        daily_out_df['gmc_category'].replace('PSU', 'PSU GMC', inplace=True)

        conditions = [
            (daily_out_df['corp_retail'].str.strip() == 'Retail'),
            (daily_out_df['gmc_category'].str.strip() == 'PSU GMC')
        ]

        # create a list of the values we want to assign for each condition
        values = ['RETAIL', 'PSU GMC']

        # create a new column and use np.select to assign values to it using our lists as arguments
        daily_out_df['client'] = np.select(conditions, values, default='PVT GMC')

        daily_out_df['lodge_date'] = pd.to_datetime(daily_out_df['lodge_date'], format='%y-%m-%d')
        LODGE_DATE = daily_out_df['lodge_date']

        TAT_LODGEDATE = []
        for d in LODGE_DATE:
            TAT_LODGEDATE.append(today - d.date() - timedelta(days=1))

        daily_out_df['TAT_LODGEDATE'] = TAT_LODGEDATE
        daily_out_df['TAT_LODGEDATE'] = daily_out_df['TAT_LODGEDATE'].dt.days

        # LODGEDATE_BAND

        conditions = [
            (daily_out_df['TAT_LODGEDATE'] > 0) & (daily_out_df['TAT_LODGEDATE'] <= 3),
            (daily_out_df['TAT_LODGEDATE'] > 3) & (daily_out_df['TAT_LODGEDATE'] <= 7),
            (daily_out_df['TAT_LODGEDATE'] > 7) & (daily_out_df['TAT_LODGEDATE'] <= 10),
            (daily_out_df['TAT_LODGEDATE'] > 10) & (daily_out_df['TAT_LODGEDATE'] <= 15),
            (daily_out_df['TAT_LODGEDATE'] > 15) & (daily_out_df['TAT_LODGEDATE'] <= 21),
            (daily_out_df['TAT_LODGEDATE'] > 21) & (daily_out_df['TAT_LODGEDATE'] <= 30),
            (daily_out_df['TAT_LODGEDATE'] > 30) & (daily_out_df['TAT_LODGEDATE'] <= 45),
            (daily_out_df['TAT_LODGEDATE'] > 45) & (daily_out_df['TAT_LODGEDATE'] <= 60),
            (daily_out_df['TAT_LODGEDATE'] > 60) & (daily_out_df['TAT_LODGEDATE'] <= 90),
            (daily_out_df['TAT_LODGEDATE'] > 90) & (daily_out_df['TAT_LODGEDATE'] <= 120)
        ]

        # create a list of the values we want to assign for each condition
        values = ['00-03 Days', '04-07 Days', '08-10 Days', '11-15 Days', '16-21 Days', '22-30 Days', '31-45 Days',
                  '46-60 Days', '61-90 Days', '91-120 Days']

        # create a new column and use np.select to assign values to it using our lists as arguments
        daily_out_df['LODGEDATE_BAND'] = np.select(conditions, values, default='ABOVE 121 DAYS')

        # RI/Cashless

        conditions = [
            (daily_out_df['lodge_type'] == 'Cash Less'),
            (daily_out_df['lodge_type'] == 'AL Issued'),
            (daily_out_df['lodge_type'] == 'RAL Lodged'),
            ((daily_out_df['lodge_type'] == 'Reconsideration') & (
                        daily_out_df['actual_loss_type'] == 'Cash Less')),
        ]

        # create a list of the values we want to assign for each condition
        values = ['Cash Less', 'Cash Less', 'Cash Less', 'Cash Less']

        # create a new column and use np.select to assign values to it using our lists as arguments
        daily_out_df['RI/Cashless'] = np.select(conditions, values, default='Reimbursement')

        RI_Data = daily_out_df[daily_out_df['RI/Cashless'] == 'Reimbursement']

        corp_retail_ri_claims = pd.crosstab(RI_Data['client'], RI_Data['LODGEDATE_BAND'], margins=True,
                                            margins_name='Total')
        corp_retail_ri_claims1 = corp_retail_ri_claims.reset_index()
        # corp_retail_ri_claims1.rename(columns={'corp retail':'TAT from Lodge Date Client Type'}, inplace=True)

        corp_retail_ri_claims_per = pd.crosstab(RI_Data['client'], RI_Data['LODGEDATE_BAND'], margins=True,
                                                margins_name='Total', normalize='index').round(4) * 100

        # corp_retail_ri_claims_per.reset_index(drop=True, inplace=True)
        corp_retail_ri_claims_per.reset_index(inplace=True)
        corp_retail_ri_claims_per.loc[:, 'Total'] = corp_retail_ri_claims_per.sum(numeric_only=True,axis=1)
        corp_retail_ri_claims_per = corp_retail_ri_claims_per.round(2)

        # Apply percentage symbol to numeric columns
        numeric_cols = corp_retail_ri_claims_per.select_dtypes(include=[float, int]).columns
        corp_retail_ri_claims_per[numeric_cols] = corp_retail_ri_claims_per[numeric_cols].applymap(lambda x: f"{x}%")

        corp_retail_ri_claims_value_per = pd.concat([corp_retail_ri_claims1, corp_retail_ri_claims_per], axis=0)

        # GMC & Retail: Outstanding Reimbursement Claims

        corp_retail_ri_claims_value_per_fig = go.Figure(
            data=[go.Table(columnwidth=[100] + [50] * (len(corp_retail_ri_claims_value_per) - 1),
                           header=dict(values=corp_retail_ri_claims_value_per.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=corp_retail_ri_claims_value_per.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))], layout=table_layout)

        corp_retail_ri_claims_value_per_fig.update_layout(height=len(corp_retail_ri_claims_value_per) * 40)


        corp_retail_ri_claims_value_per_fig_1 = plot(corp_retail_ri_claims_value_per_fig, output_type='div', config=config)


        # PVT GMC Lodge Type
        lodge_type_wise_pvt_gmc_RIclaims = RI_Data[RI_Data['client'] == 'PVT GMC']
        lodge_type_wise_pvt_gmc_RIclaims = lodge_type_wise_pvt_gmc_RIclaims.pivot_table(index=['client', 'lodge_type'],
                                                                                        columns='LODGEDATE_BAND',
                                                                                        values='sla_heading',
                                                                                        aggfunc='count')

        lodge_type_wise_pvt_gmc_RIclaims.columns.name = None
        lodge_type_wise_pvt_gmc_RIclaims = lodge_type_wise_pvt_gmc_RIclaims.reset_index()
        lodge_type_wise_pvt_gmc_RIclaims = lodge_type_wise_pvt_gmc_RIclaims.iloc[:, 1:]
        lodge_type_wise_pvt_gmc_RIclaims.loc['Total'] = lodge_type_wise_pvt_gmc_RIclaims.sum(numeric_only=True, axis=0)
        lodge_type_wise_pvt_gmc_RIclaims['Total'] = lodge_type_wise_pvt_gmc_RIclaims.sum(numeric_only=True,axis=1)
        lodge_type_wise_pvt_gmc_RIclaims['lodge_type'].replace(np.NaN, 'Total', inplace=True)
        lodge_type_wise_pvt_gmc_RIclaims.replace(np.NaN, 0, inplace=True)
        lodge_type_wise_pvt_gmc_RIclaims.rename(columns={'lodge_type': 'PVT GMC Lodge Type'}, inplace=True)

        # PSU GMC Lodge Type
        lodge_type_wise_psu_gmc_RIclaims = RI_Data[RI_Data['client'] == 'PSU GMC']
        lodge_type_wise_psu_gmc_RIclaims = lodge_type_wise_psu_gmc_RIclaims.pivot_table(index=['client', 'lodge_type'],
                                                                                        columns='LODGEDATE_BAND',
                                                                                        values='sla_heading',
                                                                                        aggfunc='count')

        # x.columns = x.columns.droplevel(0)
        lodge_type_wise_psu_gmc_RIclaims.columns.name = None
        lodge_type_wise_psu_gmc_RIclaims = lodge_type_wise_psu_gmc_RIclaims.reset_index()
        lodge_type_wise_psu_gmc_RIclaims = lodge_type_wise_psu_gmc_RIclaims.iloc[:, 1:]
        lodge_type_wise_psu_gmc_RIclaims.loc['Total'] = lodge_type_wise_psu_gmc_RIclaims.sum(numeric_only=True, axis=0)
        lodge_type_wise_psu_gmc_RIclaims['Total'] = lodge_type_wise_psu_gmc_RIclaims.sum(numeric_only=True,axis=1)
        lodge_type_wise_psu_gmc_RIclaims['lodge_type'].replace(np.NaN, 'Total', inplace=True)
        lodge_type_wise_psu_gmc_RIclaims.replace(np.NaN, 0, inplace=True)
        lodge_type_wise_psu_gmc_RIclaims.rename(columns={'lodge_type': 'PSU GMC Lodge Type'}, inplace=True)

        lodge_type_wise_retail_RIclaims = RI_Data[RI_Data['corp_retail'] == 'Retail']
        lodge_type_wise_retail_RIclaims = lodge_type_wise_retail_RIclaims.pivot_table(
            index=['corp_retail', 'lodge_type'], columns='LODGEDATE_BAND', values='sla_heading', aggfunc='count')

        # x.columns = x.columns.droplevel(0)
        lodge_type_wise_retail_RIclaims.columns.name = None
        lodge_type_wise_retail_RIclaims = lodge_type_wise_retail_RIclaims.reset_index()
        lodge_type_wise_retail_RIclaims = lodge_type_wise_retail_RIclaims.iloc[:, 1:]
        lodge_type_wise_retail_RIclaims.loc['Total'] = lodge_type_wise_retail_RIclaims.sum(numeric_only=True, axis=0)
        lodge_type_wise_retail_RIclaims['Total'] = lodge_type_wise_retail_RIclaims.sum(numeric_only=True,axis=1)
        lodge_type_wise_retail_RIclaims['lodge_type'].replace(np.NaN, 'Total', inplace=True)
        lodge_type_wise_retail_RIclaims.replace(np.NaN, 0, inplace=True)
        lodge_type_wise_retail_RIclaims.rename(columns={'Lodge Type': 'Retail Lodge Type wise TAT'}, inplace=True)


        lodge_type_wise_pvt_gmc_RIclaims_fig = go.Figure(
            data=[go.Table(columnwidth=[100] + [50] * (len(lodge_type_wise_pvt_gmc_RIclaims) - 1),
                           header=dict(values=lodge_type_wise_pvt_gmc_RIclaims.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=lodge_type_wise_pvt_gmc_RIclaims.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))], layout=table_layout)

        lodge_type_wise_pvt_gmc_RIclaims_fig.update_layout(height=len(lodge_type_wise_pvt_gmc_RIclaims) * 40)

        lodge_type_wise_pvt_gmc_RIclaims_fig_1 = plot(lodge_type_wise_pvt_gmc_RIclaims_fig, output_type='div',
                                                      config=config)


        lodge_type_wise_psu_gmc_RIclaims_fig = go.Figure(
            data=[go.Table(columnwidth=[100] + [50] * (len(lodge_type_wise_psu_gmc_RIclaims) - 1),
                           header=dict(values=lodge_type_wise_psu_gmc_RIclaims.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=lodge_type_wise_psu_gmc_RIclaims.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))], layout=table_layout)

        lodge_type_wise_psu_gmc_RIclaims_fig.update_layout(height=len(lodge_type_wise_psu_gmc_RIclaims) * 40)

        lodge_type_wise_psu_gmc_RIclaims_fig_1 = plot(lodge_type_wise_psu_gmc_RIclaims_fig, output_type='div',
                                                      config=config)



        lodge_type_wise_retail_RIclaims = RI_Data[RI_Data['corp_retail'] == 'Retail']
        lodge_type_wise_retail_RIclaims = lodge_type_wise_retail_RIclaims.pivot_table(
            index=['corp_retail', 'lodge_type'], columns='LODGEDATE_BAND', values='sla_heading', aggfunc='count')

        # x.columns = x.columns.droplevel(0)
        lodge_type_wise_retail_RIclaims.columns.name = None
        lodge_type_wise_retail_RIclaims = lodge_type_wise_retail_RIclaims.reset_index()
        lodge_type_wise_retail_RIclaims = lodge_type_wise_retail_RIclaims.iloc[:, 1:]
        lodge_type_wise_retail_RIclaims.loc['Total'] = lodge_type_wise_retail_RIclaims.sum(numeric_only=True, axis=0)
        lodge_type_wise_retail_RIclaims['Total'] = lodge_type_wise_retail_RIclaims.sum(numeric_only=True,axis=1)
        lodge_type_wise_retail_RIclaims['lodge_type'].replace(np.NaN, 'Total', inplace=True)
        lodge_type_wise_retail_RIclaims.replace(np.NaN, 0, inplace=True)
        lodge_type_wise_retail_RIclaims.rename(columns={'lodge_type': 'Retail Lodge Type wise TAT'}, inplace=True)


        lodge_type_wise_retail_RIclaims_fig = go.Figure(
            data=[go.Table(columnwidth=[100] + [50] * (len(lodge_type_wise_retail_RIclaims) - 1),
                           header=dict(values=lodge_type_wise_retail_RIclaims.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=lodge_type_wise_retail_RIclaims.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))], layout=table_layout)

        lodge_type_wise_retail_RIclaims_fig.update_layout(height=len(lodge_type_wise_retail_RIclaims) * 40)

        lodge_type_wise_retail_RIclaims_fig_1 = plot(lodge_type_wise_retail_RIclaims_fig, output_type='div',
                                                      config=config)




        # Corporate: Reason Wise Outstanding RI Claims

        reason_wise_corporate_RIclaims = RI_Data[RI_Data['corp_retail'] == 'Corporate']

        reason_wise_corporate_RIclaims = reason_wise_corporate_RIclaims.pivot_table(index=['client', 'sub_head'],
                                                                                columns='LODGEDATE_BAND',
                                                                                values='sla_heading', aggfunc='count')

        # x.columns = x.columns.droplevel(0)
        reason_wise_corporate_RIclaims.columns.name = None
        reason_wise_corporate_RIclaims = reason_wise_corporate_RIclaims.reset_index()
        reason_wise_corporate_RIclaims = reason_wise_corporate_RIclaims.iloc[:, 1:]
        reason_wise_corporate_RIclaims.loc['Total'] = reason_wise_corporate_RIclaims.sum(numeric_only=True, axis=0)
        reason_wise_corporate_RIclaims['Total'] = reason_wise_corporate_RIclaims.sum(numeric_only=True,axis=1)
        reason_wise_corporate_RIclaims['sub_head'].replace(np.NaN, 'Total', inplace=True)
        reason_wise_corporate_RIclaims.replace(np.NaN, 0, inplace=True)


        reason_wise_corporate_RIclaims_fig = go.Figure(
            data=[go.Table(columnwidth=[100] + [50] * (len(reason_wise_corporate_RIclaims) - 1),
                           header=dict(values=reason_wise_corporate_RIclaims.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=reason_wise_corporate_RIclaims.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))], layout=table_layout)

        reason_wise_corporate_RIclaims_fig.update_layout(height=len(lodge_type_wise_psu_gmc_RIclaims) * 80)

        reason_wise_corporate_RIclaims_fig_1 = plot(reason_wise_corporate_RIclaims_fig, output_type='div',
                                                      config=config)
        
        
        # Retal: Reason Wise Outstanding RI Claims

        reason_wise_retail_RIclaims = RI_Data[RI_Data['corp_retail'] == 'Retail']

        reason_wise_retail_RIclaims = reason_wise_retail_RIclaims.pivot_table(index=['client', 'sub_head'],
                                                                                columns='LODGEDATE_BAND',
                                                                                values='sla_heading', aggfunc='count')

        # x.columns = x.columns.droplevel(0)
        reason_wise_retail_RIclaims.columns.name = None
        reason_wise_retail_RIclaims = reason_wise_retail_RIclaims.reset_index()
        reason_wise_retail_RIclaims = reason_wise_retail_RIclaims.iloc[:, 1:]
        reason_wise_retail_RIclaims.loc['Total'] = reason_wise_retail_RIclaims.sum(numeric_only=True, axis=0)
        reason_wise_retail_RIclaims['Total'] = reason_wise_retail_RIclaims.sum(numeric_only=True,axis=1)
        reason_wise_retail_RIclaims['sub_head'].replace(np.NaN, 'Total', inplace=True)
        reason_wise_retail_RIclaims.replace(np.NaN, 0, inplace=True)


        reason_wise_retail_RIclaims_fig = go.Figure(
            data=[go.Table(columnwidth=[100] + [50] * (len(reason_wise_retail_RIclaims) - 1),
                           header=dict(values=reason_wise_retail_RIclaims.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=reason_wise_retail_RIclaims.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))], layout=table_layout)

        reason_wise_retail_RIclaims_fig.update_layout(height=len(lodge_type_wise_psu_gmc_RIclaims) * 80)

        reason_wise_retail_RIclaims_fig_1 = plot(reason_wise_retail_RIclaims_fig, output_type='div',
                                                      config=config)


        # PVT GMC: Reason Wise Outstanding RI Claims

        reason_wise_pvt_gmc_RIclaims = RI_Data[RI_Data['client'] == 'PVT GMC']
        reason_wise_pvt_gmc_RIclaims = reason_wise_pvt_gmc_RIclaims.pivot_table(index=['client', 'sub_head'],
                                                                                columns='LODGEDATE_BAND',
                                                                                values='sla_heading', aggfunc='count')

        # x.columns = x.columns.droplevel(0)
        reason_wise_pvt_gmc_RIclaims.columns.name = None
        reason_wise_pvt_gmc_RIclaims = reason_wise_pvt_gmc_RIclaims.reset_index()
        reason_wise_pvt_gmc_RIclaims = reason_wise_pvt_gmc_RIclaims.iloc[:, 1:]
        reason_wise_pvt_gmc_RIclaims.loc['Total'] = reason_wise_pvt_gmc_RIclaims.sum(numeric_only=True, axis=0)
        reason_wise_pvt_gmc_RIclaims['Total'] = reason_wise_pvt_gmc_RIclaims.sum(numeric_only=True,axis=1)
        reason_wise_pvt_gmc_RIclaims['sub_head'].replace(np.NaN, 'Total', inplace=True)
        reason_wise_pvt_gmc_RIclaims.replace(np.NaN, 0, inplace=True)

        reason_wise_pvt_gmc_RIclaims1 = go.Figure(
            data=[go.Table(columnwidth=[200] + [50] * (len(reason_wise_pvt_gmc_RIclaims) - 1),
                           header=dict(values=reason_wise_pvt_gmc_RIclaims.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=reason_wise_pvt_gmc_RIclaims.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))])
        reason_wise_pvt_gmc_RIclaims1.update_layout(height=750)

        reason_wise_pvt_gmc_RIclaims_fig = plot(reason_wise_pvt_gmc_RIclaims1, output_type='div',
                                                      config=config)
        
        
        # PSU GMC: Reason Wise Outstanding RI Claims

        reason_wise_psu_gmc_RIclaims = RI_Data[RI_Data['client'] == 'PSU GMC']
        reason_wise_psu_gmc_RIclaims = reason_wise_psu_gmc_RIclaims.pivot_table(index=['client', 'sub_head'],
                                                                                columns='LODGEDATE_BAND',
                                                                                values='sla_heading', aggfunc='count')

        # x.columns = x.columns.droplevel(0)
        reason_wise_psu_gmc_RIclaims.columns.name = None
        reason_wise_psu_gmc_RIclaims = reason_wise_psu_gmc_RIclaims.reset_index()
        reason_wise_psu_gmc_RIclaims = reason_wise_psu_gmc_RIclaims.iloc[:, 1:]
        reason_wise_psu_gmc_RIclaims.loc['Total'] = reason_wise_psu_gmc_RIclaims.sum(numeric_only=True, axis=0)
        reason_wise_psu_gmc_RIclaims['Total'] = reason_wise_psu_gmc_RIclaims.sum(numeric_only=True,axis=1)
        reason_wise_psu_gmc_RIclaims['sub_head'].replace(np.NaN, 'Total', inplace=True)
        reason_wise_psu_gmc_RIclaims.replace(np.NaN, 0, inplace=True)

        reason_wise_psu_gmc_RIclaims1 = go.Figure(
            data=[go.Table(columnwidth=[200] + [50] * (len(reason_wise_psu_gmc_RIclaims) - 1),
                           header=dict(values=reason_wise_psu_gmc_RIclaims.columns,
                                       fill_color='#094780',
                                       line_color='darkslategray',
                                       align='left',
                                       font=dict(size=13, color='white')),
                           cells=dict(values=reason_wise_psu_gmc_RIclaims.T,
                                      fill_color='white',
                                      line_color='darkslategray',
                                      align='left'))])
        reason_wise_psu_gmc_RIclaims1.update_layout(height=750)

        reason_wise_psu_gmc_RIclaims_fig = plot(reason_wise_psu_gmc_RIclaims1, output_type='div',
                                                      config=config)


        mydict = {'corp_retail_ri_claims_value_per_fig_1':corp_retail_ri_claims_value_per_fig_1,
                  'lodge_type_wise_pvt_gmc_RIclaims_fig_1':lodge_type_wise_pvt_gmc_RIclaims_fig_1,
                  'lodge_type_wise_psu_gmc_RIclaims_fig_1':lodge_type_wise_psu_gmc_RIclaims_fig_1,
                  'reason_wise_corporate_RIclaims_fig_1':reason_wise_corporate_RIclaims_fig_1,
                  'reason_wise_retail_RIclaims_fig_1':reason_wise_retail_RIclaims_fig_1,
                  'lodge_type_wise_retail_RIclaims_fig_1':lodge_type_wise_retail_RIclaims_fig_1,
                  'reason_wise_pvt_gmc_RIclaims_fig':reason_wise_pvt_gmc_RIclaims_fig,
                  'reason_wise_psu_gmc_RIclaims_fig':reason_wise_psu_gmc_RIclaims_fig,}


        return render(request, 'Management/outstandingclaims.html', context=mydict )

def profile(request):
    if request.user.is_authenticated:
        user_management = Management.objects.filter(user=request.user.id).values()

        return render(request, 'Management/profile.html', {'data': user_management})



def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Important!
            messages.success(request, 'Your password was successfully updated!')
            return redirect('change_password')
        else:
            messages.error(request, 'Please correct the error below.')
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'Management/change_password.html', {
        'form': form
    })


# def abc(request):
#     all_out = Allicdailyoutstanding.objects.values('last_document_received','doa','first_intimation_date','revised_servicing_branch','servicing_branch')
#     all_outdf = pd.DataFrame(all_out)
#     today = date.today()
#     ldr = calculate_tat_ldr(all_outdf,today)
#     print(ldr)
#     return ldr


# Now, in your abc function:
def abc(request):
    all_out = Allicdailyoutstanding.objects.values('last_document_received', 'doa', 'first_intimation_date',
                                                   'revised_servicing_branch', 'servicing_branch')
    all_outdf = pd.DataFrame(all_out)
    today = date.today()

    # Call calculate_tat_ldr to get the Series
    ldr_series = calculate_tat_ldr(all_outdf, today)

    # Convert the Series to a list if needed
    ldr_list = ldr_series.tolist()

    print(ldr_list)  # Print or process the list as needed
    return ldr_list  # Return the list if required



