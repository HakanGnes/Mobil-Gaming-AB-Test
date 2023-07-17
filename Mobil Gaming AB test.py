# Cookie Cats is a hugely popular mobile puzzle game developed by Tactile Entertainment.
# It's a classic "connect three" style puzzle game where the player must connect tiles of the same color in order to
# clear the board and win the level. It also features singing cats. We're not kidding!
#
# As players progress through the game they will encounter gates that force them
# to wait some time before they can progress or make an in-app purchase.
# In this project, we will analyze the result of an A/B test where the first gate in Cookie Cats was moved from level 30 to level 40.
# In particular, we will analyze the impact on player retention.
#
# To complete this project, you should be comfortable working with pandas DataFrames and with using the pandas plot method.
# You should also have some understanding of hypothesis testing and bootstrap analysis.

# userid - a unique number that identifies each player.
# version - whether the player was put in the control group (gate_30 - a gate at level 30) or the test group (gate_40 - a gate at level 40).
# sum_gamerounds - the number of game rounds played by the player during the first week after installation.
# retention_1 - did the player come back and play 1 day after installing?
# retention_7 - did the player come back and play 7 days after installing?
# When a player installed the game, he or she was randomly assigned to either gate_30 or gate_40.


import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#  Data Preparing

df_ = pd.read_csv("Data/cookie_cats.csv")
df = df_.copy()
df.head()

# Data Control
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Outlier Check

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.25, q3=0.75)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


outlier_thresholds(df,"sum_gamerounds")
check_outlier(df,"sum_gamerounds")
replace_with_thresholds(df,"sum_gamerounds")
check_outlier(df,"sum_gamerounds")

# HYPOTHESIS

""" 
HO : M1 = M2 : There is no a statistically difference between A Version and B Version
H1 : M1 != M2 : There is a statistically difference between A Version and B Version) 
"""

# Let's take a look at the averages of version usage.

df.groupby("version").agg({"sum_gamerounds": "mean"})

# AB Testing (Independent Two-Sample T-Test)

# After checking the normality assumption and variance homogeneity,
# we will decide to apply a parametric or non-parametric test.

############################
# Normality Assumption
############################

# H0: Normal distribution assumption is provided.
# H1: The assumption of normal distribution is not provided.

test_stat, pvalue = shapiro(df.loc[df["version"] == "gate_30", "sum_gamerounds"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 0.7756, p-value = 0.0000

test_stat, pvalue = shapiro(df.loc[df["version"] == "gate_40", "sum_gamerounds"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 0.7732, p-value = 0.0000

"We reject H0 because p-value < 0.005."

############################
# Assumption of Variance Homogeneity
############################
# Actually, in this instance we do not need to check the homogeneity of variance.
# Because the assumption of normality was rejected.
# In this case automatically we should use non-parametric method.

# H0: Variances are Homogeneous
# H1: Variances Are Not Homogeneous

test_stat, pvalue = levene(df.loc[df["version"] == "gate_30", "sum_gamerounds"],
                           df.loc[df["version"] == "gate_40", "sum_gamerounds"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 0.8786, p-value = 0.3486

"We cannot reject H0,Variances are Homogeneous but we must use non-parametric way."

############################
# Mann Whitney U Test
############################

test_stat, pvalue = mannwhitneyu(df.loc[df["version"] == "gate_30", "sum_gamerounds"],
                                 df.loc[df["version"] == "gate_40", "sum_gamerounds"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 1024214124.5000, p-value = 0.0537

"We cannot reject H0,There is no a statistically difference between A Version and B Version "


# Functionalization A/B Test

def AB_Test(df, pthres=0.05):
    # H0:  There is no statistical difference between the gate_30 and the gate_40.

    print(df.groupby('version').agg({"sum_gamerounds": ["count", "mean"]}))

    print("NORMAL DISTRIBUTION ASSUMPTION".center(70, "*"))

    # H0 : The compared groups have a normal distribution

    pvalue_gate_30 = shapiro([df["version"] == "gate_30"])[1]
    pvalue_gate_40 = shapiro([df["version"] == "gate_40"])[1]

    print('p-value_gate30 = %.5f' % (pvalue_gate_30))
    print('p-value_gate40 = %.5f' % (pvalue_gate_40))

    if (pvalue_gate_30 < pthres) & (pvalue_gate_40 < pthres):
        print("Normality H0 is rejected.\n\n")
    else:
        print("Normality H0 is not rejected.\n")

    print("VARIANCE HOMOGENEOUS ASSUMPTION ".center(70, "*"))

    # H0 : The variance of compared groups is homegenous.

    p_value_levene = levene(df.loc[df["version"] == "gate_30", "sum_gamerounds"],
                            df.loc[df["version"] == "gate_40", "sum_gamerounds"])[1]

    print('p_value_levene = %.5f' % p_value_levene)

    if p_value_levene < pthres:
        print("Variance Homogeneity H0 is rejected.\n")
    else:
        print("Variance Homogeneity H0 is not rejected.\n")

    if ((pvalue_gate_30 > pthres) & (pvalue_gate_40 > pthres)) & (p_value_levene > pthres):
        p_value_ttest = ttest_ind(df.loc[df["version"] == "gate_30"],
                                  df.loc[df["version"] == "gate_40"],
                                  equal_var=True)[1]

        print('p_value_ttest = %.5f' % p_value_ttest)

    elif ((pvalue_gate_30 > pthres) & (pvalue_gate_40 > pthres)) & (p_value_levene < pthres):
        p_value_ttest = ttest_ind(df.loc[df["version"] == "gate_30"],
                                  df.loc[df["version"] == "gate_40"],
                                  equal_var=False)[1]

        print('p_value_ttest = %.5f' % p_value_ttest)
    else:
        print("Non-Parametric test should be done.\n\n")
        pvalue = mannwhitneyu(df.loc[df["version"] == "gate_30", "sum_gamerounds"],
                              df.loc[df["version"] == "gate_40", "sum_gamerounds"])[1]

        print('p_value = %.5f' % pvalue)

    print(" RESULT ".center(70, "*"))

    if pvalue < pthres:
        print(
            f"p-value {round(pvalue, 5)} < 0.05  H0 Hypothesis is Rejected. That is, there is a statistically significant difference between them.")

    else:
        print(
            f"p-value > {pthres} H0 is Not Rejected, That is, there is no statistically significant difference between them. The difference was made by chance.")


AB_Test(df, 0.05)










