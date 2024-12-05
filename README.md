# ENV3040-Project
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# Initalize variables
path_excel_2014 = "epa_everglades_emap_2014_data.xlsx"
path_excel_2013 = "epa_everglades_emap_2013_data.xlsx"

# For my excel 2014 I need to join the table to get subarea4
excel_df_2014_field = pd.read_excel(path_excel_2013, sheet_name = 0)
excel_df_2014_field.columns = excel_df_2014_field.loc[0]
excel_df_2014_field = excel_df_2014_field[["STATION", "SUBAREA4"]].drop(index=0).dropna().reset_index(drop=True)
excel_df_2014_field

# Excel field tab, station column has leading zeros in field data sheet need to remove the zeros
excel_df_2014_field.iloc[:21, 0] = excel_df_2014_field.iloc[:21, 0].apply(lambda x: x[1:] if x.startswith("0") else x)

excel_df_2014 = pd.read_excel(path_excel_2014, sheet_name = 1)
excel_df_2014.head()

excel_df_2013 = pd.read_excel(path_excel_2013, sheet_name = 1)
excel_df_2013.head()

# Setting this row as the dataframe column header
excel_df_2014.columns = excel_df_2014.loc[1]  

# 2013 column name is located at row 0
excel_df_2013.columns = excel_df_2013.loc[0]

df_4_columns_2014 = excel_df_2014[["THGSDFC", "THGSWFC", "TPSDFB", "TPSWFB", "Station "]]
df_4_columns_2014 = df_4_columns_2014.drop(index = [0,1,2]).dropna().reset_index(drop=True)
#.drop(index = [0,1,2]).dropna()
df_4_columns_2014.head()

df_4_columns_2014 = pd.merge(df_4_columns_2014, excel_df_2014_field, left_on = "Station ", right_on = "STATION")\
.drop(columns= {"Station ", "STATION"})
df_4_columns_2014.head()

df_4_columns_2013 = excel_df_2013[["THGSDFC", "THGSWFC", "TPSDFB", "TPSWFB", "SUBAREA 4"]].drop(index = [0, 1]).dropna()
df_4_columns_2013 = df_4_columns_2013.reset_index(drop = True)
df_4_columns_2013.head()

# remove the duplicate column and save it to a csv file
df_4_columns_2014.iloc[:, [0,1,2,4, 5]].to_csv("epa_everglades_emap_2014_data.csv")
df_4_columns_2013.iloc[:, [0,1,2,4, 5]].to_csv("epa_everglades_emap_2013_data.csv")

# Converting excel to csv
path_csv_2014 = "epa_everglades_emap_2014_data.csv"
path_csv_2013 = "epa_everglades_emap_2013_data.csv"

df2014 = pd.read_csv(path_csv_2014, index_col = "Unnamed: 0")
df2013 = pd.read_csv(path_csv_2013, index_col = "Unnamed: 0")

display(df2014.shape)
df2014.head()

display(df2013.shape)
df2013.head()

# Data description
def data_description(df, col_name, year):
    print(f"The mean value of {col_name} is:", df[col_name].mean())
    print(f"The median value of {col_name} is:", df[col_name].median())
    print(f"The standard deviation value of {col_name} is:", df[col_name].std())
    my_color = "lightblue"
    if year == 2013:
        my_color = "blue"
    plt.figure(figsize = (10,8))
    df[col_name].plot(kind = "hist", bins = 30, color = my_color, edgecolor = "orange", 
                   title = f"{col_name} hist for {year}", fontsize = 15)
    plt.show()

**##THGSDFC**

data_description(df2014, "THGSDFC", 2014)
data_description(df2013, "THGSDFC", 2013)

**##THGSWFC**
data_description(df2014, "THGSWFC", 2014)
data_description(df2013, "THGSWFC", 2013)

**#TPSDFB**
data_description(df2014, "TPSDFB", 2014)
data_description(df2013, "TPSDFB", 2013)

**#TPSWFB**
data_description(df2014, "TPSWFB", 2014)
data_description(df2013, "TPSWFB", 2013)

# Correlation
df2014.head()

# 2014 DATA
# Between THGSDFC and THGSWFC
st.pearsonr(df2014["THGSDFC"], df2014["THGSWFC"]).statistic

# Between THGSDFC and TPSDFB
st.pearsonr(df2014["THGSDFC"], df2014["TPSDFB"]).statistic

# Between TPSDFB and TPSWFB
st.pearsonr(df2014["TPSDFB"], df2014["TPSWFB"]).statistic

# Between THGSWFC and TPSWFB
st.pearsonr(df2014["THGSWFC"], df2014["TPSWFB"]).statistic

# 2013 DATA

# Between THGSDFC and THGSWFC
st.pearsonr(df2013["THGSDFC"], df2013["THGSWFC"]).statistic

# Between THGSDFC and TPSDFB
st.pearsonr(df2013["THGSDFC"], df2013["TPSDFB"]).statistic

# Between TPSDFB and TPSWFB
st.pearsonr(df2013["TPSDFB"], df2013["TPSWFB"]).statistic

# Between THGSWFC and TPSWFB
st.pearsonr(df2013["THGSWFC"], df2013["TPSWFB"]).statistic

# Linear regression

def linear_reg_and_plot(df, col_name1, col_name2, year):
    slope, intercept, r, p, stderr = st.linregress(df[col_name1], df[col_name2])
    print("slope of this linear regression is:", slope)
    print("interception of linear regression is:", intercept)
    print("r_value of linear regression is:", r)
    print("p_value of linear regression is:", p)
    plt.figure(figsize = (10, 8))
    plt.scatter(df[col_name1], df[col_name2], color = "green")
    plt.xlabel(col_name1, fontsize = 20)
    plt.ylabel(col_name2, fontsize = 20)
    plt.title(f"Linear regression plot for {year} between {col_name1} and {col_name2}", fontsize = 15)
    plt.plot(df[col_name1], slope * df[col_name1] + intercept, color = "orange")

**#THGSDFC vs THGSWFC Linear**
linear_reg_and_plot(df2014, "THGSDFC", "THGSWFC", 2014)

linear_reg_and_plot(df2013, "THGSDFC", "THGSWFC", 2013)

**#THGSDFC vs TPSDFB**
linear_reg_and_plot(df2014, "THGSDFC", "TPSDFB", 2014)
linear_reg_and_plot(df2013, "THGSDFC", "TPSDFB", 2013)

**#TPSDFB vs TPSWFB Linear**
linear_reg_and_plot(df2014, "TPSDFB", "TPSWFB", 2014)
linear_reg_and_plot(df2013, "TPSDFB", "TPSWFB", 2013)

**#THGSWFC vs TPSWFB Linear**
linear_reg_and_plot(df2014, "THGSWFC", "TPSWFB", 2014)
linear_reg_and_plot(df2013, "THGSWFC", "TPSWFB", 2013)

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df2014.head()

label2014 = df2014["SUBAREA4"]
df2014 = df2014.drop(columns= ["SUBAREA4"])

df2014_standard = StandardScaler().fit_transform(df2014)
df2014_standard

model = PCA(n_components = 2)

df2014_pca = model.fit_transform(df2014_standard)
df2014_pca_df = pd.DataFrame(df2014_pca)
df2014_pca_df.columns = ["PC1", "PC2"]
df2014_pca_df.head()

importance2014 = model.explained_variance_ratio_
importance2014


def plot_barScree(importance, year):
    plt.figure(figsize = (6, 10))
    sns.barplot(x = [1,2], y = importance)
    plt.title(f"Scree plot for PCA variance ratio importanc {year}", fontsize = 15)
    plt.xticks([0,1], ["PC1", "PC2"])
    plt.xlabel("Pricipal Component", fontweight = "bold")
    plt.ylabel("Variance Explained Ratio", fontweight = "bold")
    plt.grid(True)
    plt.show()

plot_barScree(importance2014, 2014)

def visualize_pca(df, label, year):
    plt.figure(figsize = (10, 10))
    sns.scatterplot(x = "PC1", y= "PC2", data = df, hue = label, style = label, s=100 )
    plt.ylabel("PC2", fontsize = 20)
    plt.xlabel("PC1", fontsize = 20)
    plt.title(f"PCA plot for {year}")
    plt.grid()
    plt.show()

visualize_pca(df2014_pca_df, label2014, 2014)

df2013.head()

label2013 = df2013["SUBAREA 4"]
df2013 = df2013.drop(columns= ["SUBAREA 4"])

df2013_standard = StandardScaler().fit_transform(df2013)
df2013_standard

df2013_pca = model.fit_transform(df2013_standard)
df2013_pca_df = pd.DataFrame(df2013_pca)
df2013_pca_df.columns = ["PC1", "PC2"]
df2013_pca_df.head()

importance2013 = model.explained_variance_ratio_
importance2013

plot_barScree(importance2013, 2013)

visualize_pca(df2013_pca_df, label2013, 2013)

sample1 = df2013["THGSDFC"]
sample2 = df2014["THGSDFC"]
display(sample1.head(5))
display(sample2.head(5))

def t_test(sample1, sample2, column_name): 
    print(f"==================T_Test on column {column_name}===================")
    t, p = st.ttest_ind(sample1, sample2)
    print("P_value of those two samples is:", p)
    alpha = 0.05
    if p < alpha:
        print("Reject the null hypothesis, there is a significant difference")
    else:
        print("Fail to reject the null hypothesis, there is no significant difference")
    print("=" * 61)

t_test(df2013["THGSDFC"], df2014["THGSDFC"], "THGSDFC")

t_test(df2013["THGSWFC"], df2014["THGSWFC"], "THGSWFC")

t_test(df2013["TPSDFB"], df2014["TPSDFB"], "TPSDFB")

t_test(df2013["TPSWFB"], df2014["TPSWFB"], "TPSWFB")
