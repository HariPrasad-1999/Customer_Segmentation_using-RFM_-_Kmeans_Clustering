import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, date, time
import os
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)




st.title("Customer Segmentation Using Clustering Algorithms")

def dataset(folder_path='./Datasets'):
    filenames = os.listdir(folder_path)
    selected_dataset = st.sidebar.selectbox("Select Dataset",filenames)
    return os.path.join(folder_path,selected_dataset)

filename=dataset()
st.info("You Selected {}".format(filename))

#Read Data
df=pd.read_csv(filename, encoding='unicode_escape')
#Show Dataset
if st.checkbox("Show Dataset"):
    number=st.number_input("Number of Rows to View",10)
    st.dataframe(df.head(number))
#Show Columns
if st.button("Columns Names"):
    st.write(df.columns)
# shape of database
if st.checkbox("Shape of Dataset"):
    st.write(df.shape)

#Remove  missing values from Customer ID column, can ignore missing values in description column
# Detailing the Country distribution and customerid
country_data = df[['Country','CustomerID']].drop_duplicates()
country_data.groupby(['Country']).agg({'CustomerID' : 'count'}).sort_values('CustomerID',ascending = False).reset_index().rename(columns = {'CustomerID':'CustomerID Count'})
df=df[pd.notnull(df['CustomerID'])]

df = df[df['Country'] == 'United Kingdom'].reset_index(drop = True)
df.shape
df.isna().sum()
df= df[pd.notnull(df['CustomerID'])]
df=df.query("Country=='United Kingdom'").reset_index(drop =True)
df.isnull().sum(axis=0)
#Checking the description of the data
df.describe()
df=df[pd.notnull(df['CustomerID'])]
df = df.query("Quantity > 0")
df.shape
#code to run totalamount and convert date
df=df[(df['Quantity']>0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
#print(dataset['InvoiceDate']) 
df['TotalAmount']= df['Quantity'] * df['UnitPrice']
if st.checkbox("Add Total AMOUNT To dataset"):
    st.write(df.shape)
    #print(dataset['TotalAmount']) 
    # 

st.write("""
# RFM Modelling :
""")

import datetime as  dt

st.write("""set Latest date 2011-12-10 as last invoice date was 2011-12-09.
this is to caluculate the number of days from recent purchase""")
def getfromdate():
    
    date=st.date_input('start date')
    ld=pd.to_datetime(date)
    return ld

latest_date=getfromdate()
st.write(latest_date)

RFMScores =df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (latest_date -x.max()).days, 'InvoiceNo':lambda x: len(x),
                                                                       'TotalAmount':lambda x: x.sum()})
                                                                        


#Convert Invoice Date into type int
RFMScores['InvoiceDate']= RFMScores['InvoiceDate'] = RFMScores['InvoiceDate'].astype(int)


#RENAME COLUMN NAMES TO RECENCY, FREQUENCY, AND MONETARY

RFMScores.rename(columns={'InvoiceDate':'Recency',
                         'InvoiceNo':'Frequency',
                         'TotalAmount':'Monetary'}, inplace =True)


#st.write(RFMScores.reset_index().head())
if st.checkbox("DEscriptive analytics",True):
    if st.checkbox("Recency describe"):
        st.write(RFMScores.Recency.describe())
    elif st.checkbox("Frequency describe"):
        st.write(RFMScores.Frequency.describe())
    elif st.checkbox("Monetary describe"):
        st.write(RFMScores.Monetary.describe())

#Split into four Segments using quantiles
quantiles =RFMScores.quantile(q=[0.25,0.5,0.75])
quantiles =quantiles.to_dict()
#quantiles
#FUNCTIONS TO CREATE R, F AND M SEGMENTS
#imp
def RScoring(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4
def FnMScoring(x,p,d):
    if x <=d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1
#calculate ADD R, F AND M SEGMENT VALUE COLUMNS IN THE EXISTING DATASET TO SHOW R, F, AND M SEGMENT VALUES

RFMScores['R']= RFMScores['Recency'].apply(RScoring, args=('Recency',quantiles,))
RFMScores['F']= RFMScores['Frequency'].apply(FnMScoring, args=('Frequency',quantiles,))
RFMScores['M']= RFMScores['Monetary'].apply(FnMScoring, args=('Monetary',quantiles,))

RFMScores.head()
#calculate and Add RFM group value column showing combined concatenated score of RFM
RFMScores['RFMGroup']=RFMScores.R.map(str) + RFMScores.F.map(str) + RFMScores.M.map(str)


#Calculate and Add RFMScore value column showing total sum of RFM Group values

RFMScores['RFMScore'] =RFMScores[['R' ,'F', 'M']].sum(axis = 1)
RFMScores.head()
#assign loyalty level to each customer

loyaltyLevel=['HIGHLY VISITED', 'REGULAR', 'OCCASIONALLY VISITED', 'NEW VISITORS']
scoreCuts= pd.qcut(RFMScores.RFMScore, q=4, labels=loyaltyLevel)
RFMScores['RFM_Loyalty_Level']=scoreCuts.values

RFMScores.reset_index()
RFMScores[RFMScores['RFMGroup']=='111'].sort_values('Monetary', ascending=False).reset_index().head(10)


##data visulization

#st.subheader("Data Visulization")
#if st.checkbox("Correlation Plot[Seaborn]"):
#    fig, ax = plt.subplots()
#    sns.heatmap(df.corr(), ax=ax)
#    st.write(fig)


#Handle negative and zero values so as to handle infinte nubers during log transformation
def handle_neg_n_zero(num):
    if num <=0:
        return 1
    else:
        return num
#handle_neg_n_zero function to Recency and Monetary columns

RFMScores['Recency'] = [handle_neg_n_zero(x) for x in RFMScores.Recency]
RFMScores['Monetary'] = [handle_neg_n_zero(x) for x in RFMScores.Monetary]


#performing log transformation to bring data into normal or near normal distribution

Log_Tfd_data = RFMScores[['Recency' , 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)


from sklearn.preprocessing import StandardScaler

# bring the data on same scale
scaleobj =StandardScaler()
Scaled_Data = scaleobj.fit_transform(Log_Tfd_data)

Scaled_Data = pd.DataFrame(Scaled_Data, index= RFMScores.index, columns = Log_Tfd_data.columns)
from sklearn.cluster import KMeans

sumOfSqDist = {}
for k in range(1,15):
    km=KMeans(n_clusters=k, init='k-means++', max_iter=1000)
    km=km.fit(Scaled_Data)
    sumOfSqDist[k] = km.inertia_
    
    
#plot the graph for sum of square distance values and Number of Clusters
if st.checkbox("run Elblow Method"):
    sns.pointplot(x= list(sumOfSqDist.keys()),y= list(sumOfSqDist.values()))
    plt.xlabel('number of Clusters(k)')
    plt.ylabel('Sum of Square Distances')
    plt.title('Elbow Method For Optimal K')
    plt.show()
    st.pyplot()


#perform K-mean clustering or build the K-means clustering model

KMean_clust = KMeans(n_clusters= 4, init='k-means++', max_iter =1000)
KMean_clust.fit(Scaled_Data)
RFMScores['Cluster']= KMean_clust.labels_
RFMScores.tail()


from  matplotlib import pyplot as plt
plt.figure(figsize=(7,7))

##Scatter plot frequency VS Recency
colors=["red", "green", "blue","black"]
RFMScores['Color']= RFMScores['Cluster'].map(lambda p: colors[p])
ax=RFMScores.plot(kind="scatter",
                 x="Recency",
                 y="Frequency",
                 figsize=(10,8),
                 c = RFMScores['Color'])
st.pyplot()





