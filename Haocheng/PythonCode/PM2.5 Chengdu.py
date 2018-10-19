
# coding: utf-8

# In[1]:


import findspark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
spark = SparkSession.builder.appName('PM2.5 Chengdu').getOrCreate()


# In[2]:


# Read from the data source
df = spark.read.csv('PMDataSets/ChengduPM20100101_20151231.csv',header=True,inferSchema=True)


# In[ ]:


df.printSchema()


# In[ ]:


df.describe().toPandas()


# In[ ]:


df.select('PM_US Post').describe().show()


# In[3]:


# Data Cleaning: Features Selection.
df_1 = df.drop('No','year','month','day','hour','season','PM_Caotangsi','PM_Shahepu')
# Data Cleaning: Remove instances with 'NA' value.
df_2 = df_1.filter(df['PM_US Post']!='NA')
df_3 = df_2.filter(df['DEWP']!='NA')
df_4 = df_3.filter(df['HUMI']!='NA')
df_5 = df_4.filter(df['PRES']!='NA')
df_6 = df_5.filter(df['TEMP']!='NA')
df_7 = df_6.filter(df['cbwd']!='NA')
df_8 = df_7.filter(df['Iws']!='NA')
df_9 = df_8.filter(df['precipitation']!='NA')
df_10 = df_9.filter(df['Iprec']!='NA')
df = df_10


# In[ ]:


# This step shows the inconsistency of the data type.
df.printSchema()
df.describe().toPandas()


# In[4]:


# Data Cleaning: Manually changing the data type.
df = df.withColumn('PM_US Post', df['PM_US Post'].cast("double"))
df = df.withColumn('DEWP', df['DEWP'].cast("double"))
df = df.withColumn('HUMI', df['HUMI'].cast("double"))
df = df.withColumn('PRES', df['PRES'].cast("double"))
df = df.withColumn('TEMP', df['TEMP'].cast("double"))
df = df.withColumn('Iws', df['Iws'].cast("double"))
df = df.withColumn('precipitation', df['precipitation'].cast("double"))
df = df.withColumn('Iprec', df['Iprec'].cast("double"))
df.printSchema()
df.describe().toPandas()


# In[5]:


# Data Cleaning: New feature construction
def judgement(x):
    if (x>=0) & (x<75):
        return 0
    elif (x>=75) & (x<150):
        return 1
    else:
        return 2
udf_judgement = udf(judgement, IntegerType())
df = df.withColumn('Harm',udf_judgement(df['PM_US Post']))
# Changing cbwd to numeric expression
def cbwd_vc(x):
    if x=='cv':
        return 0
    elif x=='SW':
        return 1
    elif x=='SE':
        return 2
    elif x=='NE':
        return 3
    else:
        return 4
udf_cbwd_vc = udf(cbwd_vc, IntegerType())
df = df.withColumn('cbwd',udf_cbwd_vc(df['cbwd']))
df.describe().toPandas()


# In[6]:


# Data Transformation: Further reduction
df = df.drop('PM_US Post')
# Data Transformation: Correlation Check
print("DEWP-Harm Correlation:",df.corr('DEWP','Harm'))
print("HUMI-Harm Correlation:",df.corr('HUMI','Harm'))
print("PRES-Harm Correlation:",df.corr('PRES','Harm'))
print("TEMP-Harm Correlation:",df.corr('TEMP','Harm'))
print("cbwd-Harm Correlation:",df.corr('cbwd','Harm'))
print("Iws-Harm Correlation:",df.corr('Iws','Harm'))
print("Precipitation-Harm Correlation:",df.corr('precipitation','Harm'))
print("Iprec-Harm Correlation:",df.corr('Iprec','Harm'))


# In[7]:


# Data Transformation: Further reduction according to correlation
df = df.drop('precipitation','Iprec')
df.toPandas()


# In[ ]:


# Changing to pandas dataframe for Data Balancing: OverSampling
df_pandas = df.toPandas()
print(df_pandas['Harm'].value_counts())
df_0 = df_pandas[df_pandas['Harm'] == 0]
df_1 = df_pandas[df_pandas['Harm'] == 1]
df_2 = df_pandas[df_pandas['Harm'] == 2]
df_1_new = df_1.sample(frac=1.64,replace=True)
df_2_new = df_2.sample(frac=4.3, replace=True)
df_pandas = pd.concat([df_0, df_1_new, df_2_new], ignore_index=True)
df_pandas.sample(frac=1)
print(df_pandas['Harm'].value_counts())
df = spark.createDataFrame(df_pandas)
df.toPandas()


# In[8]:


# Preparing for machine learning
cbwd_Indexer = StringIndexer(inputCol = 'cbwd', outputCol = 'cbwdIndex')
cbwd_encoder = OneHotEncoder(inputCol = 'cbwdIndex', outputCol = 'cbwdVec')

Harm_Indexer = StringIndexer(inputCol = 'Harm', outputCol = 'label')

assembler = VectorAssembler(inputCols=['DEWP','HUMI','PRES','cbwdVec','TEMP','Iws'], outputCol="features")


# In[9]:


# Pipeline
pipeline = Pipeline(stages=[cbwd_Indexer, Harm_Indexer, cbwd_encoder, assembler])
pipeline_model = pipeline.fit(df)
pipe_df = pipeline_model.transform(df)
pipe_df = pipe_df.select('label', 'features')
pipe_df.printSchema()


# In[ ]:


# Decision Tree Classifier
from pyspark.ml.classification import DecisionTreeClassifier
train_data, test_data = pipe_df.randomSplit([0.8,0.2])
print("Training Dataset Count: " + str(train_data.count()))
print("Test Dataset Count: " + str(test_data.count()))
dt = DecisionTreeClassifier(featuresCol='features',labelCol='label')
dt_model = dt.fit(train_data)
predictions = dt_model.transform(test_data)
predictions.select("prediction", "label", "features").show(5)


# In[10]:


# Randomforest Classifier
from pyspark.ml.classification import RandomForestClassifier
train_data, test_data = pipe_df.randomSplit([0.8,0.2])
print("Training Dataset Count: " + str(train_data.count()))
print("Test Dataset Count: " + str(test_data.count()))
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label',numTrees=30,  maxDepth = 6)
rf_Model = rf.fit(train_data)
predictions = rf_Model.transform(test_data)
predictions.select("prediction", "label", "features").show(5)


# In[ ]:


# Evaluating Accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % (accuracy))


# In[ ]:


rf_Model.featureImportances


# In[11]:


importances = rf_Model.featureImportances
plt.bar(['DEWP','HUMI','PRES','cbwd01','cbwd02','cbwd03','cbwd04','TEMP','Iws'],importances)
plt.title("Feature Importance")
plt.xticks(rotation=90)
plt.show()


# In[13]:


df_pandas = df.toPandas()

plt.scatter(df_pandas['PRES'],df_pandas['Harm'])
plt.title("Pressure, Harm Diagram")
plt.xlabel("Pressure")
plt.ylabel("Harm")
plt.show()

plt.scatter(df_pandas['Harm'],df_pandas['Iws'])
plt.title("Wind Speed, Harm Diagram")
plt.xlabel("Harm")
plt.ylabel("Iws")
plt.show()

plt.scatter(df_pandas['HUMI'],df_pandas['TEMP'],df_pandas['Harm']*20, alpha=0.3)
plt.title("HUMI, TEMP, Harm Diagram")
plt.xlabel("Humidity")
plt.ylabel("Temperature")
plt.legend(["Harm"])
plt.show()

