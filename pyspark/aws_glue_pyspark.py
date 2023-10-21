import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# Set up GlueContext and SparkContext
glueContext = GlueContext(SparkContext.getOrCreate())
spark = glueContext.spark_session
job = Job(glueContext)

# Get the input and output S3 paths from the command line arguments
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'input_path', 'output_path'])
input_path = args['input_path']
output_path = args['output_path']

# Step 1: Read titanic.csv stored in the first S3
titanic_data = spark.read.option("header", "true").csv(input_path + "/titanic.csv")

# Step 2: Store in a data frame
titanic_data.createOrReplaceTempView("titanic_data")

# Step 3: Fill in missing values with the median value of each column using SQL
titanic_data = spark.sql('''
    SELECT *,
           IF(age IS NULL, AVG(age) OVER() AS age,
           IF(fare IS NULL, AVG(fare) OVER() AS fare,
           IF(sibsp IS NULL, AVG(sibsp) OVER() AS sibsp,
           IF(parch IS NULL, AVG(parch) OVER() AS parch
      FROM titanic_data
''')

# Step 4: Separate numeric columns (X_num) and character columns (X_cat) for separate processing in SQL
titanic_data.createOrReplaceTempView("titanic_data_filled")
titanic_data_num = spark.sql('''
    SELECT pclass, age, sibsp, parch, fare FROM titanic_data_filled
''')
titanic_data_cat = spark.sql('''
    SELECT sex, embark_town, who, adult_male, alone FROM titanic_data_filled
''')

# Step 5: Convert all character columns (X_cat) to numeric with One-Hot-Encoding
titanic_data_cat_encoded = titanic_data_cat

# Your One-Hot-Encoding logic goes here

# Step 6: Merge X_num and X_cat now that they are all numeric data
titanic_data_merged = titanic_data_num.join(titanic_data_cat_encoded)

# Step 7: Store the data processed in 3~6 in the second S3
titanic_data_merged.write.option("header", "true").csv(output_path + "/processed_titanic")

job.commit()