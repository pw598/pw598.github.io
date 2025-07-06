---
layout: post
title:  "MongoDB II: Aggregation Pipelines with PyMongo"
date:   2025-06-21 00:00:00 +0000
categories: MongoDB SQL Python
---

In this article, we'll continue to work with the Kirana Store clickstream data, creating an aggregated collection with fewer records, and performing some analytics. I started with the intent to include some SQL-analogies, and ended up providing a full companion notebook to make the Mongo code more relatable.


<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/mg2.png" style="height: 350px; width:auto;">



# Outline

1. Introduction
2. The Kirana Store Clickstream Data
3. MongoDB Aggregation Pipelines
4. Import Libraries and Data
5. Select DB and View Collections
6. Data Exploration
7. Classify Device Type (Bot, Desktop, or Mobile)
8. Export Flattened Data to CSV
9. Create <code>users_weekly</code> Collecton
10. Plot Summary Statistics
11. What's Next?



# Introduction

To briefly recap, the last article included the basics of operating MongoDB through the Mongo shell, Bash, or Python (PyMongo), such as for basic queries and CRUD operations. Because this article is a little more involved, we'll narrow the focus to PyMongo, though as I mentioned above, an SQL notebook (using Google Colab) is provided as a companion piece. For myself at least, this makes the PyMongo code a lot more relatable.

To be honest, it took quite a bit of troubleshooting to get matching results in SQL (though of course in hindsight, it's clear what the issues were). Flattening the data was relatively straightforward, as there aren't many layers of nesting in the clickstream data. Matching top-line results, user-level results, and country-level results was more of a process. AI assistance (from Grok) was helpful, but although it is very impressive, and useful, recommendations toward complex issues often lacked either precision or quality. Trickle-charging with step-by-step input, and modularizing code into attachments were productive strategies, though limitations were still bumped up against, as has historically been the case.



# The Kirana Store <code>clickstream</code> Data

As a reminder, while we refer to structured databases as containing tables full of records, which themselves contain fields, we refer to unstructured databases as containing collections of documents, which themselves contain fields. A document from the data we are dealing with looks something like this:

```js
{'_id': ObjectId('60df102aad74d9467c94272a'),
 'webClientID': 'WI10000021937',
 'VisitDateTime': datetime.datetime(2018, 5, 23, 14, 27, 15, 118000),
 'ProductID': 'Pr100472',
 'Activity': 'click',
 'device': {'Browser': 'Safari', 'OS': 'Mac OS X'},
 'user': {'UserID': 'U100095', 'Country': 'Turkey'}}
```

This is similar to Javascript Object Notation (JSON), though <code>ObjectId</code> is a MongoDB-specific data type, the <code>datetime</code> element is a Python data type, and true JSON would include double-quotes instead of apostrophes.

It's also the case with unstructured data that the fields above may not exist for all records, and other records may contain data (like <code>user.City</code>) which the above does not.

The dataset is available here, in a .zip file <a href="https://drive.google.com/file/d/1ZRrNKa9sBtyRi1jZ5ocFMrcuuY_erVYH/view?usp=drive_link">on Google Drive</a>.

The PyMongo Jupyter notebook is available <a href="https://github.com/pw598/Articles/blob/main/MongoDB-II-Aggregation-Pipelines.ipynb">here</a>, and the MySQL companion piece, which uses a Google Colab notebook (for replicability) is located <a href="https://github.com/pw598/Articles/blob/main/MySQL-Companion-to-MongoDB-II-Aggregation-Pipelines.ipynb">here</a>.



# MongoDB Aggregation Pipelines

We refer to the stage-based framework of aggregation in MongoDB as aggregation pipelines, which analyze and transform data into filtered, aggregated, or calculated results. These stages include operations like:

- <code>$match</code>: acts like <code>WHERE</code> in SQL, filters the document stream to those matching particular criteria.

- <code>$sum</code>: acts like <code>SUM()</code> in SQL, I assume this one is self-explanatory.

- <code>$count</code>: acts like <code>COUNT()</code> in SQL, refers to count of documents in the stream.

- <code>$group</code>: acts like <code>GROUP BY</code> in SQL, combines multiple documents with the same field, fields, or expression into a single document according to a group key, resulting in one document per unique group key.

- <code>$project</code>: reshapes each document in the stream, such as by adding new fields or removing existing fields (like <code>SELECT</code>, except you specify fields to include with a <code>1</code>, and to not include with a <code>0</code>).

- <code>$sort</code>: acts like <code>ORDER BY</code> in SQL, sorting the documents in the stream (takes as an argument <code>1</code> to specify ascending order, and <code>-1</code> to specify descending order).

- <code>$limit</code>: acts like <code>LIMIT</code> in SQL, truncating the number of documents returned in the results.

- <code>$unwind</code>: deconstructs an array field from the input documents, outputting a separate document for each element in the array.

- <code>$addFields</code>: adds new fields to documents in the aggregation pipeline, preserving existing fields (like <code>SELECT</code> combined with a calculated field).

- <code>$lookup</code>: performs a left outer join to another collection in the same database.

For more, see <a href="https://www.mongodb.com/docs/manual/reference/operator/aggregation-pipeline/">here</a> for the user docs on 'stages', and <a href="https://www.mongodb.com/docs/manual/reference/operator/aggregation/">here</a> for the user docs on 'operators'.



# Import Libraries and Data

## Import Libraries

We'll be using the following libraries. First install them using the conda shell, <code>!pip</code>, or <code>pip</code> if necessary, depending on your environment.

```python
# to connect to a MongoDB instance
from pymongo import MongoClient
# for certain types of data manipulation
import pandas as pd
# for nicer document printing
import pprint as pp
# for working with datetime objects
from datetime import datetime
# for timing operations
import time
# for exporting checkpoints
import subprocess
# for plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```



## Establish a MongoDB Connection

Next, we establish a connection to a MongoDB instance, and print the list of databases currently in existence.


```python
client = MongoClient("mongodb://localhost:27017/")
print(client.list_database_names())
```

<p></p>

```python
# ['admin', 'config', 'local']
```

In my case, we see only the system-related data.



### Drop <code>clickstream</code> if Exists (Optional)

If the <code>clickstream</code> database exists from prior tinkering, and you wish to delete it, use the following.

```python
db_name = "clickstream"
client.drop_database(db_name)
print(client.list_database_names())
```


## Import Data

Next, we import the data from the <code>clicks.bson</code> and <code>clicks.metadata.json</code> files. The following cell is about setting variables to be used in the next cell below, and you can go ahead and skip to the following cell if hard-coding.

```python
HOST = "localhost"
PORT = "27017"
DBNAME = "clickstream"
IMPORT_FILE_FOLDER = "data"
BSON_FILE_NAME = "clicks"
JSON_FILE_NAME = "clicks.metadata"
bson_file = f"{IMPORT_FILE_FOLDER}\\{BSON_FILE_NAME}.bson"
json_file = f"{IMPORT_FILE_FOLDER}\\{JSON_FILE_NAME}.json"
collection_bson = BSON_FILE_NAME
collection_json = JSON_FILE_NAME
```

The following shell commands, facilitated in Jupyter notebook via the preceding exclamation mark, will import the data from file, so long as you have the Mongo tools package (referenced in the prior article) installed. The <code>--drop</code> command will result in any existing data from the same database and collection being cleared before the import.


```bash
!mongorestore --host {HOST}:{PORT} --db {DBNAME} --collection {collection_bson} --drop "{bson_file}"
!mongoimport --host {HOST}:{PORT} --db {DBNAME} --collection {collection_json} --drop --type json "{json_file}"
```

<p></p>

```bash
# 2025-06-08T13:59:20.871-0600  finished restoring clickstream.clicks (6100000 documents, 0 failures)
# 2025-06-08T13:59:20.872-0600  no indexes to restore for collection clickstream.clicks
# 2025-06-08T13:59:20.873-0600  6100000 document(s) restored successfully. 0 document(s) failed to restore.
# 2025-06-08T13:59:21.819-0600  connected to: mongodb://localhost:27017/
# 2025-06-08T13:59:21.821-0600  dropping: clickstream.clicks.metadata
# 2025-06-08T13:59:21.853-0600  1 document(s) imported successfully. 0 document(s) failed to import.
```


# Select DB and View Collections


With the database created and data imported, we next list out the collection names.


```python
db_name = "clickstream"
db = client[db_name]
collections = db.list_collection_names()
print(collections)
```

<p></p>

```python
# ['clicks.metadata', 'clicks']
```



# Data Exploration

## Sample Record


Next, view a sample record of the data.


```python
collection = db['clicks']
collection.find_one()
```

<p></p>

```python
# {'_id': ObjectId('60df1029ad74d9467c91a932'),
#  'webClientID': 'WI100000244987',
#  'VisitDateTime': datetime.datetime(2018, 5, 25, 4, 51, 14, 179000),
#  'ProductID': 'Pr100037',
#  'Activity': 'click',
#  'device': {'Browser': 'Firefox', 'OS': 'Windows'},
#  'user': {'City': 'Colombo', 'Country': 'Sri Lanka'}}
```


## Count of Records

Though the number of documents is mentioned upon import, we can also use the following command. To apply a single filter, we would use something like <code>collection.count_documents({"user.City": "Colombo"})</code>, with either apostrophes or double-quotes being acceptable. For a double-filter, we would use something like <code>collection.count_documents({"user.City": "Colombo", "device.Browser": "Firefox"})</code>, and so on.


```python
collection.count_documents({})
```

<p></p>

```python
# 6100000
```


## Get Date Range


<p>To get the range of dates in the data, we can use something like the following. This is our first pipeline, in which the <code>$group</code> operator is used to aggregate documents. Setting <code>_id</code> to <code>None</code> effectively removes any grouping by specific field values, and treats the entire collection as one group. Within this group, we compute the earliest and latest values of the <code>VisitDateTime</code>.</p>


```python
pipeline = [
    {
        "$group": {
            "_id": None,
            "minDate": {"$min": "$VisitDateTime"},
            "maxDate": {"$max": "$VisitDateTime"}
        }
    }
]

result = collection.aggregate(pipeline)
for doc in result:
    pp.pprint(doc)
```

<p></p>

```python
# {'_id': None,
#  'maxDate': datetime.datetime(2018, 5, 27, 23, 59, 59, 576000),
#  'minDate': datetime.datetime(2018, 5, 7, 0, 0, 1, 190000)}
```


## Count of Unique <code>webClientID</code> Values


<code>webClientID</code> values group individual visits by user, so long as they connect using the same device and system as prior visits. As we'll see later, the data also include a <code>user.UserID</code> field for those who have created (and logged into) an account, however a minority of documents contain that information, whereas all documents contain a <code>webClientID</code>. This introduces the <code>$count</code> operator.


```python
pipeline = [
    { "$group": { "_id": "$webClientID" } },
    { "$count": "uniqueCount" }
]

result = list(collection.aggregate(pipeline))
num_unique = result[0]['uniqueCount'] if result else 0
print(f"Number of unique webClientID values: {num_unique}")
```

<p></p>

```python
# Number of unique webClientID values: 1091455
```

With smaller data (less than 16MB in Jupyter notebook), we could do this in simpler fashion, using the <code>distinct</code> keyword.

```python
# collection = db['clicks']
# len(collection.distinct('webClientID'))
```

It is analogous to <code>COUNT(DISTINCT ...)</code> in SQL:

```sql
SELECT COUNT(DISTINCT webClientID)
FROM clicks;
```


## Count of <code>webClientID</code> Values with a <code>user.UserID</code>


Next, we'll see how many instances of <code>webClientID</code> correspond to having a <code>user.UserID</code>. This introduces the <code>$match</code> operator, which lets us filter to only records where the field <code>user.UserID</code> exists, and is not equal (<code>$ne</code>) to being <code>None</code> (null).


```python
result = collection.distinct(
    "webClientID",
    {"user.UserID": {"$exists": True, "$ne": None}}
)

len(result)
```

<p></p>

```python
# 36791
```

Clearly, only a small proportion of visitors have an account. The SQL analog to the above would be as follows.


```sql
SELECT COUNT(DISTINCT webClientID)
FROM clicks
WHERE user_UserID IS NOT NULL;
```



## Count of Unique <code>user.UserID</code> Values


The number of unique user IDs is slightly smaller still.


```python
pipeline = [
    {"$match": {"user.UserID": {"$exists": True, "$ne": None}}},
    {"$group": {"_id": "$user.UserID"}},
    {"$count": "uniqueUserIDs"}
]

result = collection.aggregate(pipeline)
count = next(result, {"uniqueUserIDs": 0})["uniqueUserIDs"]
print(count)
```

<p></p>

```python
# 34050
```

The SQL analog:

```sql
SELECT COUNT(DISTINCT user_UserID)
FROM clicks;
```



# Classify Device Type as Bot, Desktop, or Mobile

Let's suppose we are interested in understanding our customers' device-related preferences or behavior. For this, we can utilize the nested <code>device.OS</code> and <code>device.Browser</code> fields. The operating systems are a good indicator of whether a user is on a mobile or desktop device, and the browsers may offer a clear indication of whether a visitor is a robot. I relied on AI to make the classifications, so forgive any technical inaccuracies, but we will use the logic that a visitor is a robot if the browser indicates so, and this classification will take precedence over distinctions in operating system. If not a robot, we will classify the visitor as either desktop or mobile based on operating system, and label each document accordingly by adding a field called <code>device_type</code>.


## Distinct Values for <code>device.OS</code>

To get the list of unique operating systems, we can use the <code>distinct</code> keyword as follows. The second line simply provides the output as a horizontal list to save space.

```python
os_list = collection.distinct("device.OS")
print(", ".join(map(str, os_list)))
```

<p></p>

```python
# Android, BlackBerry OS, Chrome OS, Chromecast, Fedora, FreeBSD, Kindle, Linux, Mac OS X, NetBSD, OpenBSD, Other, Solaris, Tizen, Ubuntu, Windows, Windows Phone, iOS
```

The SQL analog would be as follows for a vertical list:

```sql
SELECT COUNT(DISTINCT device_OS) AS Num_Device_OS
FROM clicks
WHERE device_OS IS NOT NULL;
```

Or as follows for a horizontal one:

```sql
SELECT STRING_AGG(DISTINCT device_OS, ', ') AS os_list
FROM clicks
WHERE device_OS IS NOT NULL; 
```


## Distinct Values for <code>device.Browser</code>


Similarly, for the list of unique browsers:


```python
os_list = collection.distinct("device.Browser")
print(", ".join(map(str, os_list)))
```

<p></p>

```python
# AdsBot-Google, AhrefsBot, Amazon Silk, Android, AppEngine-Google, Apple Mail, BingPreview, BlackBerry WebKit, Chrome, Chrome Mobile, Chrome Mobile WebView, Chrome Mobile iOS, Chromium, Coc Coc, Coveobot, Crosswalk, Dragon, DuckDuckBot, Edge, Edge Mobile, Electron, Epiphany, Facebook, FacebookBot, Firefox, Firefox Mobile, Firefox iOS, HbbTV, HeadlessChrome, HubSpot Crawler, IE, IE Mobile, Iceweasel, Iron, JobBot, Jooblebot, K-Meleon, Kindle, Konqueror, Magus Bot, Mail.ru Chromium Browser, Maxthon, Mobile Safari, Mobile Safari UI/WKWebView, MobileIron, NetFront, Netscape, Opera, Opera Coast, Opera Mini, Opera Mobile, Other, PagePeeker, Pale Moon, PetalBot, PhantomJS, Pinterest, Puffin, Python Requests, QQ Browser, QQ Browser Mobile, Radius Compliance Bot, Safari, Samsung Internet, SeaMonkey, Seekport Crawler, SiteScoreBot, Sleipnir, Sogou Explorer, Thunderbird, UC Browser, Vivaldi, WebKit Nightly, WordPress, Yandex Browser, YandexAccessibilityBot, YandexBot, YandexSearch, Yeti, YisouSpider, moatbot, net/bot
```


## Classify as Bot, Desktop, or Mobile 


The categorizations are as follows, in Python list format:


```python
mobile_os = [
    'Android', 'iOS', 'Windows Phone', 'BlackBerry OS', 'Tizen', 'Kindle', 'Chromecast'
]
desktop_os = [
    'Windows', 'Mac OS X', 'Linux', 'Ubuntu', 'Fedora', 'FreeBSD', 
    'NetBSD', 'OpenBSD', 'Solaris', 'Chrome OS', 'Other'
]
bot_browsers = [
    'AdsBot-Google', 'AhrefsBot', 'BingPreview', 'DuckDuckBot', 'FacebookBot',
    'HubSpot Crawler', 'JobBot', 'Jooblebot', 'Magus Bot', 'PetalBot',
    'Radius Compliance Bot', 'Seekport Crawler', 'SiteScoreBot', 'YandexBot',
    'YandexAccessibilityBot', 'YandexSearch', 'Yeti', 'YisouSpider', 'moatbot',
    'net/bot', 'AppEngine-Google', 'PagePeeker', 'Pinterest', 'Facebook',
    'Python Requests', 'Coveobot', 'HeadlessChrome', 'PhantomJS', 'WordPress'
]
```


The classification operation will leverage Python, with a bit of PyMongo, using the <code>$set</code> command for updating:


```python
# normalize list elements to lower-case
mobile_os = [x.lower() for x in mobile_os]
desktop_os = [x.lower() for x in desktop_os]
bot_browsers = [x.lower() for x in bot_browsers]

# report progress while updating
record_count = 0
progress_interval = 100000

# start a timer
start_time = time.time()

# loop through records in the collection
for record in collection.find({}, {'device.OS': 1, 'device.Browser': 1}):
    browser = record.get('device', {}).get('Browser', '').lower().strip()
    os = record.get('device', {}).get('OS', '').lower().strip()

    # determine device_type for record
    if browser in bot_browsers:
        device_type = 'bot'
    elif os in mobile_os:
        device_type = 'mobile'
    elif os in desktop_os:
        device_type = 'desktop'
    else:
        device_type = None
    
    # update the record's device_type
    result = collection.update_one(
        {"_id": record['_id']},
        {"$set": {
            "device_type": device_type
        }}
    )
    
    # increment count
    if result.modified_count > 0:
        record_count += 1

    # report progress
    if record_count % progress_interval == 0 and record_count > 0:
        print(f"Processed {record_count} records")

elapsed_time = time.time() - start_time
print(f"Completed: Updated device_type for {record_count} records in {elapsed_time:.0f} seconds")
```

<p></p>

```python
# Processed 100000 records
# Processed 200000 records
# ...
# Processed 6000000 records
# Processed 6100000 records
# Completed: Updated device_type for 6100000 records in 2638 seconds
```

The SQL analog, which I'll write as if we are using SQL magic commands in a Python notebook, is as follows.


```sql
%sql UPDATE clicks \
SET device_type = \
    CASE \
        WHEN LOWER(TRIM(device_Browser)) IN ( \
            'adsbot-google', 'ahrefsbot', 'bingpreview', 'duckduckbot', 'facebookbot', \
            'hubspot crawler', 'jobbot', 'jooblebot', 'magus bot', 'petalbot', \
            'radius compliance bot', 'seekport crawler', 'sitescorebot', 'yandexbot', \
            'yandexaccessibilitybot', 'yandexsearch', 'yeti', 'yisouspider', 'moatbot', \
            'net/bot', 'appengine-google', 'pagepeeker', 'pinterest', 'facebook', \
            'python requests', 'coveobot', 'headlesschrome', 'phantomjs', 'wordpress' \
        ) THEN 'bot' \
        WHEN LOWER(TRIM(device_OS)) IN ( \
            'android', 'ios', 'windows phone', 'blackberry os', 'tizen', 'kindle', 'chromecast' \
        ) THEN 'mobile' \
        WHEN LOWER(TRIM(device_OS)) IN ( \
            'windows', 'mac os x', 'linux', 'ubuntu', 'fedora', 'freebsd', \
            'netbsd', 'openbsd', 'solaris', 'chrome os', 'other' \
        ) THEN 'desktop' \
        ELSE NULL \
    END;
# Record end time and calculate duration
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.0f} seconds")
```


The above PyMongo code used <code>update_one</code> to make reporting progress easier, but a bulk-write method is also available, and we could even weave in some progress tracking, such as the following (somewhat confusingly, the below will leverage a PyMongo function called <code>UpdateOne</code>).


```python
# from pymongo import UpdateOne

# mobile_os = [x.lower() for x in mobile_os]
# desktop_os = [x.lower() for x in desktop_os]
# bot_browsers = [x.lower() for x in bot_browsers]

# batch_size = 500000
# operations = []
# records_processed = 0
# records_written = 0

# # start a timer
# start_time = time.time()

# for record in collection.find({}, {'device.OS': 1, 'device.Browser': 1}):
#     browser = record.get('device', {}).get('Browser', '').lower().strip()
#     os = record.get('device', {}).get('OS', '').lower().strip()

#     if browser in bot_browsers:
#         device_type = 'bot'
#     elif os in mobile_os:
#         device_type = 'mobile'
#     elif os in desktop_os:
#         device_type = 'desktop'
#     else:
#         device_type = None
    
#     operations.append(
#         UpdateOne(
#             {"_id": record['_id']},
#             {"$set": {"device_type": device_type}}
#         )
#     )
    
#     records_processed += 1
#     if records_processed % batch_size == 0:
#         print(f"Read {records_processed} records")

#     if len(operations) >= batch_size:
#         collection.bulk_write(operations)
#         records_written += len(operations)
#         print(f"Written {records_written} records")
#         operations = []

# if operations:
#     collection.bulk_write(operations)
#     records_written += len(operations)
#     print(f"Written {records_written} records")

# elapsed_time = time.time() - start_time
# print(f"Completed: Updated device_type in {elapsed_time:.0f} seconds")
```

<p></p>

```python
# Read 500000 records
# Written 500000 records
# Read 1000000 records
# Written 1000000 records
# ...
# Read 5500000 records
# Written 5500000 records
# Read 6000000 records
# Written 6000000 records
# Written 6100000 records
# Completed: Updated device_type in 920 seconds
```


## Post-Update Records Inspection


If all went well, we should fail to see any records that were skipped over (unless they didn't contain the corresponding <code>device</code> fields), and the below confirms that this is the case.

```python
count = collection.count_documents({ "device_type": None })
print(f"Number of records with device_type == None: {count}")
```

<p></p>

```python
# Number of records with device_type == None: 0
```


## Export a Checkpoint


With all that work and all that waiting done, you may want to export the current state of the database to file, such that you can skip re-doing the <code>device_type</code> classification, if for some reason you decide to delete the database or collection. Unless you are working in an environment like Google Colab, your data should persist even if you close your session.


```python
# this will create the json metadata file as well

export_folder = r'data\checkpoint'

subprocess.run([
    'mongodump',
    '--host', 'localhost',
    '--port', '27017',
    '--db', 'clickstream',
    '--collection', 'clicks',
    '--out', export_folder
], check=True)

print(f"Exported to {export_folder}")
```

<p></p>

```python
# Exported to data\checkpoint
```

The SQL analog (via MySQL through Bash):

```bash
mysqldump --host=localhost --port=3306 --user=root --password --databases clickstream --tables clicks > data/checkpoint/clickstream_clicks.sql
```

If you've been following along, including the requisite waiting, this might be a good time to grab a coffee, or go for a walk.



## Import From Checkpoint

You can import the data from the checkpoint file as follows. The process is identical to the original import.

```python
HOST = "localhost"
PORT = "27017"
DBNAME = "clickstream"
IMPORT_FILE_FOLDER = r"data\checkpoint\clickstream"
BSON_FILE_NAME = "clicks"
JSON_FILE_NAME = "clicks.metadata"
bson_file = f"{IMPORT_FILE_FOLDER}\\{BSON_FILE_NAME}.bson"
json_file = f"{IMPORT_FILE_FOLDER}\\{JSON_FILE_NAME}.json"
collection_bson = BSON_FILE_NAME
collection_json = JSON_FILE_NAME
```

<p></p>

```bash
!mongorestore --host {HOST}:{PORT} --db {DBNAME} --collection {collection_bson} --drop "{bson_file}"
!mongoimport --host {HOST}:{PORT} --db {DBNAME} --collection {collection_json} --drop --type json "{json_file}"
```

The MySQL via Bash analog would look something like the following.


```bash
# create DB first if doesn't exist
!mysql --host=localhost --port=3306 --user=root --password -e "CREATE DATABASE IF NOT EXISTS clickstream;"

# delete records from table if they exist
!mysql --host=localhost --port=3306 --user=root --password -e "TRUNCATE TABLE clickstream.clicks;"

# import from file
!mysql --host=localhost --port=3306 --user=root --password -e clickstream < checkpoint/clickstream_clicks.sql
```



# Export Flattened Data to CSV

It's a very conceivable use-case that we may want to bring data from an unstructured MongoDB database, such as used in the early stages of analysis in a data lake, into a structured SQL data warehouse with enforcable schema, normalizing entity relationships, etc. Below, we will 'flatten' the data such that nested fields are brought out of their hierarchy, and assign null values in the places where the data do not exist. The MySQL Colab notebook linked to above, after some library installations and imports, will import this flattened data, and provide query analogies to illuminate our understanding of the increasingly complex MongoDB operations. This flattening and export operation is performed using pandas, after exporting the data as a list from the MongoDB collection.


```python
collection = db['clicks']

start_time = time.time()

# Fetch data
data = list(collection.find())

elapsed_time = time.time() - start_time
print(f"Fetched records in {elapsed_time:.0f} seconds")
start_time = time.time()

# Flatten nested fields
flattened_data = []
for doc in data:
    flat_doc = {}
    def flatten(d, parent=''):
        for k, v in d.items():
            new_key = f"{parent}{k}" if parent else k
            if isinstance(v, dict):
                flatten(v, f"{new_key}.")
            else:
                flat_doc[new_key] = v
    flatten(doc)
    flattened_data.append(flat_doc)

# Convert to DataFrame
df = pd.DataFrame(flattened_data)
# Replace periods with underscores in column names
df.columns = df.columns.str.replace('.', '_')

# Export to CSV
df.to_csv(r'clicks_flattened.csv', index=False)

elapsed_time = time.time() - start_time
print(f"Completed: Exported to structured format CSV in {elapsed_time:.0f} seconds")
```

<p></p>

```python
# Fetched records in 79 seconds
# Completed: Exported to structured format CSV in 177 seconds
```

Notice that the nested fields now stand alone, with a format like <code>user_City</code> instead of the hierarchical <code>user.City</code>.


```python
df = pd.DataFrame(pd.read_csv('clicks_structured.csv', nrows=5))
list(df.columns)
```

<p></p>

```python
# ['_id',
#  'webClientID',
#  'VisitDateTime',
#  'ProductID',
#  'Activity',
#  'device_Browser',
#  'device_OS',
#  'user_City',
#  'user_Country',
#  'device_type',
#  'user_UserID']
```

The SQL equivalent of the above:

```sql
USE clickstream;
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'clicks';
```

As a sanity check, we'll count the records in the pandas dataframe exported to CSV, expecting 6.1M, as with the <code>.bson</code> data originally imported.

```python
df = pd.DataFrame(pd.read_csv('clicks_structured.csv'))
len(df)
```

<p></p>

```python
# 6100000
```



# Create <code>users_weekly</code> Collection

We can see that a large number of documents (6.1M) have accrued only over a relatively short amount of time (3 weeks). It may be helpful to create a collection with aggregated, and perhaps filtered data to facilitate shorter run-time of queries for analytical purposes. It would be optimal to retain some level of date information, so we'll aggregate to the week level. I'll filter to only users with accounts as well, and to ensure consistency with the SQL operations at a country level, include a breakout by country. If we wanted to ensure consistent results at the city level as well, we would have to either break things out further, or ensure a consistent method of assigning only a single city per user.

It will help, in terms of speed, to add an index to the <code>user.UserID</code> field. In MongoDB, it will be a sparse index, meaning it only applies to the minority of documents which contain that field. 


## Add a Sparse Index to user.UserID

```python
# add a userID index
db.clicks.create_index([("user.UserID", 1)], sparse=True)
```

The SQL analog:

```sql
CREATE INDEX idx_UserID ON clicks (user_UserID(255) ASC);
SHOW INDEXES FROM clicks WHERE Key_name = 'idx_UserID';
```


## Create Aggregated Collection

The aggregation pipeline will involve 5 stages, and introduce the usage of many of the pipeline operators mentioned earlier on:

1. Filter to Valid Records (Users)
- i.e., <code>user.UserID</code> exists
- Uses <code>$match</code>

2. Add Fields
- Normalize <code>device_type</code> to lower case
- Calculate week number, with a start date of Monday
- Extract date, so that we can include the number of unique days visited
- Uses <code>$addFields</code>

3. Group by User, Week, and Country
- Groups clicks and pageloads at the aggregated level 
- Uses <code>$group</code>

4. Compute Metrics per Device Type
- Uses <code>$project</code>

5. Write to New Collection 
- Uses <code>$out</code>


Admittedly, it's a lot to take in, so I'll provide plenty of comments in the PyMongo code, and break tradition with the above by providing the SQL analog first.

SQL:

```sql
%%sql

-- create a table to store weekly aggregated user activity data
CREATE TABLE users_weekly (
    userID VARCHAR(255),      -- unique identifier for each user
    weeknum INT,              -- week number in year (Monday as start of week)
    numDays INT,              -- distinct active days for user in week
    City VARCHAR(100),        -- user city
    Country VARCHAR(100),     -- user country
    pageloads_mobile BIGINT,  -- mobile pageload events
    pageloads_desktop BIGINT, -- desktop pageload events
    pageloads_bot BIGINT,     -- bot pageload events
    clicks_mobile BIGINT,     -- mobile click events
    clicks_desktop BIGINT,    -- desktop click events
    clicks_bot BIGINT,        -- bot click events
    PRIMARY KEY (userID, weeknum, Country)  -- composite primary key
);

INSERT INTO users_weekly
SELECT
    user_UserID AS userID,
    WEEK(DATE_SUB(VisitDateTime, INTERVAL 1 DAY), 3) AS weeknum, 
    COUNT(DISTINCT DATE_FORMAT(VisitDateTime, '%Y-%m-%d')) AS numDays,
    ANY_VALUE(user_City) AS City,
    COALESCE(user_Country, 'Null') AS Country,
    SUM(CASE WHEN Activity = 'pageload' AND device_type = 'mobile' THEN 1 ELSE 0 END) AS pageloads_mobile,
    SUM(CASE WHEN Activity = 'pageload' AND device_type = 'desktop' THEN 1 ELSE 0 END) AS pageloads_desktop,
    SUM(CASE WHEN Activity = 'pageload' AND device_type = 'bot' THEN 1 ELSE 0 END) AS pageloads_bot,
    SUM(CASE WHEN Activity = 'click' AND device_type = 'mobile' THEN 1 ELSE 0 END) AS clicks_mobile,
    SUM(CASE WHEN Activity = 'click' AND device_type = 'desktop' THEN 1 ELSE 0 END) AS clicks_desktop,
    SUM(CASE WHEN Activity = 'click' AND device_type = 'bot' THEN 1 ELSE 0 END) AS clicks_bot
FROM clicks
WHERE user_UserID IS NOT NULL AND device_type IS NOT NULL
GROUP BY user_UserID, user_Country, WEEK(DATE_SUB(VisitDateTime, INTERVAL 1 DAY), 3);
```


PyMongo:

```python
collection = db["clicks"]

# Define aggregation pipeline
pipeline = [

    # Stage 1: Filter to Valid Records
    # Match documents with valid user.UserID
    {"$match": {"user.UserID": {"$exists": True, "$ne": None}}},

    # Stage 2: Add Fields
    # normalize device_type
    {
        "$addFields": {
            "device_type": {"$toLower": "$device_type"},

            # calculate week number, subtracting 1 day to align with week start
            "weeknum": {
                "$week": {
                    "$dateSubtract": {"startDate": "$VisitDateTime", "unit": "day", "amount": 1}
                }
            },

            # extract date for grouping by day
            "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$VisitDateTime"}}
        }
    },
    # Stage 3: Group by User, Week, and Country
    {
        # grouping
        "$group": {
            "_id": {"userID": "$user.UserID", "weeknum": "$weeknum", "Country": "$user.Country"},

            # collect unique days visited using a set to avoid duplicates
            "uniqueDays": {"$addToSet": "$day"},

            # retain the first city value for each group
            "City": {"$first": "$user.City"},

            # push device_type and activity counts (pageload, click) into an array
            "counts": {
                "$push": {
                    "device_type": "$device_type",
                    "pageload": {"$cond": [{"$eq": ["$Activity", "pageload"]}, 1, 0]},
                    "click": {"$cond": [{"$eq": ["$Activity", "click"]}, 1, 0]}
                }
            }
        }
    },
    # Stage 4: Compute Metrics per Device Type
    # project the final structure, calculating pageloads and clicks by device type
    {
        "$project": {

            # exclude the _id field from output
            "_id": 0,

            # extract userID, weeknum, and Country from the grouped _id
            "userID": "$_id.userID",
            "weeknum": "$_id.weeknum",
            "Country": "$_id.Country",

            # calculate the number of unique days
            "numDays": {"$size": "$uniqueDays"},
            "City": 1,

            # calculate total pageloads for mobile devices
            "pageloads.mobile": {

                # sum the results of the mapped array to get the total number of clicks
                "$sum": {

                    # transform the 'counts' array to extract click counts for bot devices
                    "$map": {

                        # input array containing device_type, pageload, and click data
                        "input": "$counts",

                        # lias for each element in the counts array
                        "as": "count",

                        # expression to process each element
                        "in": {

                            # return the click count if device_type is 'mobile', otherwise 0
                            "$cond": [

                                # check if the device_type of the current element is 'mobile'
                                {"$eq": ["$$count.device_type", "mobile"]},

                                # if true, return the click count (1 or 0) for this element
                                "$$count.pageload",
                                0
                            ]
                        }
                    }
                }
            },

            # calculate total pageloads for desktop devices
            # analogous process to described with comments above 
            "pageloads.desktop": {
                "$sum": {
                    "$map": {
                        "input": "$counts",
                        "as": "count",
                        "in": {
                            "$cond": [
                                {"$eq": ["$$count.device_type", "desktop"]},
                                "$$count.pageload",
                                0
                            ]
                        }
                    }
                }
            },

            # calculate total pageloads for bot devices
            "pageloads.bot": {
                "$sum": {
                    "$map": {
                        "input": "$counts",
                        "as": "count",
                        "in": {
                            "$cond": [
                                {"$eq": ["$$count.device_type", "bot"]},
                                "$$count.pageload",
                                0
                            ]
                        }
                    }
                }
            },

            # calculate total clicks for mobile devices
            "clicks.mobile": {
                "$sum": {
                    "$map": {
                        "input": "$counts",
                        "as": "count",
                        "in": {
                            "$cond": [
                                {"$eq": ["$$count.device_type", "mobile"]},
                                "$$count.click",
                                0
                            ]
                        }
                    }
                }
            },

            # calculate total clicks for desktop devices
            "clicks.desktop": {
                "$sum": {
                    "$map": {
                        "input": "$counts",
                        "as": "count",
                        "in": {
                            "$cond": [
                                {"$eq": ["$$count.device_type", "desktop"]},
                                "$$count.click",
                                0
                            ]
                        }
                    }
                }
            },

            # calculate total clicks for bot devices
            "clicks.bot": {
                "$sum": {
                    "$map": {
                        "input": "$counts",
                        "as": "count",
                        "in": {
                            "$cond": [
                                {"$eq": ["$$count.device_type", "bot"]},
                                "$$count.click",
                                0
                            ]
                        }
                    }
                }
            }
        }
    },
    # Stage 5: Write results to the 'users_weekly' collection
    # output the transformed data to the 'users_weekly' collection
    {"$out": "users_weekly"}
]

# Execute pipeline
collection.aggregate(pipeline)
```

We can check the top-line results as follows:


```python
collection = db['users_weekly']

# Aggregation pipeline
pipeline = [
    {
        "$group": {
            "_id": None,  # Group all documents into a single group
            "total_pageloads_mobile": {"$sum": "$pageloads.mobile"},
            "total_pageloads_desktop": {"$sum": "$pageloads.desktop"},
            "total_pageloads_bot": {"$sum": "$pageloads.bot"},
            "total_clicks_mobile": {"$sum": "$clicks.mobile"},
            "total_clicks_desktop": {"$sum": "$clicks.desktop"},
            "total_clicks_bot": {"$sum": "$clicks.bot"}
        }
    },
    {
        "$project": {
            "_id": 0,  # Exclude _id field
            "total_pageloads_mobile": 1,
            "total_pageloads_desktop": 1,
            "total_pageloads_bot": 1,
            "total_clicks_mobile": 1,
            "total_clicks_desktop": 1,
            "total_clicks_bot": 1
        }
    }
]

# Execute the aggregation
result = list(collection.aggregate(pipeline))

# Print result
if result:
    print(result[0])
else:
    print("No documents found.")
```

<p></p>

```python
# {'total_pageloads_mobile': 46649, 'total_pageloads_desktop': 132217, 'total_pageloads_bot': 62, 'total_clicks_mobile': 39884, 'total_clicks_desktop': 383409, 'total_clicks_bot': 72}
```

If you are following along in the SQL notebook, you will see that the results match.


<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/mg2-1.png" style="height: 60px; width:auto;">


Breaking it out by country:


PyMongo:

```python
pipeline = [
    {
        "$group": {
            "_id": "$Country",
            "total_count": {
                "$sum": {
                    "$add": [
                        "$pageloads.mobile",
                        "$pageloads.desktop",
                        "$pageloads.bot",
                        "$clicks.mobile",
                        "$clicks.desktop",
                        "$clicks.bot"
                    ]
                }
            }
        }
    },
    {
        "$sort": {"total_count": -1}
    }
]

result = db["users_weekly"].aggregate(pipeline)
for doc in list(result)[0:10]:
    print(doc)
```

<p></p>

```python
# {'_id': 'India', 'total_count': 452969}
# {'_id': 'United States', 'total_count': 29541}
# {'_id': None, 'total_count': 16416}
# {'_id': 'United Kingdom', 'total_count': 6182}
# {'_id': 'Singapore', 'total_count': 4436}
# {'_id': 'Australia', 'total_count': 4173}
# {'_id': 'Nigeria', 'total_count': 4012}
# {'_id': 'Pakistan', 'total_count': 3757}
# {'_id': 'Canada', 'total_count': 3397}
# {'_id': 'Germany', 'total_count': 3336}
```


SQL:

```sql 
%%sql

SELECT Country,
       SUM(pageloads_mobile) +
       SUM(pageloads_desktop) +
       SUM(pageloads_bot) +
       SUM(clicks_mobile) +
       SUM(clicks_desktop) +
       SUM(clicks_bot) as total_count
FROM users_weekly
GROUP BY Country
ORDER BY total_count DESC;
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/mg2-2.png" style="height: 250px; width:auto;">


## Create a Checkpoint

This would be a good time to write the <code>users_weekly</code> collection to a file.

```python
export_folder = r'data\checkpoint'

subprocess.run([
    'mongodump',
    '--host', 'localhost',
    '--port', '27017',
    '--db', 'clickstream',
    '--collection', 'users_weekly',
    '--out', export_folder
], check=True)

print(f"Exported to {export_folder}")
```

<p></p>

```python
# Exported to data\checkpoint
```

In MySQL, using Bash:

```bash
!mysqldump -e clickstream users_weekly > users_weekly.sql
```


# Plot Summary Statistics

We could do something similar based on <code>webClientID</code> for the 'non-users', though without horizontal scaling, the operations would take fairly long, and I would like to avoid getting repetitive. What we'll do below is create a 2 x 1 subplot of charts, with the first breaking out activity type by country, for the top 5 countries (though you can easily adjust that number), and the bottom plotting clicks vs. pageloads by user for those countries, with the dots being color-coded by country.

Pretty much all of the aggregation pipeline operators mentioned above will be used in the functions for aggregating data from the <code>users_weekly</code> collection, and I'll leave it largely to the extensive comments below to explain the code. If you're like me, what's most illustrative are the SQL analogies, which I'll lay out first.



### The <code>fetch_top_countries</code> Function

The <code>fetch_top_countries(n)</code> function is used to first determine which countries to show. The criteria is the amount of total activity (pageloads plus clicks), and <code>n</code> is obviously the number of countries to select, by total activity in descending order.

```python
def fetch_top_countries(n):
    query_top_countries = f"""
```
```sql
    SELECT Country
    FROM (
        SELECT Country, SUM(pageloads_mobile + 
                            pageloads_desktop + 
                            pageloads_bot + 
                            clicks_mobile + 
                            clicks_desktop + 
                            clicks_bot) as total_count
        FROM users_weekly
        WHERE Country IS NOT NULL AND Country != 'Null'
        GROUP BY Country
        ORDER BY total_count DESC
        LIMIT {n}
    )
```
```python
    """
```
```python
    return pd.read_sql(query_top_countries, engine)['Country'].tolist()
```

What that looks like in PyMongo is the following:


```python
def fetch_top_countries(collection, n):
    pipeline = [
        # filter documents with valid Country field
        {"$match": {"Country": {"$exists": True, "$ne": None}}},
        # convert pageloads and clicks to arrays
        {"$project": {"Country": 1, "pageloads": {"$objectToArray": "$pageloads"}, "clicks": {"$objectToArray": "$clicks"}}},
        # unwind pageloads and clicks arrays
        {"$unwind": "$pageloads"},
        {"$unwind": "$clicks"},
        # group by Country, summing pageloads and clicks
        {"$group": {"_id": "$Country", "total_count": {"$sum": {"$add": ["$pageloads.v", "$clicks.v"]}}}},
        # sort by total count in descending order
        {"$sort": {"total_count": -1}},
        # limit to top N countries
        {"$limit": n},
        # project to extract Country name
        {"$project": {"_id": 0, "Country": "$_id"}}
    ]
    return [doc['Country'] for doc in collection.aggregate(pipeline)]
```



### The <code>fetch_bar_data</code> Function

Getting the sum of pageloads and clicks by country for the bar chart at the top of the subplot are the following two queries wrapped in a Python function, for SQL:


```python
def fetch_bar_data(top_countries):
    countries_str = ','.join(f"'{c}'" for c in top_countries)
    query_pageloads = f"""
```
```sql
    SELECT Country, SUM(pageloads_mobile + pageloads_desktop + pageloads_bot) as pageloads_count
    FROM users_weekly
    WHERE Country IS NOT NULL AND Country != 'Null' AND Country IN ({countries_str})
    GROUP BY Country
```
```python
    """
```
```python
    query_clicks = f"""
```
```sql
    SELECT Country, SUM(clicks_mobile + clicks_desktop + clicks_bot) as clicks_count
    FROM users_weekly
    WHERE Country IS NOT NULL AND Country != 'Null' AND Country IN ({countries_str})
    GROUP BY Country
```
```python
    """
```
```python
    pageloads_data = pd.read_sql(query_pageloads, engine)
    clicks_data = pd.read_sql(query_clicks, engine)

    countries = top_countries
    pageloads_counts = [pageloads_data[pageloads_data['Country'] == c]['pageloads_count'].iloc[0] if c in pageloads_data['Country'].values else 0 for c in countries]
    clicks_counts = [clicks_data[clicks_data['Country'] == c]['clicks_count'].iloc[0] if c in clicks_data['Country'].values else 0 for c in countries]

    return countries, pageloads_counts, clicks_counts
```


And the PyMongo equivalent is as follows:


```python
def fetch_bar_data(collection, countries):
    # pipeline to aggregate pageloads by country
    pipeline = [
        # filter documents with valid Country field and specified countries
        {"$match": {"Country": {"$exists": True, "$ne": None, "$in": countries}}},
        # convert pageloads object to array for unwinding
        {"$project": {"Country": 1, "pageloads": {"$objectToArray": "$pageloads"}}},
        # unwind pageloads array to process each entry
        {"$unwind": "$pageloads"},
        # group by Country, summing pageload values
        {"$group": {"_id": "$Country", "pageloads_count": {"$sum": "$pageloads.v"}}},
        # project to rename _id to Country and keep pageloads_count
        {"$project": {"_id": 0, "Country": "$_id", "pageloads_count": 1}}
    ]
    pageloads_data = list(collection.aggregate(pipeline))

    # pipeline to aggregate clicks by country
    pipeline = [
        # filter documents with valid Country field and specified countries
        {"$match": {"Country": {"$exists": True, "$ne": None, "$in": countries}}},
        # convert clicks object to array for unwinding
        {"$project": {"Country": 1, "clicks": {"$objectToArray": "$clicks"}}},
        # unwind clicks array to process each entry
        {"$unwind": "$clicks"},
        # group by Country, summing click values
        {"$group": {"_id": "$Country", "clicks_count": {"$sum": "$clicks.v"}}},
        # project to rename _id to Country and keep clicks_count
        {"$project": {"_id": 0, "Country": "$_id", "clicks_count": 1}}
    ]
    clicks_data = list(collection.aggregate(pipeline))

    # Ensure all specified countries are included, defaulting to 0 if no data
    pageloads_counts = [next((d['pageloads_count'] for d in pageloads_data if d['Country'] == c), 0) for c in countries]
    clicks_counts = [next((d['clicks_count'] for d in clicks_data if d['Country'] == c), 0) for c in countries]

    return countries, pageloads_counts, clicks_counts
```


Finally, the SQL version of the function to grab the scatter plot data is as follows:


```python
def fetch_scatter_data(top_countries):
    countries_str = ','.join(f"'{c}'" for c in top_countries)
    query_scatter = f"""
```
```sql
    SELECT userID, Country,
           SUM(pageloads_mobile + pageloads_desktop + pageloads_bot) as total_pageloads,
           SUM(clicks_mobile + clicks_desktop + clicks_bot) as total_clicks
    FROM users_weekly
    WHERE Country IS NOT NULL AND Country != 'Null' AND Country IN ({countries_str})
    GROUP BY userID, Country
```
```python
    """
```
```python
    return pd.read_sql(query_scatter, engine)
```


And the PyMongo equivalent is:


```python
def fetch_scatter_data(collection, countries):
    pipeline = [
        # filter documents for specified countries
        {"$match": {"Country": {"$exists": True, "$ne": None, "$in": countries}}},
        # group by userID and Country, summing pageloads and clicks
        {"$group": {
            "_id": {"userID": "$userID", "Country": "$Country"},
            "totalPageloads": {"$sum": {"$add": ["$pageloads.mobile", "$pageloads.desktop", "$pageloads.bot"]}},
            "totalClicks": {"$sum": {"$add": ["$clicks.mobile", "$clicks.desktop", "$clicks.bot"]}}
        }},
        # add totalActivity field for sorting
        {"$addFields": {"totalActivity": {"$add": ["$totalPageloads", "$totalClicks"]}}},
        # sort by total activity in descending order
        {"$sort": {"totalActivity": -1}},
        # project relevant fields
        {"$project": {"_id": 0, "userID": "$_id.userID", "Country": "$_id.Country", "totalPageloads": 1, "totalClicks": 1}}
    ]
    return list(collection.aggregate(pipeline))
```


Putting the PyMongo functions all together, along with the plotting functions, we have:


```python
collection = db['users_weekly']

# function to fetch top N countries by total pageloads and clicks
def fetch_top_countries(collection, n):
    pipeline = [
        # filter documents with valid Country field
        {"$match": {"Country": {"$exists": True, "$ne": None}}},
        # convert pageloads and clicks to arrays
        {"$project": {"Country": 1, "pageloads": {"$objectToArray": "$pageloads"}, "clicks": {"$objectToArray": "$clicks"}}},
        # unwind pageloads and clicks arrays
        {"$unwind": "$pageloads"},
        {"$unwind": "$clicks"},
        # group by Country, summing pageloads and clicks
        {"$group": {"_id": "$Country", "total_count": {"$sum": {"$add": ["$pageloads.v", "$clicks.v"]}}}},
        # sort by total count in descending order
        {"$sort": {"total_count": -1}},
        # limit to top N countries
        {"$limit": n},
        # project to extract Country name
        {"$project": {"_id": 0, "Country": "$_id"}}
    ]
    return [doc['Country'] for doc in collection.aggregate(pipeline)]

# function to fetch bar plot data for specified countries
def fetch_bar_data(collection, countries):
    # pipeline to aggregate pageloads by country
    pipeline = [
        # filter documents with valid Country field and specified countries
        {"$match": {"Country": {"$exists": True, "$ne": None, "$in": countries}}},
        # convert pageloads object to array for unwinding
        {"$project": {"Country": 1, "pageloads": {"$objectToArray": "$pageloads"}}},
        # unwind pageloads array to process each entry
        {"$unwind": "$pageloads"},
        # group by Country, summing pageload values
        {"$group": {"_id": "$Country", "pageloads_count": {"$sum": "$pageloads.v"}}},
        # project to rename _id to Country and keep pageloads_count
        {"$project": {"_id": 0, "Country": "$_id", "pageloads_count": 1}}
    ]
    pageloads_data = list(collection.aggregate(pipeline))

    # pipeline to aggregate clicks by country
    pipeline = [
        # filter documents with valid Country field and specified countries
        {"$match": {"Country": {"$exists": True, "$ne": None, "$in": countries}}},
        # convert clicks object to array for unwinding
        {"$project": {"Country": 1, "clicks": {"$objectToArray": "$clicks"}}},
        # unwind clicks array to process each entry
        {"$unwind": "$clicks"},
        # group by Country, summing click values
        {"$group": {"_id": "$Country", "clicks_count": {"$sum": "$clicks.v"}}},
        # project to rename _id to Country and keep clicks_count
        {"$project": {"_id": 0, "Country": "$_id", "clicks_count": 1}}
    ]
    clicks_data = list(collection.aggregate(pipeline))

    # Ensure all specified countries are included, defaulting to 0 if no data
    pageloads_counts = [next((d['pageloads_count'] for d in pageloads_data if d['Country'] == c), 0) for c in countries]
    clicks_counts = [next((d['clicks_count'] for d in clicks_data if d['Country'] == c), 0) for c in countries]

    return countries, pageloads_counts, clicks_counts

# function to fetch scatter plot data for specified countries
def fetch_scatter_data(collection, countries):
    pipeline = [
        # filter documents for specified countries
        {"$match": {"Country": {"$exists": True, "$ne": None, "$in": countries}}},
        # group by userID and Country, summing pageloads and clicks
        {"$group": {
            "_id": {"userID": "$userID", "Country": "$Country"},
            "totalPageloads": {"$sum": {"$add": ["$pageloads.mobile", "$pageloads.desktop", "$pageloads.bot"]}},
            "totalClicks": {"$sum": {"$add": ["$clicks.mobile", "$clicks.desktop", "$clicks.bot"]}}
        }},
        # add totalActivity field for sorting
        {"$addFields": {"totalActivity": {"$add": ["$totalPageloads", "$totalClicks"]}}},
        # sort by total activity in descending order
        {"$sort": {"totalActivity": -1}},
        # project relevant fields
        {"$project": {"_id": 0, "userID": "$_id.userID", "Country": "$_id.Country", "totalPageloads": 1, "totalClicks": 1}}
    ]
    return list(collection.aggregate(pipeline))

# fetch top 5 countries
top_countries = fetch_top_countries(collection, 5)

# fetch bar plot data
bar_data = fetch_bar_data(collection, top_countries)

# fetch scatter plot data
scatter_data = fetch_scatter_data(collection, top_countries)

# prepare scatter plot data
countries_data = {country: {"pageloads": [], "clicks": [], "userIDs": []} for country in top_countries}
for doc in scatter_data:
    country = doc["Country"]
    countries_data[country]["pageloads"].append(doc["totalPageloads"])
    countries_data[country]["clicks"].append(doc["totalClicks"])
    countries_data[country]["userIDs"].append(doc["userID"])

# create subplot with 2 rows, 1 column
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Country Distribution (Top 5)', 'Pageloads vs Clicks by User'),
    vertical_spacing=0.15
)

# add bar plot traces
fig.add_trace(
    go.Bar(
        name='Pageloads',
        x=bar_data[0],
        y=bar_data[1],
        marker_color='#40E0D0',
        showlegend=True,
        legendgroup='bar',
        legend='legend1'
    ),
    row=1, col=1
)
fig.add_trace(
    go.Bar(
        name='Clicks',
        x=bar_data[0],
        y=bar_data[2],
        marker_color='#C71585',
        showlegend=True,
        legendgroup='bar',
        legend='legend1'
    ),
    row=1, col=1
)

# define color map for scatter plot
color_map = {
    top_countries[0]: "#FF6347",
    top_countries[1]: "#4682B4",
    top_countries[2]: "#32CD32",
    top_countries[3]: "#FFD700",
    top_countries[4]: "#9932CC"
}
# add scatter plot traces
for country in top_countries:
    if countries_data[country]["pageloads"]:
        fig.add_trace(
            go.Scatter(
                x=countries_data[country]["pageloads"],
                y=countries_data[country]["clicks"],
                mode='markers',
                name=country,
                marker=dict(size=10, color=color_map[country], opacity=0.5, line=dict(width=0.5, color='black')),
                text=countries_data[country]["userIDs"],
                hovertemplate="User: %{text}<br>Pageloads: %{x}<br>Clicks: %{y}<extra></extra>",
                showlegend=True,
                legendgroup='scatter',
                legend='legend2'
            ),
            row=2, col=1
        )

# update layout
fig.update_layout(
    barmode='group',
    title_text='Users Weekly: Country Distribution and Pageloads vs Clicks',
    template='plotly_dark',
    showlegend=True,
    legend1=dict(x=1.05, y=1.0, xanchor='left', yanchor='top', title='Bar Chart'),
    legend2=dict(x=1.05, y=0.2, xanchor='left', yanchor='bottom', title='Scatter Plot'),
    height=800
)

# update axes
fig.update_xaxes(title_text='Country', row=1, col=1)
fig.update_xaxes(title_text='Total Pageloads', range=[0, 750], row=2, col=1)
fig.update_yaxes(title_text='Count', row=1, col=1)
fig.update_yaxes(title_text='Total Clicks', range=[0, 5000], row=2, col=1)

# save and show plot
fig.write_html('top_5_country_pageloads_and_clicks.html')
fig.show()
```


Flip to light mode to render the following plots if you don't see them (sorry, working on it).


{% include top_5_country_pageloads_and_clicks.html %}


I don't mean to imply that working with the 6.1M-record data (minus those with null country values) is unmanageable; and with horizontal scaling, even a much larger number of records could be highly manageable. However, if you try running the below, you'll see that despite using the <code>clicks</code> collection (or the <code>clicks</code> table in SQL) to get activity data for users and non-users alike, the code executes pretty quickly. The queries are simple, so I'll skip the extensive comments and SQL analogies. 

In the first chart below, I'll show the total activity for India vs. all other (non-null) countries, broken out by <code>device_type</code>.


```python
collection = db["clicks"]

# Aggregate data
pipeline = [
    {"$match": {
        "user.Country": {"$ne": None, "$ne": "", "$exists": True},
        "device_type": {"$in": ["desktop", "mobile", "bot"]}
    }},
    {"$group": {
        "_id": {
            "device": "$device_type",
            "category": {"$cond": [{"$eq": ["$user.Country", "India"]}, "India", "Other"]}
        },
        "count": {"$sum": 1}
    }}
]
data = list(collection.aggregate(pipeline))

# Process data
device_types = ["desktop", "mobile", "bot"]
plot_data = {device: {"India": 0, "Other": 0} for device in device_types}

for item in data:
    device = item["_id"]["device"]
    category = item["_id"]["category"]
    count = item["count"]
    plot_data[device][category] = count

# Prepare bar plot data
india_counts = [plot_data[device]["India"] for device in device_types]
other_counts = [plot_data[device]["Other"] for device in device_types]
x_positions = [0, 1, 2]

# Create bar plot
fig = go.Figure(data=[
    go.Bar(
        x=x_positions,
        y=india_counts,
        name="India",
        text=india_counts,
        textposition="auto",
        marker_color="#1f77b4",
        offset=-0.2,
        width=0.4
    ),
    go.Bar(
        x=x_positions,
        y=other_counts,
        name="Other",
        text=other_counts,
        textposition="auto",
        marker_color="#ff7f0e",
        offset=0.2,
        width=0.4
    )
])

# Update layout
fig.update_layout(
    title="Record Count: India vs Other Countries by Device Type",
    xaxis=dict(
        title="Device Type",
        tickvals=x_positions,
        ticktext=device_types
    ),
    yaxis_title="Number of Records",
    barmode="group",
    template="plotly_white",
    height=500,
    legend=dict(x=0.85, y=1.0)
)

# Save and show
fig.write_html("india_vs_other_by_device_bar.html")
fig.show()
```

(Again, flip to light mode if not in it)


{% include india_vs_other_by_device_bar.html %}


Interestingly, with regard to desktop visitation, the sum of other countries are more prominent, but for mobile visitation, India is more prominent. We can see that bot visitation is also greater for the sum of other countries, but that overall, it accounts for a very low proportion of total visits.

Finally, we'll create a map-based visual, which with Plotly, you can zoom in and out of, plus drag in all 4 directions. I'll make the following design choices:

1. Include a dropdown-list filter for <code>device_type</code>.
2. Have the colormap reset each time a new option is chosen from the filter, to be relative only to the countries currently showing.
2. Omit India to avoid occlusion.
3. Use a logarithm-based colormap to avoid occlusion (due to a comparitively large amount of activity from the United States).


```python
# install pycountry to get abbreviations necessary for plotting
# !pip install pycountry
```

<p></p>

```python
import pycountry
import math

collection = db["clicks"]

# Aggregate data
pipeline = [
    {"$match": {
        "user.Country": {"$ne": None, "$ne": "", "$ne": "India", "$exists": True},
        "device_type": {"$ne": None, "$exists": True}
    }},
    {"$group": {"_id": {"country": "$user.Country", "device": "$device_type"}, "count": {"$sum": 1}}},
]
data = list(collection.aggregate(pipeline))

# Map country names to ISO 3-letter codes
country_code_map = {c.name: c.alpha_3 for c in pycountry.countries}
manual_map = {
    "United States": "USA",
    "United Kingdom": "GBR",
    "South Korea": "KOR",
    "Russia": "RUS",
    "Hashemite Kingdom of Jordan": "JOR",
    "Republic of Korea": "KOR",
    "Republic of Lithuania": "LTU",
    "Republic of Moldova": "MDA",
    "Republic of the Congo": "COG",
    "Democratic Republic of Timor-Leste": "TLS",
    "Bonaire, Sint Eustatius, and Saba": "BES",
}
country_code_map.update(manual_map)

# Process data
device_types = set(item["_id"]["device"] for item in data)
plot_data = {device: {"countries": [], "counts": [], "log_counts": [], "iso_codes": []} for device in device_types}

for item in data:
    country = item["_id"]["country"]
    device = item["_id"]["device"]
    count = item["count"]
    log_count = math.log2(count + 1)  # Log scale
    code = country_code_map.get(country)
    if code:
        plot_data[device]["countries"].append(country)
        plot_data[device]["counts"].append(count)
        plot_data[device]["log_counts"].append(log_count)
        plot_data[device]["iso_codes"].append(code)

# Create figure
fig = go.Figure()

# Add choropleth for the first device type
default_device = list(device_types)[0]
fig.add_choropleth(
    locations=plot_data[default_device]["iso_codes"],
    z=plot_data[default_device]["log_counts"],
    text=plot_data[default_device]["countries"],
    customdata=plot_data[default_device]["counts"],
    colorscale="Viridis",
    zmin=min(plot_data[default_device]["log_counts"] or [0]),
    zmax=max(plot_data[default_device]["log_counts"] or [1]),
    marker_line_color="white",
    marker_line_width=0.5,
    colorbar_title="Log(Count + 1)",
    hovertemplate="%{text}<br>Count: %{customdata}<extra></extra>",
)

# Update layout with dropdown
fig.update_layout(
    title=f"Record Count by Country (Device: {default_device})",
    template="plotly_white",
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type="equirectangular"
    ),
    height=600,
    updatemenus=[
        dict(
            buttons=[
                dict(
                    args=[{
                        "locations": [plot_data[device]["iso_codes"]],
                        "z": [plot_data[device]["log_counts"]],
                        "text": [plot_data[device]["countries"]],
                        "customdata": [plot_data[device]["counts"]],
                        "zmin": [min(plot_data[device]["log_counts"] or [0])],
                        "zmax": [max(plot_data[device]["log_counts"] or [1])],
                        "title.text": f"Record Count by Country (Device: {device}, Log Scale)"
                    }],
                    label=device,
                    method="update"
                ) for device in device_types
            ],
            direction="down",
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        )
    ]
)

# Save and show
fig.write_html("country_count_map_log.html")
fig.show()
```

(Again, flip to light mode if not in it)


{% include country_count_map_log.html %}


To take it a step further, we could look at conversion rates. They are clearly going to often be greater than 100%, in terms of clicks as a proportion of pageloads (I assume navigating directly to a product counts as a click but not a pageload), but that doesn't necessarily make them uninformative. We could also analyze the behavior of those with an account vs. 'non-users', perhaps creating an aggregated table for the non-users as well.


# What's Next?

My intention for the next MongoDB article, which may or may not be the next one in general, is to use Latent Dirichlet Allocation upon some Reddit data for topic modeling, transform it to supervised LDA (sLDA) to make predictions we can score based on accuracy, F1-score, etc., and compare it to methods like latent semantic indexing (LSI) and negative matrix-factorization (NMF).



# References

Mongo DB User Docs - General
- <a href="https://www.mongodb.com/docs/">https://www.mongodb.com/docs/</a>

Mongo DB User Docs - Aggregation Pipelines
- <a href="https://www.mongodb.com/docs/manual/core/aggregation-pipeline/">https://www.mongodb.com/docs/manual/core/aggregation-pipeline/</a>

MongoDB to SQL Cheatsheet
- <a href="https://www.mongodb.com/docs/manual/reference/sql-aggregation-comparison/">https://www.mongodb.com/docs/manual/reference/sql-aggregation-comparison/</a>

Plotly User Docs
- <a href="https://plotly.com/python/">https://plotly.com/python/</a>

And a recent subscription to Grok


