---
layout: post
title:  "MongoDB via Shell and PyMongo"
date:   2025-05-23 00:00:00 +0000
categories: MongoDB Bash Python
---

Intro text...

Image?


# Outline
- ...
- ...


# Introduction

## The Advantages of Unstructured Data


## MongoDB



# Installation

- Linux
- Windows
- Conda (all systems)



# Import Common Libraries

```python
from pymongo import MongoClient
import subprocess 
import json
import pprint as pp
```


# Connect and List Databases

```python
HOST = "localhost"
PORT = "27017"
```

<p></p>


```python
conn_str = f"mongodb://{HOST}:{PORT}/"

print("Connection string:")
print(conn_str)

print("\n Databases")
client = MongoClient(conn_str)
print(client.list_database_names())
```

<p></p>

```python
# Connection string:
# mongodb://localhost:27017/

#  Databases
# ['admin', 'clickstream', 'config', 'local']
```


### To Drop <code>clickstream</code> if DB Currently Exists (Optional)

```python
DBNAME = "clickstream"
```

### Using Bash

```bash
!mongosh --quiet --eval "db.getSiblingDB('{DBNAME}').dropDatabase()"
!mongosh --quiet --eval "db.adminCommand('listDatabases').databases.forEach(db => print(db.name))"
```

<p></p>

```bash
# { ok: 1, dropped: 'clickstream' }
# admin
# config
# local
```



### Using Python 

```python
client.drop_database("clickstream")
print(client.list_database_names())
```

<p></p>

```python
# ['admin', 'config', 'local']
```



# Import Data

### Set Variables

```python
IMPORT_FILE_FOLDER = "C:\\Users\\patwh\\Downloads"
BSON_FILE_NAME = "clicks"
JSON_FILE_NAME = "clicks.metadata"

bson_file = f"{IMPORT_FILE_FOLDER}/{BSON_FILE_NAME}.bson" 
json_file = f"{IMPORT_FILE_FOLDER}/{JSON_FILE_NAME}.json" 

COLLECTION_BSON = BSON_FILE_NAME
COLLECTION_JSON = JSON_FILE_NAME
```


### Using Bash (Optional)

```bash
!mongorestore --host localhost:{PORT} --db {DBNAME} --collection {COLLECTION_BSON} --drop "{bson_file}"
!mongoimport --host localhost:{PORT} --db {DBNAME} --collection {CELLECTION_JSON} --drop --type json "{json_file}"
```


### Using Python

```python
db = client[DBNAME]

# Import bson file
subprocess.run(f'mongorestore --host {HOST}:{PORT} --db {DBNAME} --collection {COLLECTION_BSON} --drop "{bson_file}"')

# Import json file
with open(json_file) as f:
    data = json.load(f)
    db[COLLECTION_JSON].insert_many(data) if isinstance(data, list) else db[COLLECTION_JSON].insert_one(data)

# Print document counts
print(f"Records in {COLLECTION_BSON}: {db[COLLECTION_BSON].count_documents({})}")
print(f"Records in {COLLECTION_JSON}: {db[COLLECTION_JSON].count_documents({})}")
```

<p></p>

```python
# Records in clicks: 6100000
# Records in clicks.metadata: 1
```



# List Databases


### Using Bash

Method 1:

```bash
!mongosh --host {HOST} --port {PORT} --quiet --eval "show databases"
```

<p></p>

```bash 
# admin        132.00 KiB
# clickstream  428.59 MiB
# config       108.00 KiB
# local         96.00 KiB
```

Method 2:

```bash
!mongosh --host {HOST} --port {PORT} --quiet --eval "db.getMongo().getDBs().databases.map(db => db.name)"
```

<p></p>

```bash
[ 'admin', 'config', 'local' ]
```


### Using Python

```python
print(client.list_database_names())
```

<p></p>

```python
['admin', 'config', 'local']
```



# Select Clickstream Database












