---
layout: post
title:  "MySQL via Shell and Python"
date:   2025-05-14 00:00:00 +0000
categories: SQL RDBMS NLP DimensionReduction
---

MySQL is one of the most widely used relational database management systems (RDBMS), due in part to its open-source community edition. To ensure a high degree of no-fuss replicability, the following was done in a Google Colab notebook. Commanding MySQL from within a Python environment offers a wide range of cool integrations.



# Outline
- Introduction
- Installing and Importing Libraries
- Importing a Sample Database
- Interacting with Text and CSV Files
- Inspecting the Data
- Magic Commands
- Creating a Table
- Constructing Field Lists
- Query Examples
- Integrating Machine Learning
- What's Next?



# Introduction

The ubiquitous SQL language has been the standard for database-querying since the 70's, though numerous versions exist. This article focuses on MySQL, a variant for which the community version is free. Similar alternatives include SQL Server Express, PostgreSQL, and MariaDB.

Similarly, Python has become one of the most widely adopted programming languages, for tasks which don't require low-level coding such as C++. This popularity has led to the development of many open-source libraries, from which powerful functions may be imported and called upon using few lines of code. The reasons IPython notebooks (.ipynb) have become widely used include: 
- Shareability, readability, and transparency
- Ability to render as a webpage
- Workflow documentation
- Ability to include markdown 
- Convenient debugging
- "Magic commands", which provide functionality such as allowing other languages

Magic commands are not the only way to integrate SQL into a Python environment. Many libraries and functions, some of which we'll explore, are designed to accept plain-text SQL queries. A perk of the magic commands which refer to other languages is that they should yield the kind of keyword-formatting you would expect of an editor for the language.

Working in a Google Colab Python notebook, which you can download <a href="https://github.com/pw598/Articles/blob/main/notebooks/MySQL_via_Shell_and_Python.ipynb">here</a>, we will start with shell commands, and progress to include Python functions which wrap SQL queries. Of course, hosting your business's database in Google Colab (or through a notebook alone) would be highly problematic, unless you somehow mitigate the fact that the data will disappear at the end of your session. However, for our purposes of demonstration and replicability, it will work nicely.

I must also mention that you should not include passwords in your code, even though I have, for simplicity. Instead, keep the sensitive commands stored in a secure .sql or text file to be read upon execution, or store the passwords securely in your operating system environment.

Since Google Colab runs on a Linux environment, the shell commands are in Bash. Don't be intimidated if the commands look unfamiliar, but do check out the following resources if looking for context.

- <a href="https://www.w3schools.com/bash/index.php">https://www.w3schools.com/bash/index.php</a>
- <a href="https://www.w3schools.com/sql/">https://www.w3schools.com/sql/</a>



# Installing and Importing Libraries

We'll start by installing <code>mysql-server</code>. 
- <code>!apt-get update</code> refreshes the list of available packages
- <code>-y</code> in the install command will automatically answer all prompts with yes during installation
- <code>/dev&#8203;/null 2>&1</code> is simply a command to suppress cell output. Remove it if you need to debug the installation.

```bash
!apt-get update > /dev/null 2>&1
!apt-get install -y mysql-server > /dev/null 2>&1
```

The below moves the user’s home directory with where MySQL’s data files are stored, which will help to avoid encountering errors in Google Colab.

```bash
!usermod -d /var/lib/mysql mysql
```

Next, we start the SQL service.

```bash
!service mysql start
```

<p></p>

```bash
# * Starting MySQL database server mysqld
#   ...done.
```

We can view the details of the service instance by entering the following.

```bash
# test for MySQL service
!service mysql status
```

<p></p>

```bash
#  * /usr/bin/mysqladmin  Ver 8.0.42-0ubuntu0.22.04.1 for Linux on x86_64 ((Ubuntu))
# Copyright (c) 2000, 2025, Oracle and/or its affiliates.

# Oracle is a registered trademark of Oracle Corporation and/or its
# affiliates. Other names may be trademarks of their respective
# owners.

# Server version    8.0.42-0ubuntu0.22.04.1
# Protocol version  10
# Connection    Localhost via UNIX socket
# UNIX socket   /var/run/mysqld/mysqld.sock
# Uptime:     18 min 25 sec

# Threads: 2  Questions: 1048  Slow queries: 0  Opens: 454  
# Flush tables: 3  Open tables: 356  Queries per second avg: 0.948
```

Creating a <code>.my.cnf</code> file prevents us from having to specify our username and password each time we enter shell commands. The user will be set to "root" and the password to "pw". Again, referencing them explicitly in the code is not secure; it is only done here for simplicity.
- <code>-f</code> stands for "force", meaning do not prompt for confirmation, and ignore nonexistent files
- <code>-e</code> stands for "execute".

```bash
# Create .my.cnf for password-based authentication
!rm -f ~/.my.cnf /root/.my.cnf                              # clear if existing
!echo -e "[client]\nuser=root\npassword=pw" > ~/.my.cnf     # print text to CLI
!chmod 600 ~/.my.cnf                                        # grants read/write permissions to file owner
```

We can use the following to see if the MySQL connection has been established. 
- <code>-N</code> suppresses the column header

```bash
# Check if sudo mysql can connect
!mysql -N -e "SELECT 1;" || echo "Failed to connect"
```

<p></p>

```bash
# +---+
# | 1 |
# +---+
```

The below specifies the user and host from which we can connect.

- <code>IDENTIFIED WITH mysql_native_password</code> forces compatibility with older systems, which is seemingly necessary in Google Colab.

- <code>FLUSH PRIVILEGES</code> tells MySQL to reload the privilege tables, ensuring any changes made take effect without a restart.

```bash
!sudo mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'pw'; FLUSH PRIVILEGES;"
```

If we use <code>SHOW DATABASES</code>, we see the system-related databases which exist regardless of whether data has been imported.

```bash
!mysql -e "SHOW DATABASES;"
```

<p></p>

```bash
# +--------------------+
# | Database           |
# +--------------------+
# | information_schema |
# | mysql              |
# | performance_schema |
# | sys                |
# +--------------------+
```


# Importing a Sample Database

The data we'll work with comes from a sample SQL database called 'classicmodels'. It simulates the data of a distributor of scale model cars, motorcycles, ships, planes, etc. It is a .sql file, and can be obtained <a href="https://www.mysqltutorial.org/getting-started-with-mysql/mysql-sample-database/">here</a>. Change the filepath if necessary, so that your notebook or environment can access it.

```bash
!mysql < mysqlsampledatabase.sql
```

Now, when we run <code>SHOW DATABASES</code>, we see the 'classicmodels' database is present.

```bash
!mysql -e "SHOW DATABASES;"
```

<p></p>

```bash
# +--------------------+
# | Database           |
# +--------------------+
# | classicmodels      |
# | information_schema |
# | mysql              |
# | performance_schema |
# | sys                |
# | text_db            |
# +--------------------+
```



# Interacting with Text and CSV Files

To demonstrate how we would import the database from a .txt file, I will first export classicmodels to .txt.

- <code>sed</code> stands for Stream Editor, a Unix-based tool for searching, finding/replacing, inserting, or deleting text in files or streams.

- <code>-i</code> tells <code>sed</code> to edit the file directly, rather than just printing the result to screen.

```bash
!mysqldump -u root -ppw classicmodels > text_db.txt
!sed -i 's/`classicmodels`/`text_db`/g' text_db.txt
```

Feel free to open the .txt file and inspect the format. 

Next, we will use the appropriate commands to import that data into a database called text_db. <code>-ppw</code> is <code>-p</code> for password, followed by the password <code>pw</code>. Unlike with <code>-u root</code> for specifying our username, no space character is expected between the <code>-p</code> and the password.

```bash
!mysql -u root -ppw -e "CREATE DATABASE text_db;"
!mysql -u root -ppw text_db < text_db.txt
```

And now if we run <code>SHOW DATABASES</code>, we see text_db included.

```bash
!mysql -e "SHOW DATABASES;"
```

<p></p>

```bash
# +--------------------+
# | Database           |
# +--------------------+
# | classicmodels      |
# | information_schema |
# | mysql              |
# | performance_schema |
# | sys                |
# | text_db            |
# +--------------------+
```

What about .csv? Well, we cannot readily export an entire database to a .csv file, but we can work with individual tables, and loop through them if we want to address them all. For this, we'll use <code>pymysql</code>, <code>sqlalchemy</code>, and <code>pandas</code>.

```bash
!pip install pymysql > /dev/null 2>&1
```

<p></p>

```python
from sqlalchemy import create_engine, text
import pandas as pd
import os

# connection setup
engine = create_engine(f"mysql+pymysql://root:pw@localhost/classicmodels")

# check if 'csv_db' folder exists, create if not
output_dir = 'csv_db'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# get list of tables
with engine.connect() as conn:
    result = conn.execute(text("SHOW TABLES;"))
    tables = [row[0] for row in result]

# loop through tables and export each to the 'csv_db' directory
for table in tables:
    df = pd.read_sql(f"SELECT * FROM `{table}`", engine)
    file_path = os.path.join(output_dir, f"{table}.csv")
    df.to_csv(file_path, index=False)

print(f"Export complete. All CSV files are in the '{output_dir}' directory.")
```

<p></p>

```python
# Export complete. All CSV files are in the 'csv_db' directory.
```



# Inspecting the Data

Next, we'll use <code>SHOW TABLES</code> to list the tables in the database.

```python
database = 'classicmodels'
```

<p></p>

```bash
!mysql -e "USE {database}; SHOW TABLES;"
```

<p></p>

```bash
# +-------------------------+
# | Tables_in_classicmodels |
# +-------------------------+
# | customers               |
# | employees               |
# | offices                 |
# | orderdetails            |
# | orders                  |
# | payment_view_with_dates |
# | payments                |
# | productlines            |
# | products                |
# | simple_table            |
# +-------------------------+
```

We can drill into a particular table, such as <code>customers</code>, and use <code>DESCRIBE</code> to view the attributes of each field.

```bash
!mysql -e "USE {database}; DESCRIBE customers;"
```

<p></p>

```bash
# +------------------------+---------------+------+-----+---------+-------+
# | Field                  | Type          | Null | Key | Default | Extra |
# +------------------------+---------------+------+-----+---------+-------+
# | customerNumber         | int           | NO   | PRI | NULL    |       |
# | customerName           | varchar(50)   | NO   |     | NULL    |       |
# | contactLastName        | varchar(50)   | NO   |     | NULL    |       |
# | contactFirstName       | varchar(50)   | NO   |     | NULL    |       |
# | phone                  | varchar(50)   | NO   |     | NULL    |       |
# | addressLine1           | varchar(50)   | NO   |     | NULL    |       |
# | addressLine2           | varchar(50)   | YES  |     | NULL    |       |
# | city                   | varchar(50)   | NO   |     | NULL    |       |
# | state                  | varchar(50)   | YES  |     | NULL    |       |
# | postalCode             | varchar(15)   | YES  |     | NULL    |       |
# | country                | varchar(50)   | NO   |     | NULL    |       |
# | salesRepEmployeeNumber | int           | YES  | MUL | NULL    |       |
# | creditLimit            | decimal(10,2) | YES  |     | NULL    |       |
# +------------------------+---------------+------+-----+---------+-------+
```

If we want to view indexes, we can use <code>SHOW INDEXES FROM table</code>. We must apply a <code>USE</code> statement to select the appropriate database.

- If you are unfamiliar with indexes, they are data structures that improve the speed of retrieval operations. By mapping key values to row locations in the table, the database does not have to scan every row to find matches.

```bash
!mysql -e "USE {database}; SHOW INDEXES FROM customers;"
```

<p></p>

```bash
# +-----------+------------+------------------------+--------------+------------------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
# | Table     | Non_unique | Key_name               | Seq_in_index | Column_name            | Collation | Cardinality | Sub_part | Packed | Null | Index_type | Comment | Index_comment | Visible | Expression |
# +-----------+------------+------------------------+--------------+------------------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
# | customers |          0 | PRIMARY                |            1 | customerNumber         | A         |         122 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
# | customers |          1 | salesRepEmployeeNumber |            1 | salesRepEmployeeNumber | A         |          16 |     NULL |   NULL | YES  | BTREE      |         |               | YES     | NULL       |
# +-----------+------------+------------------------+--------------+------------------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
```

Below references <code>information_schema.tables</code> for the database in question, to print out a count of rows for each table.

```bash
!mysql -e "SELECT table_name, table_rows FROM information_schema.tables WHERE table_schema = '{database}'"
```

<p></p>

```bash
# +-------------------------+------------+
# | TABLE_NAME              | TABLE_ROWS |
# +-------------------------+------------+
# | customers               |        122 |
# | employees               |         23 |
# | offices                 |          7 |
# | orderdetails            |       2996 |
# | orders                  |        326 |
# | payment_view_with_dates |       NULL |
# | payments                |        273 |
# | productlines            |          7 |
# | products                |        110 |
# | simple_table            |          2 |
# +-------------------------+------------+
```

To get the number of columns for each table, we can use the following:

```bash
!mysql -e "SELECT table_name, COUNT(*) AS column_count FROM information_schema.columns WHERE table_schema = '{database}' GROUP BY table_name;"
```

<p></p>

```bash
# +-------------------------+--------------+
# | TABLE_NAME              | column_count |
# +-------------------------+--------------+
# | customers               |           13 |
# | employees               |            8 |
# | offices                 |            9 |
# | orderdetails            |            5 |
# | orders                  |            7 |
# | payment_view_with_dates |            8 |
# | payments                |            4 |
# | productlines            |            4 |
# | products                |            9 |
# | simple_table            |            2 |
# +-------------------------+--------------+
```



# Magic Commands

We'll take a break from the shell commands, in favor of Python magic commands. 

- This requires that we install <code>mysql-connector-python</code>, run <code>%load_ext sql</code>, and establish a connection to a particular database. 

- We'll then proceed to use <code>%%sql</code> at the top of any cells that contain only single-statement SQL commands. A slight exception is that we can use <code>%%sql << my_variable</code> on the top line to store the SQL output into a Python variable.

- To mix SQL with Python, or use multiple SQL commands in one cell, we'll precede the SQL statements with <code>%sql</code> (having only one percentage symbol).

```bash
!pip install mysql-connector-python
%load_ext sql
```

<p></p>

```python
%sql mysql+mysqlconnector://root:pw@localhost/classicmodels
```

The first configuration statement below offers backward compatibility with previous versions, which is seemingly necessary for the Google Colab notebook. <code>SqlMagic.autopandas = True</code> is a setting that will convert our SQL outputs to <code>pandas</code> dataframes, for enhanced formatting and easy manipulation.

```python
%config SqlMagic.style = '_DEPRECATED_DEFAULT'
%config SqlMagic.autopandas = True
```

Using the <code>%%sql</code> magic command looks like the below, where we select the list of table names, column names, and column types from the <code>information_schema</code> table, limiting results to the first 5 rows for brevity.

```sql
%%sql

SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = 'classicmodels'
ORDER BY TABLE_NAME, ORDINAL_POSITION
LIMIT 5;
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/sq1-1.png" style="height: 200px; width:auto;">

The below shows us the same, but rather than limiting the SQL query to return only the top 5 rows, we save the entire SQL output into a Python variable, and use <code>result.head()</code> to print only the top 5 rows. The start of the SQL command must appear on the same line as the magic command, although it's true that we could continue onto the next line by inserting a backslash, as done below. Note that any spaces after the backslash will cause an error.

```python
result = %sql SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE \
FROM information_schema.COLUMNS \
WHERE TABLE_SCHEMA = 'classicmodels' \  
ORDER BY TABLE_NAME, ORDINAL_POSITION;

# inspect top rows with pandas
result.head()
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/sq1-2.png" style="height: 200px; width:auto;">

The alternative syntax which allows us to use <code>%%sql</code>, but also capture the result in a Python variable, is as follows. Note that we must put off the <code>result.head()</code> command until the next cell, as it's expected that all lines below the first are strictly SQL.

```sql
%%sql result <<
SELECT
TABLE_NAME, COLUMN_NAME, COLUMN_TYPE
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = 'classicmodels'
ORDER BY TABLE_NAME, ORDINAL_POSITION;
```

<p></p>

```python
# inspect top rows with pandas
result.head()
```



# Creating a Table

Below, we'll create a new table using <code>mysql-connector</code>, and then prove that it shows up regardless of the method of querying.

```sql
%%sql

CREATE TABLE IF NOT EXISTS simple_table (  -- will not attempt creation if table exists
    first_name VARCHAR(50),
    last_name VARCHAR(50)
);

INSERT IGNORE INTO simple_table (first_name, last_name) -- will ignore if the records exist
VALUES ("Alice", "Anderson"),
       ("Bob", "Browning");
```

<p></p>

```python
#  * mysql+mysqlconnector://root:***@localhost/classicmodels
# 0 rows affected.
# 2 rows affected.
```

Selecting with a magic command:

```sql
%%sql
SELECT * FROM simple_table;
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/sq1-3.png" style="height: 250px; width:auto;">

Selecting with a shell command:

```bash
!mysql -e "USE classicmodels; SELECT * FROM simple_table;"
```

<p></p>

```bash
# +------------+-----------+
# | first_name | last_name |
# +------------+-----------+
# | Alice      | Anderson  |
# | Bob        | Browning  |
# | Alice      | Anderson  |
# | Bob        | Browning  |
# | Alice      | Anderson  |
# | Bob        | Browning  |
# +------------+-----------+
```



# Constructing Field Lists

In a notebook, for the sake of less scrolling, it might be nice to list out field names horizontally, rather than column-wise. To do it with a shell command, we could use the following:

- <code>|</code> takes the output of the previous command and passes it to the next one

- <code>tail -n +2</code> skips the header and prints from the second row downward

- <code>paste</code> joins lines together

- <code>-s</code> merges all lines into a single line

- <code>-d</code> means to use a comma delimiter

- <code>-</code> tells <code>paste</code> to read from the piped output. 


```bash
!mysql -e "USE classicmodels; SHOW TABLES;" | tail -n +2 | paste -sd, -
```

<p></p>

```bash
# customers,employees,offices,orderdetails,orders,payment_view_with_dates,payments,productlines,products,simple_table
```
Another option would have been to extract the data into a Python variable, and then perform a transpose operation.

If we want to insert spaces after each comma, we can do it with the following. 

- <code>'s/,/, /g'</code> is a substitution rule using regular expressions to replace each comma with a comma and space.

```bash
!mysql -e "USE classicmodels; SHOW TABLES;" | tail -n +2 | paste -sd, - | sed 's/,/, /g'
```

<p></p>

```bash
# customers, employees, offices, orderdetails, orders, payment_view_with_dates, payments, productlines, products, simple_table
```

If we want to limit the number of items per line, let's say to 5, then we can use the following. To keep it brief, <code>awk</code> is a command used to control how the output is printed, and the rest provides it with instructions.

```bash
!mysql -e "USE classicmodels; SHOW TABLES;" | tail -n +2 | awk 'ORS=(NR%5==0) ? "\n" : ", "'
```

<p></p>

```bash
# customers, employees, offices, orderdetails, orders
# payment_view_with_dates, payments, productlines, products, simple_table
```

To rely on Python rather than shell scripts, we can use the following.

```python
import mysql.connector
```

<p></p>

```python
def list_table_columns(host, user, password, database, table):
  conn = mysql.connector.connect(host=host,
                                  user=user,
                                  password=password,
                                  database=database)

  # Create a cursor object to interact with the database
  cursor = conn.cursor()

  # Execute the DESCRIBE query
  str = "DESCRIBE " + table + ";"
  cursor.execute(str)

  # Fetch all results
  columns = cursor.fetchall()

  # Extract column names into a list
  column_names = [column[0] for column in columns]

  # Print the column names with a new line after every 5 items
  for i in range(0, len(column_names), 5):
      # Join the next 5 column names with commas and print
      print(', '.join(column_names[i:i+5]))

  # Close the cursor and connection
  cursor.close()
  conn.close()
```

I'll take this opportunity to list out the fields contained in the tables we will work with shortly.

```python
host = "localhost"
user = "root"
password = "pw"
database = "classicmodels"
table = 'offices'
list_table_columns(host, user, password, database, table)
```

<p></p>

```python
# officeCode, city, phone, addressLine1, addressLine2
# state, country, postalCode, territory
```

<p></p>

```python
table = 'customers'
list_table_columns(host, user, password, database, table)
```

<p></p>

```python
# customerNumber, customerName, contactLastName, contactFirstName, phone
# addressLine1, addressLine2, city, state, postalCode
# country, salesRepEmployeeNumber, creditLimit
```

<p></p>

```python
table = 'employees'
list_table_columns(host, user, password, database, table)
```

<p></p>

```python
# employeeNumber, lastName, firstName, extension, email
# officeCode, reportsTo, jobTitle
```

<p></p>

```python
table = 'payments'
list_table_columns(host, user, password, database, table)
```

<p></p>

```python
# customerNumber, checkNumber, paymentDate, amount
```



# Query Examples

We've seen several simple <code>SELECT</code>/<code>FROM</code>/<code>WHERE</code> queries above, so I won't belabour the basics. Below demonstrates various forms of aggregation, using the <code>amount</code> field of the payments table. We need to provide an alias for the aggregated fields, which is the name by which it will appear in the output (even if we are using the same name as the field).

```sql
%%sql

SELECT
    COUNT(amount) AS count,
    AVG(amount) AS average,
    MIN(amount) AS minimum,
    MAX(amount) AS maximum,
    SUM(amount) AS total
FROM payments;
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/sq1-4.png" style="height: 75px; width:auto;">

To break an aggregation out into multiple rows, such as by customer, we need to incorporate a <code>GROUP BY</code> statement on the non-aggregated fields, such as below.

```sql
%%sql

SELECT
    customerNumber,
    SUM(amount) as amount
 FROM payments
 GROUP BY customerNumber;
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/sq1-5.png" style="height: 400px; width:auto;">

To bring in customer name, we can do a join with the customers table. An <code>INNER JOIN</code>, as done below, assumes that we are only interested in records for which the field we are joining on has a match in both tables. An <code>ORDER BY</code> statement is used at the bottom to sort by sum of payments in descending fashion.

```sql
%%sql

SELECT
    customers.customerNumber,
    customers.customerName,
    SUM(payments.amount) as amount
FROM customers
INNER JOIN payments ON customers.customerNumber = payments.customerNumber
GROUP BY customers.customerNumber, customers.customerName
ORDER BY amount DESC;
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/sq1-13.png" style="height: 200px; width:auto;">

Let's get a little fancier. Below, we will join more than two tables together, getting the number of employees, number of customers, and sum of payments per office for 2003.

- The <code>DISTINCT</code> keyword will allow us to not just count the number of rows in the database, as <code>COUNT</code> would, but rather, count the number of unique values for the specified field. In this case, the level of detail will be by office.

- Instead of an <code>INNER JOIN</code>, we will use a <code>LEFT JOIN</code>, which means we keep all records from the 'left' table (i.e., the first one mentioned, offices), and bring in matches from the other tables, but associate null values for the right table with entries on the left which cannot be matched to.

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/joins.png" style="height: 275px; width:auto;">

- The <code>CONVERT(SUM(payments.amount), SIGNED)</code> statement below is performing the MySQL equivalent of <code>CAST(SUM(payments.amount) AS INT)</code> in SQL Server. It is rounding the result to the nearest integer and also setting the data type.

<p><i>source: https://www.w3schools.com/sql/sql_join.asp</i></p>


```sql
%%sql

SELECT
    offices.officeCode,
    COUNT(DISTINCT customers.customerNumber) AS customer_count,
    COUNT(DISTINCT employees.employeeNumber) AS employee_count,
    COUNT(payments.amount) AS payment_count,
    CONVERT(SUM(payments.amount), SIGNED) AS payment_total
FROM offices
LEFT JOIN employees ON offices.officeCode = employees.officeCode
LEFT JOIN customers ON employees.employeeNumber = customers.salesRepEmployeeNumber
LEFT JOIN payments ON customers.customerNumber = payments.customerNumber
                   AND YEAR(payments.paymentDate) = 2003
GROUP BY offices.officeCode
ORDER BY payment_total DESC;
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/sq1-6.png" style="height: 250px; width:auto;">

Below, we'll use date functions to create columns expressing the year, month number, week number, and day of week for each row. To avoid altering the original table, we'll create a <code>VIEW</code> of that table. Tables are persistent, and their data saved to disk, whereas a view is a virtual table generated by a query, and stored as a query.

```sql
%sql DROP VIEW IF EXISTS payment_view_with_dates;

%sql CREATE VIEW payment_view_with_dates AS \
SELECT \
    p.*, \
    YEAR(p.paymentDate) AS payment_year, \
    MONTH(p.paymentDate) AS payment_month, \
    WEEK(p.paymentDate, 0) AS payment_week, \
    DAYOFWEEK(p.paymentDate) AS payment_dayofweek \
FROM payments p;

result = %sql SELECT * FROM payment_view_with_dates;
result.head()
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/sq1-7.png" style="height: 200px; width:auto;">


Below, we'll create a crosstab of amount per customer (as rows) by year (as columns). With SQL Server, we could use the <code>PIVOT</code> keyword, but with MySQL, we'll have to setlle for <code>CASE</code> statement logic.

```python
import mysql.connector
import pandas as pd

# Connect to your MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="pw",
    database="classicmodels"
)

cursor = conn.cursor()

# Step 1: Manually define pivot columns for known years
manual_columns = """
    SUM(CASE WHEN YEAR(paymentDate) = 2003 THEN amount ELSE 0 END) AS `2003`,
    SUM(CASE WHEN YEAR(paymentDate) = 2004 THEN amount ELSE 0 END) AS `2004`,
    SUM(CASE WHEN YEAR(paymentDate) = 2005 THEN amount ELSE 0 END) AS `2005`
"""

# Step 2: Build and execute the final SQL query
sql = f"""
    SELECT customerNumber AS customer, {manual_columns}
    FROM payments
    GROUP BY customer
    ORDER BY customer;
"""
cursor.execute(sql)

# Step 3: Fetch column names and results
columns = [desc[0] for desc in cursor.description]
rows = cursor.fetchall()

# Step 4: Create a pandas DataFrame
df = pd.DataFrame(rows, columns=columns)

# Clean up
cursor.close()
conn.close()

# Display the result
df.head()
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/sq1-8.png" style="height: 200px; width:auto;">


To get a little fancier, and make the code dynamic toward any number of years, we can do the following.

```python
import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="pw",
    database="classicmodels"
)

cursor = conn.cursor()

# increase group_concat limit (optional but helpful for many years)
# cursor.execute("SET SESSION group_concat_max_len = 100000;")

# build dynamic column definitions
cursor.execute("""
    SELECT GROUP_CONCAT(
        DISTINCT CONCAT(
            'SUM(CASE WHEN YEAR(paymentDate) = ', YEAR(paymentDate),
            ' THEN amount ELSE 0 END) AS `', YEAR(paymentDate), '`'
        )
        ORDER BY YEAR(paymentDate)
    )
    FROM payments;
""")
pivot_columns = cursor.fetchone()[0]

# build and execute SQL
sql = f"""
    SELECT customerNumber AS customer, {pivot_columns}
    FROM payments
    GROUP BY customer
    ORDER BY customer;
"""
cursor.execute(sql)

# fetch column names and results
columns = [desc[0] for desc in cursor.description]
rows = cursor.fetchall()

# create a pandas DataFrame
df = pd.DataFrame(rows, columns=columns)

cursor.close()
conn.close()

df.head()
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/sq1-9.png" style="height: 200px; width:auto;">


<code>SELECT GROUP_CONCAT()</code> combines multiple strings into one, separated by commas. <code>DISTINCT CONCAT()</code>, and the lines within that function, build a string for each year that looks like:

<code>SUM(CASE WHEN YEAR(paymentDate) = 2004 THEN amount ELSE 0 END) AS `2004`</code>

We also have the option of doing a very simple SQL query, and letting Python do the pivoting, as below.

```python
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="pw",
    database="classicmodels"
)

query = """
    SELECT customerNumber, payment_year, amount
    FROM payment_view_with_dates;
"""
df_raw = pd.read_sql(query, conn)

# rename columns for readability
df_raw.rename(columns={'customerNumber': 'customer'}, inplace=True)

# create pivot using pandas
df_pivot = pd.pivot_table(
    df_raw,
    index='customer',
    columns='payment_year',
    values='amount',
    aggfunc='sum',
    fill_value=0
).reset_index()

# Remove index name from columns
df_pivot.columns.name = None  

# close connection
conn.close()

df_pivot.head()
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/sq1-10.png" style="height: 200px; width:auto;">


Below, we'll break out the sum of payments at the office, customer, and employee level. Notice that the tables have been aliased, so that they can be referenced with abbreviations to save us some typing. We use <code>INNER JOIN</code>, thereby assuming we are not interested in the data for customers and employees to which no sales are associated.

```sql
%%sql

SELECT
    o.officeCode AS office,
    e.employeeNumber AS salesRep,
    c.customerNumber,
    c.customerName,
    SUM(p.amount) AS totalPayments
FROM offices o
INNER JOIN employees e ON o.officeCode = e.officeCode
INNER JOIN customers c ON e.employeeNumber = c.salesRepEmployeeNumber
INNER JOIN payments p ON c.customerNumber = p.customerNumber
GROUP BY o.officeCode, e.employeeNumber, c.customerNumber, c.customerName
ORDER BY o.officeCode, e.employeeNumber, c.customerNumber;
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/sq1-11.png" style="height: 400px; width:auto;">

The final SQL-only task, below, will elaborate on the above. The challenge is to:

- Include payment totals for office and employee, despite the data having a customer level of detail.
- Include the percentage of office and employee totals accounted for by each row.
- Rank the row-level amounts for office and employee, descending.

The key to this is a window function, combined with the <code>OVER</code> and <code>PARTITION</code> keywords. For example,

<code>ROUND(SUM(amount) OVER (PARTITION BY officeCode), 0) AS 'total payments for office'</code>

calculates the total amount paid by all customers in the same office (then applies a <code>ROUND</code> operation). 

<code>RANK() OVER (PARTITION BY officeCode ORDER BY SUM(amount) DESC) AS 'rank in office'</code>

assigns a rank to each customer for the office in question.


```sql
%%sql

SELECT
    officeCode,
    customerNumber,
    employeeNumber,
    customerName,
    ROUND(SUM(amount), 0) AS amount,
    ROUND(SUM(amount) OVER (PARTITION BY officeCode), 0) AS 'total payments for office',
    RANK() OVER (PARTITION BY officeCode ORDER BY SUM(amount) DESC) AS 'rank in office',
    ROUND(SUM(amount) OVER (PARTITION BY employeeNumber), 0) AS 'total payments for employee',
    RANK() OVER (PARTITION BY employeeNumber ORDER BY SUM(amount) DESC) AS 'rank for employee',
    ROUND(SUM(amount) OVER (PARTITION BY customerNumber), 0) AS 'total payments for customer',
    ROUND((SUM(amount) / SUM(amount) OVER (PARTITION BY officeCode)) * 100, 2) AS 'percent of office total',
    ROUND((SUM(amount) / SUM(amount) OVER (PARTITION BY employeeNumber)) * 100, 2) AS 'percent of employee total'
FROM (
    SELECT
        o.officeCode,
        c.customerNumber,
        e.employeeNumber,
        c.customerName,
        SUM(p.amount) AS amount
    FROM offices o
    INNER JOIN employees e ON o.officeCode = e.officeCode
    INNER JOIN customers c ON e.employeeNumber = c.salesRepEmployeeNumber
    INNER JOIN payments p ON c.customerNumber = p.customerNumber
    GROUP BY o.officeCode, e.employeeNumber, c.customerNumber, c.customerName
) AS aggregated_data

GROUP BY officeCode, employeeNumber, customerNumber, customerName
ORDER BY officeCode, employeeNumber, customerNumber;
```

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/sq1-12.png" style="height: 275px; width:auto;">



# Integrating Machine Learning

A nice aspect of working with SQL in a Python environment is that machine learning libraries can be immediately integrated. We can import high-level functions for advanced data science with only a few lines of code, and wrap SQL extractions in Python loops if the data are too large to work with all at once.

SQL offers the ability to create functions as well. For example, we could create a 'fuzzy lookup' function using Levenshtein distance, or a function to calculate the number of working days between dates. However, SQL by itself does not permit data visualization, and the ability to extend its functionality to machine learning capabilities is minimal. The below will integrate Python with an SQL query in order to cluster product descriptions based on semantic similarity, and visualize the results in an interactive 3D chart.

The challenge is to:

1. From the <code>products</code> table, pull <code>productCode</code> and <code>productName</code>, and from the prodlines table, pull <code>textDescription</code>. Concatenate <code>productName</code> with <code>textDescription.</code>

2. Convert the concatenated text to word embeddings - high-dimensional vectors of real numbers in continuous space, generated by a neural network, and saved to an importable library. The more similar the description text, the closer the direction of the vectors (and there are other neat capabilities, like analogy calculations).

3. Use a dimensionality reduction technique called UMAP (Uniform Manifold Approximation Projection) to project the high-dimensional data onto a lower-dimensional space that we can visualize, with as minimal a loss of information as possible.

4. Visualize in a 3D interactive plot using Plotly.


```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# get data
df = %sql SELECT \
    products.productCode, \
    prodlines.productLine, \
    products.productName, \
    CONCAT(products.productName, ' ', prodlines.productLine) as line_name_concat \
FROM productlines prodlines \
INNER JOIN products \
ON prodlines.productLine = products.productLine

data = {
    'product_id': df['productCode'],
    'prod_desc': df['productName'],
    'description': df['line_name_concat']}

# Create DataFrame (assumes 'data' is already defined with 'productCode' and 'productName')
df2 = pd.DataFrame(data)

# Load pre-trained BERT model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(df2['description'].tolist())
df2['embedding'] = list(embeddings)

# K-Means clustering
n_clusters = min(6, len(df2))
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)
df2['cluster'] = cluster_labels

# UMAP dimensionality reduction
umap_reducer = umap.UMAP(n_components=3, random_state=42)
embeddings_3d = umap_reducer.fit_transform(embeddings)

# Plotly 3D scatter with hover labels
palette = sns.color_palette("husl", n_clusters)
palette_hex = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in palette]

fig = go.Figure()

for cluster_id in range(n_clusters):
    mask = df2['cluster'] == cluster_id
    fig.add_trace(
        go.Scatter3d(
            x=embeddings_3d[mask, 0],
            y=embeddings_3d[mask, 1],
            z=embeddings_3d[mask, 2],
            mode='markers',
            marker=dict(size=8, color=palette_hex[cluster_id], opacity=0.6),
            name=f'Cluster {cluster_id}',
            text=df2.loc[mask, 'prod_desc'],  # Hover text
            hoverinfo='text'
        )
    )

fig.update_layout(
    title='3D UMAP Projection of Product Embeddings with K-Means Clusters',
    scene=dict(
        xaxis_title='UMAP Component 1',
        yaxis_title='UMAP Component 2',
        zaxis_title='UMAP Component 3',
        xaxis=dict(showspikes=False),
        yaxis=dict(showspikes=False),
        zaxis=dict(showspikes=False)
    ),
    showlegend=True,
    width=800,
    height=600
)


fig.write_html('product_similarity_umap_kmeans_clusters_3d.html')
fig.show()
```


<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/plotly_cht.png" style="height: 400px; width:auto;">


To zoom in, rotate, and hover over points for labels, download the chart by <a href="https://github.com/pw598/pw598.github.io/blob/main/_posts/images/plotly_chart.html" download="plotly_chart.html">right-clicking here</a> and selecting "Save Link As". Or, <a href="https://github.com/pw598/Articles/blob/main/notebooks/MySQL_via_Shell_and_Python.ipynb">download the .ipynb notebook</a>.



# What's Next?

- Likely next will be an article about JSON (JavaScript Object Notation) based queries with MongoDB. Future RDBMS posts may include SQL functions, and/or a sandbox application using Streamlit.





