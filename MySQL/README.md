<div align="center">

# MySQL教程

</div>

## 1. MySQL简介
MySQL是一个开源的关系型数据库管理系统（RDBMS），由瑞典公司MySQL AB开发，现由Oracle Corporation维护。它使用结构化查询语言（SQL）进行数据库管理和操作，广泛应用于各种应用程序和网站中。
## 2. MySQL安装
### 2.1 Docker安装
使用Docker安装MySQL非常方便，可以通过以下`docker-compose.yml`文件来快速部署MySQL服务：
```yaml
version: "3.8" # 指定Docker Compose文件的版本

services: # 定义服务
  mysql: # 定义MySQL服务
    image: mysql:8.0 # 使用MySQL 8.0版本
    container_name: mysql # 设置容器名称
    restart: always # 设置容器自动重启
    ports:          # 映射MySQL默认端口
      - "3306:3306"
    environment:
      MYSQL_ROOT_PASSWORD: 123456 # 设置MySQL root用户的密码
      MYSQL_DATABASE: testdb # 创建一个名为testdb的数据库
      MYSQL_USER: testuser # 创建一个名为testuser的用户
      MYSQL_PASSWORD: testpass # 设置testuser用户的密码
    volumes:
      - ./mysql/data:/var/lib/mysql # 持久化MySQL数据
      - ./mysql/conf:/etc/mysql/conf.d # 持久化MySQL配置文件
      - ./mysql/log:/logs # 持久化MySQL日志文件
```

启动：
```bash
docker-compose up -d
```
停止：
```bash
docker-compose down
```
重新编译启动：
```bash
docker-compose up -d --build
```
### 2.2 本地安装
在本地安装MySQL可以通过以下步骤进行：
1. 访问MySQL官方网站下载适合你操作系统的安装包。
2. 运行安装程序，按照提示进行安装。
3. 在安装过程中设置root用户的密码，并选择安装类型（通常选择默认即可）。
4. 安装完成后，可以通过命令行或图形化工具（如MySQL Workbench）连接到MySQL服务器进行管理和操作。
## 3. MySQL基本操作
### 3.1 连接MySQL
使用命令行连接MySQL：
```bash
mysql -u 用户名 -p
``` 
输入密码后即可连接到MySQL服务器。
### 3.2 创建数据库
```sql
CREATE DATABASE 数据库名;
```
实际运用时，一般会指定字符集和排序规则和确保数据库不重复创建：
```sql
CREATE DATABASE IF NOT EXISTS 数据库名 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```
后面的创建表和插入数据等操作也可以使用`IF NOT EXISTS`来避免重复创建或插入数据，删除表和数据库时可以使用`IF EXISTS`来避免错误：
### 3.3 创建表
```sql
CREATE TABLE 表名 (
    列名 数据类型,
    列名 数据类型,
    ...
);
```
### 3.4 插入数据
```sql
INSERT INTO 表名 (列1, 列2, ...) VALUES (值1, 值2, ...);
```
### 3.5 查询数据
```sql
SELECT 列1, 列2, ... FROM 表名 WHERE 条件;
```
### 3.6 更新数据
```sql
UPDATE 表名 SET 列1 = 值1, 列2 = 值2, ... WHERE 条件;
```
### 3.7 删除数据
```sql
DELETE FROM 表名 WHERE 条件;
```
### 3.8 删除表
```sql
DROP TABLE 表名;
```
### 3.9 删除数据库
```sql
DROP DATABASE 数据库名;
```
### 3.10 其他常用操作
#### 显示数据库列表
```sql
SHOW DATABASES;
```
#### 显示表列表
```sql
SHOW TABLES;
```
#### 显示表结构
```sql
DESCRIBE 表名;
```
#### 显示当前数据库
```sql
SELECT DATABASE();
```
#### 切换数据库
```sql
USE 数据库名;
```
## 4. MySQL高级功能
### 4.1 索引
索引可以提高查询性能，创建索引的语法如下：
```sql
CREATE INDEX 索引名 ON 表名 (列名);
```
索引的类型有多种，如B树索引、哈希索引、全文索引等，可以根据实际需求选择合适的索引类型来优化查询性能。
索引遵循以下原则：
1. 索引列的选择：选择经常用于查询条件、排序或连接的列作为索引列。
2. 索引的数量：过多的索引会增加写操作的开销，因此应根据实际需求合理创建索引。
3. 索引的维护：定期检查和优化索引，删除不再使用的索引，以保持数据库性能。
4. 最左前缀原则：对于多列索引，查询条件必须包含索引的最左前缀才能使用索引。
### 4.2 视图
视图是一个虚拟表——即一个写好的SQL查询，可以简化复杂查询，创建视图的语法如下：
```sql
CREATE VIEW 视图名 AS SELECT 列1, 列2, ... FROM 表名 WHERE 条件;
```
调用时直接查询视图即可：
```sql
SELECT * FROM 视图名;
```
而不用每次都写复杂的SQL查询。视图的优点包括：
1. 简化查询：视图可以封装复杂的查询逻辑，使得用户可以通过简单的查询来获取所需的数据。
2. 数据安全：视图可以限制用户访问底层表的数据，只暴露必要的列和行，从而提高数据安全性。
3. 数据一致性：视图可以确保数据的一致性，特别是在多个表之间进行连接查询时，可以通过视图来统一数据的展示和访问。
4. 代码重用：视图可以被多个查询重用，避免重复编写相同的查询逻辑，提高代码的可维护性和效率。

### 4.3 存储过程
存储过程是一组预编译的SQL语句，可以提高性能和代码重用性，创建存储过程的语法如下：
```sql
CREATE PROCEDURE 过程名()
BEGIN
    -- SQL语句
END;
```
### 4.4 触发器
触发器是在特定事件发生时自动执行的SQL代码块，创建触发器的语法如下：
```sql
DELIMITER // # 更改语句结束符，避免与触发器内的SQL语句冲突
CREATE TRIGGER trigger_name # 触发器名称
AFTER INSERT ON table_name # 触发器类型（AFTER/BEFORE）和事件（INSERT/UPDATE/DELETE）
FOR EACH ROW # 每行触发
BEGIN
    -- SQL语句 # 触发器内的SQL代码
END//
DELIMITER ;
```
### 4.5 事务
事务是一组SQL操作的集合，要么全部执行成功，要么全部回滚，使用事务的语法如下：
```sql
START TRANSACTION; -- 开始事务
COMMIT; -- 提交事务
ROLLBACK; -- 回滚事务
```
### 4.6 用户管理、权限控制
MySQL提供了强大的用户管理和权限控制功能，可以通过以下语法进行用户管理：
| 权限级别 | 语法示例                 | 作用范围         |
| ---- | -------------------- | ------------ |
| 全局级  | `*.*`                | 整个 MySQL 服务器 |
| 数据库级 | `db_name.*`          | 某个数据库        |
| 表级   | `db_name.table_name` | 某张表          |
| 列级   | `(column)`           | 某个字段         |
```sql
-- 创建用户
CREATE USER 'username'@'host' IDENTIFIED BY 'password';
-- 授予全部权限
GRANT ALL PRIVILEGES ON db_name.* TO 'username'@'host';
-- 授予特定权限
GRANT SELECT, INSERT ON db_name.* TO 'username'@'host';
-- 撤销全部权限
REVOKE ALL PRIVILEGES ON db_name.* FROM 'username'@'host';
-- 撤销特定权限
REVOKE SELECT, INSERT ON db_name.* FROM 'username'@'host';
-- 删除用户
DROP USER 'username'@'host';
```
## 5. MySQL高级查询
### 5.1 联合查询
```sql
SELECT 列1, 列2 FROM 表1
UNION
SELECT 列1, 列2 FROM 表2;
```
### 5.2 子查询
```sql
SELECT 列1 FROM 表1 WHERE 列2 IN (SELECT 列2 FROM 表2 WHERE 条件);
```
### 5.3 聚合函数
```sql
SELECT COUNT(*), SUM(列1), AVG(列2) FROM 表名 WHERE 条件;
```
### 5.4 分组查询
```sql
SELECT 列1, COUNT(*) FROM 表名 GROUP BY 列1;
```
### 5.5 排序查询
```sql
SELECT 列1, 列2 FROM 表名 ORDER BY 列1 ASC, 列2 DESC;
```
### 5.6 分页查询
```sql
SELECT 列1, 列2 FROM 表名 LIMIT 偏移量, 行数;
```
### 5.7 连接查询
```sql
SELECT 列1, 列2 FROM 表1
JOIN 表2 ON 表1.列 = 表2.列;
```
连接查询的类型包括内连接（INNER JOIN）——只返回两个表中都存在匹配记录的行、左连接（LEFT JOIN）——返回左表中的所有记录以及右表中匹配的记录、右连接（RIGHT JOIN）——返回右表中的所有记录以及左表中匹配的记录、全连接（FULL JOIN）——返回两个表中的所有记录，可以根据实际需求选择合适的连接类型来获取所需的数据。