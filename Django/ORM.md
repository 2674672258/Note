## 一、ORM 基础概念

ORM（Object-Relational Mapping）是一种将对象模型与关系数据库映射的技术。Django ORM 让你可以用 Python 代码操作数据库，而不需要写 SQL。

### 核心优势
- 数据库无关性（支持 SQLite、PostgreSQL、MySQL 等）
- 防止 SQL 注入
- 代码可读性强
- 自动处理数据库迁移

---

## 二、模型定义

### 2.1 基本模型

```python
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)
    published_date = models.DateField()
    price = models.DecimalField(max_digits=6, decimal_places=2)
    is_available = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'books'  # 自定义表名
        ordering = ['published_date']  # 按published_date升序排序，降序则为['-published_date']
        verbose_name = '书籍' # 显示在admin管理视角
        verbose_name_plural = '书籍列表' # 具体作用如下所示
```
   * 进入列表页时，页面标题会显示：

     ```
     选择 书籍合集 来更改
     ```
   * 如果没写 `verbose_name_plural`，可能会显示：

     ```
     选择 书籍s 来更改
     ```

### 2.2 字段类型大全

* 1. 字符串类字段

| 字段           | 数据库类型           | 必填参数         | 特点/用途          | 备注                               |
| ------------ | --------------- | ------------ | -------------- | -------------------------------- |
| `CharField`  | VARCHAR(n)      | `max_length` | 短文本，如姓名、标题     | 可加 `choices`、`unique`、`db_index` |
| `TextField`  | TEXT / LONGTEXT | -            | 长文本，如文章内容、评论   | 不强制 `max_length`，不推荐索引           |
| `EmailField` | VARCHAR(254)    | -            | 邮箱地址，自动格式校验    | 本质是 `CharField` + 邮箱验证           |
| `URLField`   | VARCHAR(200)    | -            | URL 地址，自动格式校验  | 可加 `verify_exists`（老版本）          |
| `SlugField`  | VARCHAR(50)     | -            | URL友好字符串，如文章别名 | 默认可唯一，常用于路由                      |
| `UUIDField`  | CHAR(32)/UUID   | -            | 存储 UUID        | 可设 `default=uuid.uuid4` 自动生成     |

---

* 2. 数值类字段

| 字段                          | 数据库类型             | 必填参数                         | 特点/用途    |
| --------------------------- | ----------------- | ---------------------------- | -------- |
| `IntegerField`              | INT               | -                            | 整数       |
| `SmallIntegerField`         | SMALLINT          | -                            | 小整数      |
| `BigIntegerField`           | BIGINT            | -                            | 大整数      |
| `PositiveIntegerField`      | INT UNSIGNED      | -                            | 正整数      |
| `PositiveSmallIntegerField` | SMALLINT UNSIGNED | -                            | 小正整数     |
| `FloatField`                | FLOAT             | -                            | 浮点数      |
| `DecimalField`              | DECIMAL           | `max_digits, decimal_places` | 精确小数，如金额 |

---

* 3. 布尔与空值

| 字段                               | 数据库类型      | 必填参数 | 特点/用途               |
| -------------------------------- | ---------- | ---- | ------------------- |
| `BooleanField`                   | TINYINT(1) | -    | True / False        |
| `NullBooleanField`（Django3.1已弃用） | TINYINT(1) | -    | True / False / NULL |

---

* 4. 日期和时间

| 字段              | 数据库类型                 | 必填参数 | 特点/用途          |
| --------------- | --------------------- | ---- | -------------- |
| `DateField`     | DATE                  | -    | 日期             |
| `DateTimeField` | DATETIME              | -    | 日期+时间          |
| `TimeField`     | TIME                  | -    | 时间             |
| `DurationField` | BIGINT                | -    | 时间差（timedelta） |
| `AutoField`     | INT AUTO_INCREMENT    | -    | 主键，自增          |
| `BigAutoField`  | BIGINT AUTO_INCREMENT | -    | 主键，自增，适合大表     |

---
* 时间日期特有参数 `auto_now`、`auto_now_add` 和 `default`
* * 1. auto_now

* **作用**：每次 **保存对象时**（`save()`）自动更新为当前时间。
* **典型用途**：记录“最后修改时间”。
* **特点**：

  * 无需手动设置，每次修改都会覆盖。
  * **无法手动修改**，即使你给字段赋值，保存时也会被 `auto_now` 覆盖。

**示例**：

```python
updated_at = models.DateTimeField(auto_now=True)
```

* 每次 `obj.save()` → `updated_at` 自动更新为当前时间。
* 常用于日志、文章修改时间等。

---

* * 2. auto_now_add

* **作用**：仅在 **对象首次创建时** 自动设置为当前时间。
* **典型用途**：记录“创建时间”。
* **特点**：

  * 创建对象时自动赋值，以后保存不会被覆盖。
  * 可以理解为“默认值是创建时间”，但只在第一次生效。

**示例**：

```python
created_at = models.DateTimeField(auto_now_add=True)
```

* 第一次 `obj.save()` → `created_at` 自动填充
* 后续修改 `obj.save()` → `created_at` 不变

---

* * 3. default

* **作用**：设置字段默认值（可以是固定值，也可以是函数）。
* **典型用途**：允许手动修改，同时提供初始值。
* **特点**：

  * **手动赋值不会被覆盖**，与 `auto_now` 不同。
  * 可以用函数动态生成默认值，比如当前时间。

**示例**：

```python
from django.utils import timezone

published_at = models.DateTimeField(default=timezone.now)
```

* 创建对象时，如果没有提供 `published_at`，默认填 `timezone.now()`
* 可以手动覆盖：

```python
book.published_at = datetime.datetime(2024, 9, 30, 12, 0)
book.save()  # 保存时不会被覆盖
```
---

* 5. 文件与媒体

| 字段           | 数据库类型   | 必填参数        | 特点/用途          |
| ------------ | ------- | ----------- | -------------- |
| `FileField`  | VARCHAR | `upload_to` | 文件上传路径         |
| `ImageField` | VARCHAR | `upload_to` | 图片上传，需要 Pillow |

---

* 6. 外键与关系

| 字段                | 数据库类型 | 必填参数 | 特点/用途               |
| ----------------- | ----- | ---- | ------------------- |
| `ForeignKey`      | INT   | `to` | 多对一关系               |
| `OneToOneField`   | INT   | `to` | 一对一关系               |
| `ManyToManyField` | -     | `to` | 多对多关系，Django 会生成中间表 |

---

* 7. 特殊字段

| 字段                      | 特点                            |
| ----------------------- | ----------------------------- |
| `JSONField`             | 存储 JSON，PostgreSQL 原生 JSON 支持 |
| `BinaryField`           | 存储二进制数据                       |
| `GenericIPAddressField` | IPv4/IPv6 地址，自动校验             |

---

* 8. 公共参数（大部分字段都支持）

| 参数             | 作用              |
| -------------- | --------------- |
| `null`         | 数据库是否允许 NULL    |
| `blank`        | 表单验证是否允许为空      |
| `default`      | 默认值             |
| `unique`       | 唯一约束            |
| `db_index`     | 是否建索引           |
| `choices`      | 枚举选项            |
| `verbose_name` | 可读名称，显示在 admin  |
| `help_text`    | 表单或 admin 的提示信息 |
| `editable`     | 是否在 admin/表单可编辑 |
| `validators`   | 自定义验证函数列表       |

---

### 2.3 字段参数

#### 通用参数
```python
field = models.CharField(
    max_length=100,
    null=True,              # 数据库允许 NULL
    blank=True,             # 表单验证允许空
    default='默认值',        # 默认值
    unique=True,            # 唯一约束
    db_index=True,          # 创建索引
    db_column='custom_name', # 自定义列名
    primary_key=True,       # 设为主键
    editable=False,         # 不在表单中显示
    choices=[               # 选项
        ('draft', '草稿'),
        ('published', '已发布'),
    ],
    help_text='帮助文本',   # 帮助说明
    verbose_name='字段名称',# 人类可读名称
    validators=[validator], # 自定义验证器
)
```

### 2.4 关系字段

#### ForeignKey（一对多）
```python
class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    author = models.ForeignKey(
        Author,
        on_delete=models.CASCADE,  # 删除策略
        related_name='books',      # 反向查询名称
        related_query_name='book', # 反向查询过滤名称
        limit_choices_to={'is_active': True},  # 限制可选项
        db_constraint=True,        # 是否创建外键约束
    )

# on_delete 选项：
# CASCADE - 级联删除
# PROTECT - 保护，阻止删除
# SET_NULL - 设为 NULL（需要 null=True）
# SET_DEFAULT - 设为默认值
# SET() - 设为指定值
# DO_NOTHING - 什么都不做
```

#### ManyToManyField（多对多）
```python
class Student(models.Model):
    name = models.CharField(max_length=100)

class Course(models.Model):
    name = models.CharField(max_length=100)
    students = models.ManyToManyField(
        Student,
        related_name='courses',
        through='Enrollment',      # 中间表
        through_fields=('course', 'student'),
        db_table='custom_m2m_table',
    )

# 自定义中间表
class Enrollment(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    enrolled_date = models.DateField()
    grade = models.CharField(max_length=2)
```

#### OneToOneField（一对一）
```python
class User(models.Model):
    username = models.CharField(max_length=100)

class Profile(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile',
    )
    bio = models.TextField()
```

---

## 三、QuerySet API

### 3.1 创建对象

```python
# 方法 1：创建并保存
book = Book(title='Django 教程', author='张三', price=99.00)
book.save()

# 方法 2：create() 直接创建并保存
book = Book.objects.create(title='Django 教程', author='张三', price=99.00)

# 方法 3：get_or_create() 获取或创建
book, created = Book.objects.get_or_create(
    title='Django 教程',
    defaults={'author': '张三', 'price': 99.00}
)

# 方法 4：update_or_create() 更新或创建
book, created = Book.objects.update_or_create(
    title='Django 教程',
    defaults={'author': '李四', 'price': 89.00}
)

# 批量创建
books = [
    Book(title='书1', author='作者1', price=50),
    Book(title='书2', author='作者2', price=60),
]
Book.objects.bulk_create(books, batch_size=100)
```

### 3.2 查询对象

#### 基础查询
```python
# 获取所有对象
books = Book.objects.all()

# 获取单个对象
book = Book.objects.get(id=1)  # 不存在会抛出 DoesNotExist 异常

# 过滤
books = Book.objects.filter(author='张三')
books = Book.objects.filter(price__gt=50)  # 价格大于 50

# 排除
books = Book.objects.exclude(author='张三')

# 获取第一个/最后一个
book = Book.objects.first()
book = Book.objects.last()

# 获取最新/最早（需要 Meta.get_latest_by 或指定字段）
book = Book.objects.latest('published_date')
book = Book.objects.earliest('published_date')
```

#### 字段查找（Field Lookups）
```python
# 精确匹配
Book.objects.filter(title__exact='Django')
Book.objects.filter(title='Django')  # 等同于上面

# 不区分大小写
Book.objects.filter(title__iexact='django')

# 包含
Book.objects.filter(title__contains='Django')
Book.objects.filter(title__icontains='django')  # 不区分大小写

# 开始/结束
Book.objects.filter(title__startswith='Django')
Book.objects.filter(title__endswith='教程')
Book.objects.filter(title__istartswith='django')  # 不区分大小写

# 在列表中
Book.objects.filter(id__in=[1, 2, 3])

# 范围
Book.objects.filter(price__range=(50, 100))

# 比较
Book.objects.filter(price__gt=50)   # 大于
Book.objects.filter(price__gte=50)  # 大于等于
Book.objects.filter(price__lt=100)  # 小于
Book.objects.filter(price__lte=100) # 小于等于

# 日期
Book.objects.filter(published_date__year=2024)
Book.objects.filter(published_date__month=12)
Book.objects.filter(published_date__day=25)
Book.objects.filter(created_at__date='2024-01-01')
Book.objects.filter(created_at__time='14:30')

# NULL 判断
Book.objects.filter(description__isnull=True)

# 正则表达式
Book.objects.filter(title__regex=r'^Django')
Book.objects.filter(title__iregex=r'^django')  # 不区分大小写
```

#### 跨关系查询
```python
# 正向查询（通过外键）
books = Book.objects.filter(author__name='张三')
books = Book.objects.filter(author__age__gt=30)

# 反向查询（通过 related_name）
author = Author.objects.get(id=1)
books = author.books.all()

# 或使用 filter
authors = Author.objects.filter(books__title='Django')

# 多层关系
books = Book.objects.filter(author__country__name='中国')
```

### 3.3 复杂查询

#### Q 对象（OR、NOT 查询）
```python
from django.db.models import Q

# OR 查询
books = Book.objects.filter(Q(author='张三') | Q(author='李四'))

# AND 查询
books = Book.objects.filter(Q(author='张三') & Q(price__gt=50))

# NOT 查询
books = Book.objects.filter(~Q(author='张三'))

# 复杂组合
books = Book.objects.filter(
    Q(author='张三') & (Q(price__gt=50) | Q(is_available=True))
)
```

#### F 对象（字段比较）
```python
from django.db.models import F

# 字段间比较
books = Book.objects.filter(discount_price__lt=F('original_price'))

# 字段运算
Book.objects.update(price=F('price') * 1.1)  # 涨价 10%

# 字段引用
Book.objects.filter(title=F('author__name'))
```

### 3.4 聚合和分组

#### 聚合函数
```python
from django.db.models import Count, Sum, Avg, Max, Min

# 计数
count = Book.objects.count()
count = Book.objects.filter(author='张三').count()

# 总和
total = Book.objects.aggregate(total_price=Sum('price'))
# 返回: {'total_price': 1000}

# 平均值
avg = Book.objects.aggregate(avg_price=Avg('price'))

# 最大/最小值
Book.objects.aggregate(max_price=Max('price'), min_price=Min('price'))

# 多个聚合
result = Book.objects.aggregate(
    total=Sum('price'),
    avg=Avg('price'),
    count=Count('id')
)
```

#### 分组（annotate）
```python
# 按作者分组，统计每个作者的书籍数量
authors = Author.objects.annotate(book_count=Count('books'))
for author in authors:
    print(f"{author.name}: {author.book_count} 本书")

# 按年份分组
from django.db.models.functions import TruncYear
books_by_year = Book.objects.annotate(
    year=TruncYear('published_date')
).values('year').annotate(count=Count('id'))

# 复杂分组
Author.objects.annotate(
    total_price=Sum('books__price'),
    avg_price=Avg('books__price'),
    book_count=Count('books')
).filter(book_count__gt=5)
```

### 3.5 排序

```python
# 升序
books = Book.objects.order_by('price')

# 降序
books = Book.objects.order_by('-price')

# 多字段排序
books = Book.objects.order_by('author', '-published_date')

# 随机排序
books = Book.objects.order_by('?')

# 按关联字段排序
books = Book.objects.order_by('author__name')

# 清除默认排序
books = Book.objects.order_by()
```

### 3.6 切片和分页

```python
# 切片（不会立即执行查询）
books = Book.objects.all()[0:5]      # 前 5 本
books = Book.objects.all()[5:10]     # 第 6-10 本
book = Book.objects.all()[0]         # 第一本

# 负数索引不支持
# books = Book.objects.all()[-1]     # 错误！

# 分页
from django.core.paginator import Paginator

books = Book.objects.all()
paginator = Paginator(books, 10)  # 每页 10 条

page1 = paginator.page(1)
for book in page1:
    print(book.title)
```

### 3.7 去重

```python
# 去重
books = Book.objects.values('author').distinct()

# 指定字段去重（仅 PostgreSQL）
books = Book.objects.distinct('author', 'published_date')
```

### 3.8 限制返回字段

```python
# values() - 返回字典
books = Book.objects.values('id', 'title', 'price')
# [{'id': 1, 'title': 'Django', 'price': 99.00}, ...]

# values_list() - 返回元组
books = Book.objects.values_list('title', 'price')
# [('Django', 99.00), ...]

# 返回单个字段的列表
titles = Book.objects.values_list('title', flat=True)
# ['Django', 'Python', ...]

# only() - 只查询指定字段（返回模型实例）
books = Book.objects.only('title', 'price')

# defer() - 延迟加载指定字段
books = Book.objects.defer('description')
```

### 3.9 更新对象

```python
# 方法 1：获取并更新
book = Book.objects.get(id=1)
book.price = 89.00
book.save()

# 只更新指定字段
book.save(update_fields=['price'])

# 方法 2：批量更新
Book.objects.filter(author='张三').update(price=99.00)

# 使用 F 对象更新
Book.objects.update(price=F('price') * 1.1)

# 更新并返回受影响的行数
count = Book.objects.filter(is_available=False).update(is_available=True)
```

### 3.10 删除对象

```python
# 删除单个对象
book = Book.objects.get(id=1)
book.delete()

# 批量删除
Book.objects.filter(price__lt=10).delete()

# 删除所有
Book.objects.all().delete()

# 返回删除信息
result = Book.objects.filter(author='张三').delete()
# (5, {'app.Book': 5})  # 删除了 5 个对象
```

### 3.11 性能优化

#### select_related（一对一、多对一）
```python
# 未优化：会产生 N+1 查询
books = Book.objects.all()
for book in books:
    print(book.author.name)  # 每次循环都查询数据库

# 优化：使用 JOIN
books = Book.objects.select_related('author')
for book in books:
    print(book.author.name)  # 不会再查询数据库

# 多层关系
books = Book.objects.select_related('author__country')
```

#### prefetch_related（一对多、多对多）
```python
# 未优化
authors = Author.objects.all()
for author in authors:
    for book in author.books.all():  # N+1 查询
        print(book.title)

# 优化
authors = Author.objects.prefetch_related('books')
for author in authors:
    for book in author.books.all():  # 已预取，不会查询
        print(book.title)

# 自定义预取查询
from django.db.models import Prefetch

authors = Author.objects.prefetch_related(
    Prefetch('books', queryset=Book.objects.filter(is_available=True))
)
```

#### 批量操作
```python
# bulk_update - 批量更新
books = Book.objects.all()
for book in books:
    book.price *= 1.1
Book.objects.bulk_update(books, ['price'], batch_size=100)

# bulk_create - 批量创建（前面已提到）
```

### 3.12 原始 SQL

```python
# raw() - 执行原始 SQL 并返回模型实例
books = Book.objects.raw('SELECT * FROM books WHERE price > %s', [50])

# 使用游标直接执行 SQL
from django.db import connection

with connection.cursor() as cursor:
    cursor.execute("SELECT * FROM books WHERE price > %s", [50])
    rows = cursor.fetchall()

# 执行自定义 SQL
from django.db import connection

def custom_sql():
    with connection.cursor() as cursor:
        cursor.execute("UPDATE books SET price = price * 1.1")
```

### 3.13 事务

```python
from django.db import transaction

# 方法 1：装饰器
@transaction.atomic
def create_books():
    Book.objects.create(title='Book 1', price=50)
    Book.objects.create(title='Book 2', price=60)

# 方法 2：上下文管理器
def create_books():
    with transaction.atomic():
        Book.objects.create(title='Book 1', price=50)
        Book.objects.create(title='Book 2', price=60)
        # 如果这里抛出异常，所有操作都会回滚

# 保存点
with transaction.atomic():
    book1 = Book.objects.create(title='Book 1', price=50)
    
    sid = transaction.savepoint()
    try:
        book2 = Book.objects.create(title='Book 2', price=60)
    except:
        transaction.savepoint_rollback(sid)
    else:
        transaction.savepoint_commit(sid)
```

### 3.14 其他有用的方法

```python
# exists() - 检查是否存在
has_books = Book.objects.filter(author='张三').exists()

# contains() - 检查对象是否在 QuerySet 中
book = Book.objects.get(id=1)
is_in = Book.objects.filter(author='张三').contains(book)

# in_bulk() - 批量获取并返回字典
books = Book.objects.in_bulk([1, 2, 3])
# {1: <Book: Book 1>, 2: <Book: Book 2>, 3: <Book: Book 3>}

# iterator() - 迭代大量数据（节省内存）
for book in Book.objects.iterator(chunk_size=100):
    print(book.title)

# explain() - 查看执行计划（调试用）
print(Book.objects.filter(price__gt=50).explain())
```

---

## 四、模型方法和属性

### 4.1 模型方法

```python
class Book(models.Model):
    title = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=6, decimal_places=2)
    discount = models.DecimalField(max_digits=3, decimal_places=2)
    
    def __str__(self):
        """字符串表示"""
        return self.title
    
    def get_absolute_url(self):
        """获取对象的 URL"""
        from django.urls import reverse
        return reverse('book_detail', args=[str(self.id)])
    
    def save(self, *args, **kwargs):
        """重写 save 方法"""
        # 保存前的逻辑
        self.title = self.title.upper()
        super().save(*args, **kwargs)
        # 保存后的逻辑
    
    def delete(self, *args, **kwargs):
        """重写 delete 方法"""
        # 删除前的逻辑
        super().delete(*args, **kwargs)
        # 删除后的逻辑
    
    @property
    def discounted_price(self):
        """计算属性"""
        return self.price * (1 - self.discount)
    
    class Meta:
        ordering = ['-published_date']
        verbose_name = '书籍'
        verbose_name_plural = '书籍列表'
```

### 4.2 Manager（管理器）

```python
# 自定义 Manager
class AvailableBookManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(is_available=True)

class Book(models.Model):
    title = models.CharField(max_length=200)
    is_available = models.BooleanField(default=True)
    
    objects = models.Manager()  # 默认管理器
    available = AvailableBookManager()  # 自定义管理器

# 使用
all_books = Book.objects.all()  # 所有书籍
available_books = Book.available.all()  # 只有可用的书籍
```

---

## 五、信号（Signals）

```python
from django.db.models.signals import pre_save, post_save, pre_delete, post_delete
from django.dispatch import receiver

# pre_save - 保存前触发
@receiver(pre_save, sender=Book)
def book_pre_save(sender, instance, **kwargs):
    print(f"准备保存: {instance.title}")

# post_save - 保存后触发
@receiver(post_save, sender=Book)
def book_post_save(sender, instance, created, **kwargs):
    if created:
        print(f"创建了新书: {instance.title}")
    else:
        print(f"更新了书: {instance.title}")

# pre_delete - 删除前触发
@receiver(pre_delete, sender=Book)
def book_pre_delete(sender, instance, **kwargs):
    print(f"准备删除: {instance.title}")

# post_delete - 删除后触发
@receiver(post_delete, sender=Book)
def book_post_delete(sender, instance, **kwargs):
    print(f"已删除: {instance.title}")
```

---

## 六、数据库迁移

```bash
# 创建迁移文件
python manage.py makemigrations

# 查看迁移 SQL
python manage.py sqlmigrate app_name 0001

# 执行迁移
python manage.py migrate

# 查看迁移状态
python manage.py showmigrations

# 回滚迁移
python manage.py migrate app_name 0001

# 创建空迁移
python manage.py makemigrations --empty app_name
```

---

## 七、常见模式和最佳实践

### 7.1 软删除
```python
class SoftDeleteManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(is_deleted=False)

class Book(models.Model):
    title = models.CharField(max_length=200)
    is_deleted = models.BooleanField(default=False)
    
    objects = SoftDeleteManager()
    all_objects = models.Manager()
    
    def delete(self, *args, **kwargs):
        self.is_deleted = True
        self.save()
```

### 7.2 时间戳
```python
class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        abstract = True  # 抽象基类

class Book(TimeStampedModel):
    title = models.CharField(max_length=200)
    # 自动包含 created_at 和 updated_at
```

### 7.3 性能优化技巧
```python
# 1. 使用 select_related 和 prefetch_related
books = Book.objects.select_related('author').prefetch_related('tags')

# 2. 只查询需要的字段
books = Book.objects.only('title', 'price')

# 3. 使用 iterator() 处理大数据集
for book in Book.objects.iterator(chunk_size=1000):
    process(book)

# 4. 批量操作
Book.objects.bulk_create([...])
Book.objects.bulk_update([...], ['price'])

# 5. 使用数据库索引
class Book(models.Model):
    title = models.CharField(max_length=200, db_index=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['author', 'published_date']),
        ]
```

---

## 八、调试技巧

```python
# 1. 打印生成的 SQL
from django.db import connection
print(connection.queries)

# 2. 使用 django-debug-toolbar

# 3. 查看 QuerySet 的 SQL
print(Book.objects.filter(price__gt=50).query)

# 4. 使用 explain()
print(Book.objects.filter(price__gt=50).explain())

# 5. 统计查询次数
from django.test.utils import override_settings
from django.db import reset_queries

reset_queries()
# 执行查询
books = Book.objects.all()
for book in books:
    print(book.author.name)

print(len(connection.queries))  # 查询次数
```

---

## 九、总结

Django ORM 的核心要点：
1. **模型定义**：使用 models.Model 定义数据结构
2. **字段类型**：选择合适的字段类型和参数
3. **关系**：理解 ForeignKey、ManyToManyField、OneToOneField
4. **查询**：掌握 filter、exclude、get、all 等方法
5. **字段查找**：使用 __gt、__contains 等查找符
6. **Q/F 对象**：实现复杂查询和字段操作
7. **聚合分组**：使用 aggregate 和 annotate
8. **性能优化**：select_related、prefetch_related
9. **事务**：确保数据一致性
10. **迁移**：管理数据库结构变更

记住：ORM 的目标是用 Python 代码操作数据库，让开发更高效，但也要注意性能问题，必要时可以使用原始 SQL。