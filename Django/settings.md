Django 的 `settings.py` 是整个项目的大脑，它负责管理项目的全局配置，比如数据库、缓存、日志、静态文件、应用注册等等。一般位于项目同名子目录下
---

### 1. **基本配置**

* **BASE\_DIR**
  项目根目录路径，一般通过 `Path(__file__).resolve().parent.parent` 获取。所有路径相关的设置都最好基于它。

* **SECRET\_KEY**
  Django 的密钥，用于加密 session、CSRF token 等。生产环境一定要保密，不能泄露。

* **DEBUG**
  开发环境下设为 `True`，生产环境必须设为 `False`，否则调试信息会暴露给用户。

* **ALLOWED\_HOSTS**
  允许访问的主机名列表，例如：

  ```python
  ALLOWED_HOSTS = ["127.0.0.1", "localhost", "example.com"]
  ```

---

### 2. **应用相关**

* **INSTALLED\_APPS**
  注册的应用（app），包括 Django 内置的（如 `django.contrib.admin`）和你自己写的 app。
  它决定了哪些 app 的模型、模板标签、信号等会被加载。

* **MIDDLEWARE**
  中间件列表，每一个中间件会在请求/响应经过时处理，比如：

  * `SecurityMiddleware`（安全相关）
  * `SessionMiddleware`（启用会话）
  * `AuthenticationMiddleware`（认证用户）
  * `CsrfViewMiddleware`（CSRF 防护）

* **ROOT\_URLCONF**
  项目 URL 配置的入口，一般是 `urls.py`。

* **WSGI\_APPLICATION**
  WSGI 网关配置，告诉 WSGI 服务器入口在哪。

* **ASGI\_APPLICATION**
  如果用异步（Django Channels 等），会用到 ASGI。

---

### 3. **数据库配置**

Django 默认用 SQLite，你可以换成 MySQL/PostgreSQL 等：

```python
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",  # 数据库引擎
        "NAME": "mydb",
        "USER": "myuser",
        "PASSWORD": "mypassword",
        "HOST": "127.0.0.1",
        "PORT": "5432",
    }
}
```

---

### 4. **模板和静态文件**

* **TEMPLATES**
  模板引擎配置，决定 Django 如何加载 `.html` 文件，是否支持 Jinja2 等。

* **STATIC\_URL / STATICFILES\_DIRS / STATIC\_ROOT**

  * `STATIC_URL`：访问静态文件的 URL 前缀。
  * `STATICFILES_DIRS`：开发环境下的静态文件目录。
  * `STATIC_ROOT`：`collectstatic` 命令收集的目标目录（生产环境用）。

* **MEDIA\_URL / MEDIA\_ROOT**
  用户上传文件相关配置。

  * `MEDIA_URL = "/media/"`
  * `MEDIA_ROOT = BASE_DIR / "media"`

---

### 5. **认证与安全**

* **AUTH\_PASSWORD\_VALIDATORS**
  密码校验规则，比如最小长度、复杂度。

* **AUTH\_USER\_MODEL**
  如果需要自定义用户模型，需要在这里配置。

* **CSRF\_TRUSTED\_ORIGINS**
  允许跨站请求的域名（跨域时必须设置，否则会被 CSRF 拦截）。

---

### 6. **国际化与时区**

```python
LANGUAGE_CODE = "zh-hans"
TIME_ZONE = "Asia/Shanghai"
USE_I18N = True
USE_TZ = True
```

* `LANGUAGE_CODE`：默认语言。
* `TIME_ZONE`：时区。
* `USE_I18N`：是否启用国际化。
* `USE_TZ`：是否使用时区感知的时间（建议 True）。

---

### 7. **日志配置**

Django 的日志系统可以在 `settings.py` 配置，比如写入文件、输出到控制台等：

```python
LOGGING = {
    "version": 1,
    "handlers": {
        "file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": BASE_DIR / "debug.log",
        },
    },
    "loggers": {
        "django": {
            "handlers": ["file"],
            "level": "DEBUG",
            "propagate": True,
        },
    },
}
```

---

### 8. **缓存配置**

Django 支持多种缓存（内存、Redis、Memcached）：

```python
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/1",
        "OPTIONS": {"CLIENT_CLASS": "django_redis.client.DefaultClient"},
    }
}
```

---

### 9. **Email 配置**

用于发送验证邮件、找回密码：

```python
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = "smtp.qq.com"
EMAIL_PORT = 465
EMAIL_USE_SSL = True
EMAIL_HOST_USER = "xxx@qq.com"
EMAIL_HOST_PASSWORD = "授权码"
```

---

### 10. **第三方库的配置**

很多第三方库会依赖 `settings.py` 来存储配置，比如：

* `REST_FRAMEWORK`（DRF 配置）
* `CELERY_BROKER_URL`（Celery 配置）
* `CHANNEL_LAYERS`（Django Channels 配置）
