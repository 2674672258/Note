# 146.简述Django请求生命周期

一般是用户通过浏览器向我们的服务器发起一个请求(request)这个请求会去访问视图函数，如果不涉及到数据据调用，那么这个时候试图函数返回一个模板也就是一个网页给用户)

视图函数调用模板类去数据库查找数据，然后逐级返回，视图函数把返回的数据填充到模板中空格中，最后返回网页给用户。

1.wsgi，请求封装后交给web框架（Flask，Django)
2.中间件，对请求来进行校验或在请求对象中添加其他相关数据，例如：csrf,request.session
3.路由匹配 根据浏览器发送的不同url去匹配不同的视图函数
4.视图函数，在视图函数中进行业务逻辑的处理，可能涉及到：orm，templates
5.中间件，对响应的数据进行处理
6.wsgi，将响应的内容发送给浏览器

# 147.用的restframework完成api发送时间时区

当前的问题是用django的rest framework模块做一个get请求的发送时间以及时区信息的api
```python
class getCurrenttime(APIView):
    def get(self, request):
        local_time = time.localtime()
        time_zone = settings.TIME_ZONE
        temp = {'localtime': local_time, 'timezone': time_zone}
        return Response(temp)
        