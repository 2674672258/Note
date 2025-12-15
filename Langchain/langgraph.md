# LangGraph 从零基础到精通完整教程

## 目录
1. [LangGraph 简介](#1-langgraph-简介)
2. [环境搭建](#2-环境搭建)
3. [核心概念](#3-核心概念)
4. [基础教程](#4-基础教程)
5. [进阶教程](#5-进阶教程)
6. [高级特性](#6-高级特性)
7. [实战项目](#7-实战项目)
8. [最佳实践](#8-最佳实践)
9. [性能优化](#9-性能优化)
10. [常见问题](#10-常见问题)

---

## 1. LangGraph 简介

### 1.1 什么是 LangGraph？

LangGraph 是由 LangChain 团队开发的一个用于构建有状态、多参与者应用的框架。它专门用于创建基于 LLM（大型语言模型）的复杂、循环的工作流。

**核心特点：**
- 基于图结构的工作流设计
- 内置状态管理
- 支持循环和条件分支
- 人机交互（Human-in-the-loop）
- 持久化和检查点机制
- 流式输出支持

### 1.2 为什么需要 LangGraph？

传统的 LangChain 链式调用是线性的，难以处理：
- 需要循环的场景（如迭代优化）
- 复杂的条件分支逻辑
- 多智能体协作
- 需要人工干预的流程
- 长时间运行的任务

LangGraph 通过图结构完美解决这些问题。

### 1.3 应用场景

- **智能客服系统**：多轮对话、意图识别、问题路由
- **代码助手**：代码生成、调试、测试、优化的迭代流程
- **研究助手**：信息搜索、分析、总结的循环工作流
- **多智能体系统**：多个 AI 代理协作完成复杂任务
- **工作流自动化**：需要决策分支的业务流程

---

## 2. 环境搭建

### 2.1 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv langgraph_env
source langgraph_env/bin/activate  # Windows: langgraph_env\Scripts\activate

# 安装 LangGraph
pip install langgraph

# 安装 LangChain 相关包
pip install langchain langchain-openai langchain-community

# 安装其他常用工具
pip install python-dotenv  # 环境变量管理
pip install tavily-python  # 搜索工具
```

### 2.2 配置 API Key

创建 `.env` 文件：

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 2.3 验证安装

```python
import langgraph
from langchain_openai import ChatOpenAI

print(f"LangGraph version: {langgraph.__version__}")
print("安装成功！")
```

---

## 3. 核心概念

### 3.1 StateGraph（状态图）

StateGraph 是 LangGraph 的核心数据结构，它定义了：
- **节点（Nodes）**：执行具体任务的函数
- **边（Edges）**：定义节点之间的连接关系
- **状态（State）**：在节点间传递的数据

### 3.2 State（状态）

State 是一个 TypedDict，定义了图中传递的数据结构：

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表
    user_info: str  # 用户信息
    counter: int  # 计数器
```

**Reducer 函数**：`add_messages` 是一个 reducer，用于合并新旧状态。

### 3.3 节点（Node）

节点是处理状态的函数，接收当前状态，返回状态更新：

```python
def my_node(state: State) -> State:
    # 处理逻辑
    return {"counter": state["counter"] + 1}
```

### 3.4 边（Edge）

三种类型的边：
1. **普通边**：固定的节点连接
2. **条件边**：基于状态动态选择下一个节点
3. **入口点和结束点**：定义图的起点和终点

---

## 4. 基础教程

### 4.1 第一个 LangGraph 程序

创建一个简单的计数器：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. 定义状态
class CounterState(TypedDict):
    count: int

# 2. 定义节点函数
def increment(state: CounterState) -> CounterState:
    return {"count": state["count"] + 1}

def print_count(state: CounterState) -> CounterState:
    print(f"当前计数: {state['count']}")
    return state

# 3. 创建图
workflow = StateGraph(CounterState)

# 4. 添加节点
workflow.add_node("increment", increment)
workflow.add_node("print", print_count)

# 5. 添加边
workflow.set_entry_point("increment")  # 设置入口
workflow.add_edge("increment", "print")  # increment -> print
workflow.add_edge("print", END)  # print -> 结束

# 6. 编译图
app = workflow.compile()

# 7. 运行
result = app.invoke({"count": 0})
print(f"最终结果: {result}")
```

### 4.2 带条件分支的图

实现一个简单的决策流程：

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

class DecisionState(TypedDict):
    number: int
    result: str

def check_number(state: DecisionState) -> DecisionState:
    return state

def process_even(state: DecisionState) -> DecisionState:
    return {"result": f"{state['number']} 是偶数"}

def process_odd(state: DecisionState) -> DecisionState:
    return {"result": f"{state['number']} 是奇数"}

# 条件函数：决定下一个节点
def route_number(state: DecisionState) -> Literal["even", "odd"]:
    if state["number"] % 2 == 0:
        return "even"
    return "odd"

# 构建图
workflow = StateGraph(DecisionState)
workflow.add_node("check", check_number)
workflow.add_node("even", process_even)
workflow.add_node("odd", process_odd)

workflow.set_entry_point("check")
workflow.add_conditional_edges(
    "check",
    route_number,  # 条件函数
    {
        "even": "even",  # 如果返回 "even"，跳转到 even 节点
        "odd": "odd"     # 如果返回 "odd"，跳转到 odd 节点
    }
)
workflow.add_edge("even", END)
workflow.add_edge("odd", END)

app = workflow.compile()

# 测试
print(app.invoke({"number": 4}))
print(app.invoke({"number": 7}))
```

### 4.3 循环图

实现一个重试机制：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class RetryState(TypedDict):
    attempts: int
    max_attempts: int
    success: bool

def try_task(state: RetryState) -> RetryState:
    import random
    success = random.random() > 0.7  # 30% 成功率
    print(f"尝试 {state['attempts'] + 1}: {'成功' if success else '失败'}")
    return {
        "attempts": state["attempts"] + 1,
        "success": success
    }

def should_retry(state: RetryState) -> str:
    if state["success"]:
        return "end"
    if state["attempts"] >= state["max_attempts"]:
        return "end"
    return "retry"

workflow = StateGraph(RetryState)
workflow.add_node("try", try_task)

workflow.set_entry_point("try")
workflow.add_conditional_edges(
    "try",
    should_retry,
    {
        "retry": "try",  # 循环回到 try 节点
        "end": END
    }
)

app = workflow.compile()
result = app.invoke({"attempts": 0, "max_attempts": 5, "success": False})
print(f"\n最终结果: {result}")
```

### 4.4 与 LLM 集成

创建一个简单的聊天机器人：

```python
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

# 初始化模型
model = ChatOpenAI(model="gpt-4", temperature=0.7)

def chatbot(state: ChatState) -> ChatState:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 构建图
workflow = StateGraph(ChatState)
workflow.add_node("chatbot", chatbot)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)

app = workflow.compile()

# 对话
def chat(user_input: str, history: list = None):
    if history is None:
        history = []
    
    messages = history + [HumanMessage(content=user_input)]
    result = app.invoke({"messages": messages})
    return result["messages"]

# 测试
messages = chat("你好，请介绍一下 LangGraph")
print(messages[-1].content)

messages = chat("它有什么优势？", messages)
print(messages[-1].content)
```

---

## 5. 进阶教程

### 5.1 多智能体系统

创建一个研究助手系统，包含搜索者、分析师和写作者三个智能体：

```python
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    search_results: str
    analysis: str
    final_report: str
    next_agent: str

model = ChatOpenAI(model="gpt-4", temperature=0.7)

# 智能体 1: 搜索者
def searcher(state: ResearchState) -> ResearchState:
    prompt = f"模拟搜索关于 '{state['topic']}' 的信息，返回3个要点"
    messages = [
        SystemMessage(content="你是一个专业的信息搜索助手"),
        HumanMessage(content=prompt)
    ]
    response = model.invoke(messages)
    return {
        "search_results": response.content,
        "next_agent": "analyst"
    }

# 智能体 2: 分析师
def analyst(state: ResearchState) -> ResearchState:
    prompt = f"分析以下搜索结果并提取关键洞察:\n{state['search_results']}"
    messages = [
        SystemMessage(content="你是一个专业的数据分析师"),
        HumanMessage(content=prompt)
    ]
    response = model.invoke(messages)
    return {
        "analysis": response.content,
        "next_agent": "writer"
    }

# 智能体 3: 写作者
def writer(state: ResearchState) -> ResearchState:
    prompt = f"基于以下分析写一份简短报告:\n{state['analysis']}"
    messages = [
        SystemMessage(content="你是一个专业的技术写作者"),
        HumanMessage(content=prompt)
    ]
    response = model.invoke(messages)
    return {
        "final_report": response.content,
        "next_agent": "end"
    }

# 路由函数
def route_agent(state: ResearchState) -> str:
    return state.get("next_agent", "end")

# 构建图
workflow = StateGraph(ResearchState)
workflow.add_node("searcher", searcher)
workflow.add_node("analyst", analyst)
workflow.add_node("writer", writer)

workflow.set_entry_point("searcher")
workflow.add_conditional_edges(
    "searcher",
    route_agent,
    {"analyst": "analyst"}
)
workflow.add_conditional_edges(
    "analyst",
    route_agent,
    {"writer": "writer"}
)
workflow.add_conditional_edges(
    "writer",
    route_agent,
    {"end": END}
)

app = workflow.compile()

# 运行
result = app.invoke({"topic": "量子计算的最新进展"})
print("=== 最终报告 ===")
print(result["final_report"])
```

### 5.2 Tool Calling（工具调用）

创建一个能使用工具的智能体：

```python
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

# 定义工具
@tool
def get_weather(location: str) -> str:
    """获取指定地点的天气信息"""
    # 这里模拟天气 API
    return f"{location}的天气：晴朗，温度 25°C"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except:
        return "计算错误"

tools = [get_weather, calculate]

class ToolState(TypedDict):
    messages: Annotated[list, add_messages]

# 初始化模型并绑定工具
model = ChatOpenAI(model="gpt-4", temperature=0).bind_tools(tools)

def agent(state: ToolState) -> ToolState:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: ToolState) -> str:
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    return "tools"

# 构建图
workflow = StateGraph(ToolState)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

app = workflow.compile()

# 测试
result = app.invoke({
    "messages": [HumanMessage(content="北京的天气怎么样？然后帮我计算 123 * 456")]
})

for msg in result["messages"]:
    if isinstance(msg, AIMessage):
        print(f"AI: {msg.content}")
    elif isinstance(msg, ToolMessage):
        print(f"Tool: {msg.content}")
```

### 5.3 人机交互（Human-in-the-loop）

实现需要人工审核的工作流：

```python
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    content: str
    draft: str
    approved: bool
    feedback: str

model = ChatOpenAI(model="gpt-4")

def generate_draft(state: ApprovalState) -> ApprovalState:
    prompt = f"为以下主题写一篇简短文章: {state['content']}"
    response = model.invoke([HumanMessage(content=prompt)])
    print(f"\n=== 生成的草稿 ===\n{response.content}\n")
    return {"draft": response.content}

def human_review(state: ApprovalState) -> ApprovalState:
    # 在实际应用中，这里会暂停等待人工输入
    print("请审核草稿 (输入 'approve' 批准, 或提供修改意见):")
    user_input = input("> ")
    
    if user_input.lower() == "approve":
        return {"approved": True}
    else:
        return {"approved": False, "feedback": user_input}

def revise_draft(state: ApprovalState) -> ApprovalState:
    prompt = f"根据以下反馈修改文章:\n原文: {state['draft']}\n反馈: {state['feedback']}"
    response = model.invoke([HumanMessage(content=prompt)])
    print(f"\n=== 修改后的草稿 ===\n{response.content}\n")
    return {"draft": response.content}

def should_continue(state: ApprovalState) -> str:
    return "end" if state.get("approved") else "revise"

# 构建图
workflow = StateGraph(ApprovalState)
workflow.add_node("generate", generate_draft)
workflow.add_node("review", human_review)
workflow.add_node("revise", revise_draft)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "review")
workflow.add_conditional_edges(
    "review",
    should_continue,
    {"revise": "revise", "end": END}
)
workflow.add_edge("revise", "review")

# 使用 checkpointer 支持中断和恢复
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 运行
config = {"configurable": {"thread_id": "1"}}
result = app.invoke(
    {"content": "人工智能的未来"},
    config=config
)
```

### 5.4 子图（Subgraphs）

将复杂的工作流模块化：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

# 子图状态
class SubState(TypedDict):
    value: int

# 子图节点
def sub_node_1(state: SubState) -> SubState:
    print(f"子图节点1: {state['value']}")
    return {"value": state["value"] * 2}

def sub_node_2(state: SubState) -> SubState:
    print(f"子图节点2: {state['value']}")
    return {"value": state["value"] + 10}

# 创建子图
subgraph = StateGraph(SubState)
subgraph.add_node("sub1", sub_node_1)
subgraph.add_node("sub2", sub_node_2)
subgraph.set_entry_point("sub1")
subgraph.add_edge("sub1", "sub2")
subgraph.add_edge("sub2", END)
compiled_subgraph = subgraph.compile()

# 主图状态
class MainState(TypedDict):
    value: int
    result: int

# 主图节点
def main_node_1(state: MainState) -> MainState:
    print(f"主图节点1: {state['value']}")
    return {"value": state["value"] + 5}

def use_subgraph(state: MainState) -> MainState:
    # 调用子图
    sub_result = compiled_subgraph.invoke({"value": state["value"]})
    return {"value": sub_result["value"]}

def main_node_2(state: MainState) -> MainState:
    print(f"主图节点2: {state['value']}")
    return {"result": state["value"] * 3}

# 创建主图
main_graph = StateGraph(MainState)
main_graph.add_node("main1", main_node_1)
main_graph.add_node("subgraph", use_subgraph)
main_graph.add_node("main2", main_node_2)

main_graph.set_entry_point("main1")
main_graph.add_edge("main1", "subgraph")
main_graph.add_edge("subgraph", "main2")
main_graph.add_edge("main2", END)

app = main_graph.compile()
result = app.invoke({"value": 1})
print(f"\n最终结果: {result}")
```

---

## 6. 高级特性

### 6.1 持久化与检查点

LangGraph 支持保存和恢复图的执行状态：

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from typing import TypedDict

class PersistentState(TypedDict):
    counter: int
    history: list

def increment(state: PersistentState) -> PersistentState:
    new_counter = state["counter"] + 1
    history = state.get("history", []) + [new_counter]
    return {"counter": new_counter, "history": history}

workflow = StateGraph(PersistentState)
workflow.add_node("increment", increment)
workflow.set_entry_point("increment")
workflow.add_edge("increment", END)

# 使用 SQLite 作为持久化存储
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)
    
    # 首次运行
    config = {"configurable": {"thread_id": "thread-1"}}
    result1 = app.invoke({"counter": 0, "history": []}, config)
    print(f"第一次运行: {result1}")
    
    # 继续之前的对话
    result2 = app.invoke({"counter": result1["counter"], "history": result1["history"]}, config)
    print(f"第二次运行: {result2}")
    
    # 查看历史
    for state in app.get_state_history(config):
        print(f"历史状态: {state}")
```

### 6.2 流式输出

实时获取图的执行过程：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, add_messages
from typing import TypedDict, Annotated

class StreamState(TypedDict):
    messages: Annotated[list, add_messages]

model = ChatOpenAI(model="gpt-4", streaming=True)

def chat_node(state: StreamState) -> StreamState:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(StreamState)
workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)

app = workflow.compile()

# 流式输出
print("流式输出开始:")
for chunk in app.stream({"messages": [HumanMessage(content="写一首关于编程的诗")]}):
    print(chunk)
    print("---")
```

### 6.3 并行执行

LangGraph 支持并行执行多个节点：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
import time

class ParallelState(TypedDict):
    task1_result: str
    task2_result: str
    task3_result: str

def task_1(state: ParallelState) -> ParallelState:
    time.sleep(1)
    return {"task1_result": "任务1完成"}

def task_2(state: ParallelState) -> ParallelState:
    time.sleep(1)
    return {"task2_result": "任务2完成"}

def task_3(state: ParallelState) -> ParallelState:
    time.sleep(1)
    return {"task3_result": "任务3完成"}

def aggregate(state: ParallelState) -> ParallelState:
    print(f"汇总结果: {state}")
    return state

workflow = StateGraph(ParallelState)
workflow.add_node("task1", task_1)
workflow.add_node("task2", task_2)
workflow.add_node("task3", task_3)
workflow.add_node("aggregate", aggregate)

# 并行入口
workflow.set_entry_point("task1")
workflow.set_entry_point("task2")
workflow.set_entry_point("task3")

# 所有任务完成后汇总
workflow.add_edge("task1", "aggregate")
workflow.add_edge("task2", "aggregate")
workflow.add_edge("task3", "aggregate")
workflow.add_edge("aggregate", END)

app = workflow.compile()

start = time.time()
result = app.invoke({})
end = time.time()

print(f"\n执行时间: {end - start:.2f}秒 (并行执行)")
```

### 6.4 动态图构建

根据运行时状态动态添加节点：

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class DynamicState(TypedDict):
    tasks: List[str]
    results: List[str]
    current_index: int

def process_task(state: DynamicState) -> DynamicState:
    current = state["current_index"]
    task = state["tasks"][current]
    result = f"处理了任务: {task}"
    
    return {
        "results": state["results"] + [result],
        "current_index": current + 1
    }

def should_continue(state: DynamicState) -> str:
    if state["current_index"] < len(state["tasks"]):
        return "process"
    return "end"

workflow = StateGraph(DynamicState)
workflow.add_node("process", process_task)

workflow.set_entry_point("process")
workflow.add_conditional_edges(
    "process",
    should_continue,
    {
        "process": "process",  # 循环处理
        "end": END
    }
)

app = workflow.compile()

result = app.invoke({
    "tasks": ["任务A", "任务B", "任务C"],
    "results": [],
    "current_index": 0
})

print("处理结果:", result["results"])
```

### 6.5 错误处理与重试

实现健壮的错误处理机制：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
import random

class ErrorState(TypedDict):
    attempt: int
    max_attempts: int
    error: str
    success: bool
    result: str

def risky_operation(state: ErrorState) -> ErrorState:
    try:
        # 模拟可能失败的操作
        if random.random() < 0.6:  # 60% 失败率
            raise Exception("操作失败")
        
        return {
            "success": True,
            "result": "操作成功完成"
        }
    except Exception as e:
        return {
            "attempt": state["attempt"] + 1,
            "error": str(e),
            "success": False
        }

def handle_error(state: ErrorState) -> ErrorState:
    print(f"尝试 {state['attempt']}/{state['max_attempts']} 失败: {state['error']}")
    return state

def route_after_error(state: ErrorState) -> str:
    if state["success"]:
        return "end"
    if state["attempt"] >= state["max_attempts"]:
        return "final_failure"
    return "retry"

def final_failure(state: ErrorState) -> ErrorState:
    return {"result": "所有重试均失败"}

workflow = StateGraph(ErrorState)
workflow.add_node("operation", risky_operation)
workflow.add_node("error_handler", handle_error)
workflow.add_node("final_failure", final_failure)

workflow.set_entry_point("operation")
workflow.add_conditional_edges(
    "operation",
    route_after_error,
    {
        "retry": "error_handler",
        "final_failure": "final_failure",
        "end": END
    }
)
workflow.add_edge("error_handler", "operation")
workflow.add_edge("final_failure", END)

app = workflow.compile()

result = app.invoke({
    "attempt": 0,
    "max_attempts": 5,
    "error": "",
    "success": False,
    "result": ""
})

print(f"\n最终状态: {result}")
```

---

## 7. 实战项目

### 7.1 项目1：智能客服系统

完整的多意图客服机器人：

```python
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages

class CustomerServiceState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str  # order_query, complaint, general
    user_id: str
    resolved: bool

model = ChatOpenAI
# 意图识别
def classify_intent(state: CustomerServiceState) -> CustomerServiceState:
    last_message = state["messages"][-1].content
    prompt = f"""
    分析用户消息的意图，只返回以下之一：
    - order_query: 查询订单
    - complaint: 投诉
    - general: 一般咨询
    
    用户消息: {last_message}
    只返回意图类型，不要其他内容。
    """
    response = model.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().lower()
    print(f"识别意图: {intent}")
    return {"intent": intent}

# 订单查询处理
def handle_order_query(state: CustomerServiceState) -> CustomerServiceState:
    # 模拟订单查询
    response = f"您的订单 (用户ID: {state['user_id']}) 状态为：已发货，预计3天内送达。"
    return {
        "messages": [SystemMessage(content=response)],
        "resolved": True
    }

# 投诉处理
def handle_complaint(state: CustomerServiceState) -> CustomerServiceState:
    last_message = state["messages"][-1].content
    prompt = f"""
    你是一个专业的客服人员。用户提出了投诉：{last_message}
    请给出同情且专业的回应，并提供解决方案。
    """
    response = model.invoke([
        SystemMessage(content="你是专业的客服人员"),
        HumanMessage(content=prompt)
    ])
    return {
        "messages": [response],
        "resolved": True
    }

# 一般咨询处理
def handle_general(state: CustomerServiceState) -> CustomerServiceState:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def check_if_resolved(state: CustomerServiceState) -> CustomerServiceState:
    # 询问用户是否解决
    return {
        "messages": [SystemMessage(content="您的问题解决了吗？")]
    }

# 路由函数
def route_by_intent(state: CustomerServiceState) -> str:
    intent_map = {
        "order_query": "order",
        "complaint": "complaint",
        "general": "general"
    }
    return intent_map.get(state["intent"], "general")

# 构建图
workflow = StateGraph(CustomerServiceState)
workflow.add_node("classify", classify_intent)
workflow.add_node("order", handle_order_query)
workflow.add_node("complaint", handle_complaint)
workflow.add_node("general", handle_general)
workflow.add_node("check", check_if_resolved)

workflow.set_entry_point("classify")
workflow.add_conditional_edges(
    "classify",
    route_by_intent,
    {
        "order": "order",
        "complaint": "complaint",
        "general": "general"
    }
)
workflow.add_edge("order", "check")
workflow.add_edge("complaint", "check")
workflow.add_edge("general", "check")
workflow.add_edge("check", END)

customer_service_app = workflow.compile()

# 测试
print("=== 测试1: 订单查询 ===")
result = customer_service_app.invoke({
    "messages": [HumanMessage(content="我想查一下我的订单状态")],
    "user_id": "USER123"
})

print("=== 测试2: 投诉 ===")
result = customer_service_app.invoke({
    "messages": [HumanMessage(content="我收到的商品有质量问题，很不满意")],
    "user_id": "USER456"
})
```

### 7.2 项目2：代码助手

支持代码生成、审查、测试的完整流程：

```python
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages

class CodeAssistantState(TypedDict):
    messages: Annotated[list, add_messages]
    requirement: str
    code: str
    review_feedback: str
    test_result: str
    iteration: int
    max_iterations: int

model = ChatOpenAI(model="gpt-4", temperature=0.3)

# 代码生成
def generate_code(state: CodeAssistantState) -> CodeAssistantState:
    if state["iteration"] == 0:
        prompt = f"根据需求生成Python代码：{state['requirement']}"
    else:
        prompt = f"""
        根据审查反馈改进代码：
        原代码：{state['code']}
        反馈：{state['review_feedback']}
        """
    
    response = model.invoke([
        SystemMessage(content="你是一个专业的Python开发者"),
        HumanMessage(content=prompt)
    ])
    
    code = response.content
    print(f"\n=== 生成的代码 (迭代 {state['iteration'] + 1}) ===")
    print(code)
    
    return {
        "code": code,
        "iteration": state["iteration"] + 1
    }

# 代码审查
def review_code(state: CodeAssistantState) -> CodeAssistantState:
    prompt = f"""
    审查以下代码，检查：
    1. 代码质量
    2. 潜在bug
    3. 性能问题
    4. 最佳实践
    
    代码：
    {state['code']}
    
    如果代码很好，返回"APPROVED"
    否则，提供具体的改进建议。
    """
    
    response = model.invoke([
        SystemMessage(content="你是一个严格的代码审查员"),
        HumanMessage(content=prompt)
    ])
    
    feedback = response.content
    print(f"\n=== 审查反馈 ===")
    print(feedback)
    
    return {"review_feedback": feedback}

# 生成测试
def generate_tests(state: CodeAssistantState) -> CodeAssistantState:
    prompt = f"""
    为以下代码生成单元测试：
    {state['code']}
    """
    
    response = model.invoke([
        SystemMessage(content="你是测试专家"),
        HumanMessage(content=prompt)
    ])
    
    print(f"\n=== 生成的测试 ===")
    print(response.content)
    
    return {"test_result": "测试已生成"}

# 路由决策
def should_continue(state: CodeAssistantState) -> str:
    if "APPROVED" in state["review_feedback"]:
        return "test"
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    return "regenerate"

# 构建图
workflow = StateGraph(CodeAssistantState)
workflow.add_node("generate", generate_code)
workflow.add_node("review", review_code)
workflow.add_node("test", generate_tests)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "review")
workflow.add_conditional_edges(
    "review",
    should_continue,
    {
        "regenerate": "generate",
        "test": "test",
        "end": END
    }
)
workflow.add_edge("test", END)

code_assistant_app = workflow.compile()

# 测试
result = code_assistant_app.invoke({
    "requirement": "实现一个二分查找函数",
    "code": "",
    "review_feedback": "",
    "test_result": "",
    "iteration": 0,
    "max_iterations": 3
})
```

### 7.3 项目3：研究论文分析器

自动分析论文并生成摘要：

```python
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

class PaperAnalysisState(TypedDict):
    paper_url: str
    paper_text: str
    sections: List[str]
    key_findings: str
    methodology: str
    limitations: str
    summary: str

model = ChatOpenAI(model="gpt-4", temperature=0.2)

def extract_sections(state: PaperAnalysisState) -> PaperAnalysisState:
    # 模拟提取论文章节
    sections = [
        "Introduction: 介绍研究背景...",
        "Methodology: 描述研究方法...",
        "Results: 展示实验结果...",
        "Conclusion: 总结研究发现..."
    ]
    return {"sections": sections}

def analyze_findings(state: PaperAnalysisState) -> PaperAnalysisState:
    prompt = f"""
    分析论文的关键发现：
    {' '.join(state['sections'])}
    
    提取3-5个最重要的发现。
    """
    response = model.invoke([HumanMessage(content=prompt)])
    return {"key_findings": response.content}

def analyze_methodology(state: PaperAnalysisState) -> PaperAnalysisState:
    prompt = f"总结论文使用的研究方法：{state['sections'][1]}"
    response = model.invoke([HumanMessage(content=prompt)])
    return {"methodology": response.content}

def identify_limitations(state: PaperAnalysisState) -> PaperAnalysisState:
    prompt = f"""
    识别论文的局限性：
    {' '.join(state['sections'])}
    """
    response = model.invoke([HumanMessage(content=prompt)])
    return {"limitations": response.content}

def generate_summary(state: PaperAnalysisState) -> PaperAnalysisState:
    prompt = f"""
    生成论文的综合摘要：
    
    关键发现：{state['key_findings']}
    研究方法：{state['methodology']}
    局限性：{state['limitations']}
    
    生成一个200字的摘要。
    """
    response = model.invoke([HumanMessage(content=prompt)])
    print("\n=== 论文摘要 ===")
    print(response.content)
    return {"summary": response.content}

# 构建图
workflow = StateGraph(PaperAnalysisState)
workflow.add_node("extract", extract_sections)
workflow.add_node("findings", analyze_findings)
workflow.add_node("methodology", analyze_methodology)
workflow.add_node("limitations", identify_limitations)
workflow.add_node("summary", generate_summary)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "findings")
workflow.add_edge("extract", "methodology")
workflow.add_edge("extract", "limitations")
workflow.add_edge("findings", "summary")
workflow.add_edge("methodology", "summary")
workflow.add_edge("limitations", "summary")
workflow.add_edge("summary", END)

paper_analyzer_app = workflow.compile()

# 测试
result = paper_analyzer_app.invoke({
    "paper_url": "https://arxiv.org/paper/example",
    "paper_text": "论文全文..."
})
```

### 7.4 项目4：自动化工作流系统

处理复杂业务流程：

```python
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from datetime import datetime

class WorkflowState(TypedDict):
    task_id: str
    task_type: str  # approval, notification, processing
    priority: int  # 1-5
    assignee: str
    status: str
    history: List[str]
    result: str

model = ChatOpenAI(model="gpt-4")

def classify_task(state: WorkflowState) -> WorkflowState:
    # 根据任务内容分类
    print(f"分类任务: {state['task_id']}")
    return state

def assign_task(state: WorkflowState) -> WorkflowState:
    # 根据优先级和类型分配任务
    assignee_map = {
        "approval": "经理",
        "notification": "系统",
        "processing": "处理团队"
    }
    assignee = assignee_map.get(state["task_type"], "默认处理人")
    history = state.get("history", [])
    history.append(f"{datetime.now()}: 任务分配给 {assignee}")
    
    return {
        "assignee": assignee,
        "status": "assigned",
        "history": history
    }

def process_task(state: WorkflowState) -> WorkflowState:
    print(f"处理任务: {state['task_id']}, 类型: {state['task_type']}")
    history = state["history"]
    history.append(f"{datetime.now()}: 任务处理中")
    
    # 模拟处理
    result = f"任务 {state['task_id']} 已由 {state['assignee']} 完成"
    
    return {
        "status": "completed",
        "result": result,
        "history": history
    }

def send_notification(state: WorkflowState) -> WorkflowState:
    print(f"发送通知: {state['result']}")
    history = state["history"]
    history.append(f"{datetime.now()}: 通知已发送")
    return {"history": history}

def route_by_priority(state: WorkflowState) -> str:
    if state["priority"] >= 4:
        return "urgent"
    return "normal"

# 构建图
workflow = StateGraph(WorkflowState)
workflow.add_node("classify", classify_task)
workflow.add_node("assign", assign_task)
workflow.add_node("process", process_task)
workflow.add_node("notify", send_notification)

workflow.set_entry_point("classify")
workflow.add_edge("classify", "assign")
workflow.add_conditional_edges(
    "assign",
    route_by_priority,
    {
        "urgent": "process",
        "normal": "process"
    }
)
workflow.add_edge("process", "notify")
workflow.add_edge("notify", END)

workflow_app = workflow.compile()

# 测试
result = workflow_app.invoke({
    "task_id": "TASK-001",
    "task_type": "approval",
    "priority": 5,
    "history": []
})

print("\n=== 工作流历史 ===")
for entry in result["history"]:
    print(entry)
print(f"\n最终结果: {result['result']}")
```

---

## 8. 最佳实践

### 8.1 状态设计原则

**1. 保持状态最小化**
```python
# ❌ 不好：包含冗余数据
class BadState(TypedDict):
    user_input: str
    processed_input: str
    lower_case_input: str  # 冗余
    upper_case_input: str  # 冗余

# ✅ 好：只保存必要数据
class GoodState(TypedDict):
    user_input: str
    processed_input: str
```

**2. 使用类型注解**
```python
from typing import TypedDict, List, Optional

class WellTypedState(TypedDict):
    counter: int
    items: List[str]
    error: Optional[str]
```

**3. 使用 Reducer 函数**
```python
from langgraph.graph import add_messages
from typing import Annotated

class MessageState(TypedDict):
    messages: Annotated[list, add_messages]  # 自动合并消息
```

### 8.2 节点设计原则

**1. 单一职责**
```python
# ❌ 不好：节点做太多事情
def bad_node(state):
    # 验证输入
    # 调用API
    # 处理结果
    # 记录日志
    # 发送通知
    pass

# ✅ 好：每个节点只做一件事
def validate_input(state): ...
def call_api(state): ...
def process_result(state): ...
def log_activity(state): ...
def send_notification(state): ...
```

**2. 幂等性**
```python
# ✅ 节点应该是幂等的
def idempotent_node(state: State) -> State:
    # 多次执行产生相同结果
    return {"processed": True}
```

**3. 错误处理**
```python
def robust_node(state: State) -> State:
    try:
        result = risky_operation()
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}
```

### 8.3 图结构设计

**1. 清晰的入口和出口**
```python
workflow.set_entry_point("start")
workflow.add_edge("final_node", END)
```

**2. 避免过深的嵌套**
```python
# ❌ 不好：过于复杂
workflow.add_conditional_edges(
    "node1",
    lambda s: "a" if s["x"] > 0 else "b" if s["y"] > 0 else "c",
    {"a": "node2", "b": "node3", "c": "node4"}
)

# ✅ 好：使用中间节点简化
def route_logic(state):
    if state["x"] > 0:
        return "path_a"
    if state["y"] > 0:
        return "path_b"
    return "path_c"

workflow.add_conditional_edges("node1", route_logic, {...})
```

**3. 使用子图模块化**
```python
# 将相关节点组织成子图
data_processing_subgraph = create_data_processing_graph()
main_graph.add_node("data_processing", data_processing_subgraph)
```

### 8.4 可观测性

**1. 添加日志**
```python
import logging

logger = logging.getLogger(__name__)

def logged_node(state: State) -> State:
    logger.info(f"Processing state: {state}")
    result = process(state)
    logger.info(f"Result: {result}")
    return result
```

**2. 追踪执行路径**
```python
class TrackedState(TypedDict):
    execution_path: List[str]
    # ... 其他字段

def tracked_node(state: TrackedState) -> TrackedState:
    path = state.get("execution_path", [])
    path.append("tracked_node")
    return {"execution_path": path}
```

**3. 使用检查点调试**
```python
# 保存每一步的状态
app = workflow.compile(checkpointer=memory)

# 检查执行历史
for state in app.get_state_history(config):
    print(f"Step: {state}")
```

### 8.5 性能优化

**1. 批处理**
```python
def batch_process(state: State) -> State:
    items = state["items"]
    # 批量处理而不是逐个处理
    results = model.batch([items])
    return {"results": results}
```

**2. 缓存结果**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_operation(input_data: str) -> str:
    # 缓存昂贵的操作
    return process(input_data)
```

**3. 异步执行**
```python
async def async_node(state: State) -> State:
    result = await async_api_call()
    return {"result": result}
```

---

## 9. 性能优化

### 9.1 减少 API 调用

```python
# ❌ 不好：每次都调用API
def bad_approach(state):
    for item in state["items"]:
        result = model.invoke(item)  # 多次调用

# ✅ 好：批量调用
def good_approach(state):
    results = model.batch(state["items"])  # 一次调用
```

### 9.2 并行执行

```python
from langgraph.graph import StateGraph
import asyncio

# 并行执行独立任务
async def parallel_tasks(state):
    task1 = asyncio.create_task(async_operation_1())
    task2 = asyncio.create_task(async_operation_2())
    result1, result2 = await asyncio.gather(task1, task2)
    return {"result1": result1, "result2": result2}
```

### 9.3 流式输出优化

```python
# 使用流式输出提升用户体验
for chunk in app.stream(input_data):
    # 实时显示进度
    display(chunk)
```

### 9.4 内存管理

```python
# 定期清理大对象
def cleanup_node(state: State) -> State:
    # 处理完后删除大对象
    result = state["large_data"][:10]  # 只保留需要的部分
    return {"summary": result}
```

---

## 10. 常见问题

### Q1: 如何处理循环依赖？

使用条件边控制循环退出：

```python
def should_continue(state):
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"
    if state["converged"]:
        return "end"
    return "continue"

workflow.add_conditional_edges(
    "process",
    should_continue,
    {"continue": "process", "end": END}
)
```

### Q2: 如何调试图执行？

```python
# 1. 打印中间状态
def debug_node(state):
    print(f"当前状态: {state}")
    return state

# 2. 使用 breakpoint
def node_with_breakpoint(state):
    breakpoint()  # 程序会在这里暂停
    return process(state)

# 3. 查看执行历史
for state in app.get_state_history(config):
    print(state)
```

### Q3: 如何处理长时间运行的任务？

```python
# 使用检查点持久化
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("db.sqlite") as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)
    
    # 可以随时中断和恢复
    result = app.invoke(input_data, config)
```

### Q4: 如何实现超时控制？

```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("节点执行超时")

def node_with_timeout(state):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30秒超时
    
    try:
        result = long_running_operation()
        signal.alarm(0)  # 取消超时
        return {"result": result}
    except TimeoutError:
        return {"error": "操作超时"}
```

### Q5: 如何处理不同版本的图？

```python
# 使用版本号管理
class VersionedState(TypedDict):
    version: str
    data: dict

def migrate_v1_to_v2(state):
    if state["version"] == "v1":
        # 迁移逻辑
        return {"version": "v2", "data": migrated_data}
    return state
```

---

## 附录 A：完整示例代码

### 端到端聊天机器人

```python
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

class ChatbotState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str

model = ChatOpenAI(model="gpt-4")

def chatbot_node(state: ChatbotState) -> ChatbotState:
    system_msg = SystemMessage(content=f"上下文: {state.get('context', '')}")
    messages = [system_msg] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(ChatbotState)
workflow.add_node("chat", chatbot_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 使用
config = {"configurable": {"thread_id": "user-123"}}

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    
    print(f"Bot: {result['messages'][-1].content}")
```

---

## 附录 B：有用的资源

### 官方资源
- LangGraph 文档: https://langchain-ai.github.io/langgraph/
- LangChain 文档: https://python.langchain.com/
- GitHub 仓库: https://github.com/langchain-ai/langgraph

### 学习资源
- LangGraph 教程视频
- LangChain 博客文章
- 社区示例代码

### 工具和库
- LangSmith: 调试和监控
- LangServe: 部署 LangChain 应用
- Tavily: 网络搜索工具

---

## 结语

LangGraph 是构建复杂 AI 应用的强大工具。通过本教程，你应该已经掌握了：

1. ✅ LangGraph 的核心概念和基本用法
2. ✅ 如何设计和实现复杂的工作流
3. ✅ 多智能体系统的构建方法
4. ✅ 实战项目的开发技巧
5. ✅ 性能优化和最佳实践

**下一步建议：**
- 动手实现本教程中的所有示例
- 根据自己的需求设计新的工作流
- 参与 LangGraph 社区讨论
- 关注 LangGraph 的最新更新

祝你在 LangGraph 的学习和应用中取得成功！🚀(model="gpt-4", temperature=0.7)

def chatbot(state: ChatState) -> ChatState:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 构建图
workflow = StateGraph(ChatState)
workflow.add_node("chatbot", chatbot)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)

app = workflow.compile()

# 对话
def chat(user_input: str, history: list = None):
    if history is None:
        history = []
    
    messages = history + [HumanMessage(content=user_input)]
    result = app.invoke({"messages": messages})
    return result["messages"]

# 测试
messages = chat("你好，请介绍一下 LangGraph")
print(messages[-1].content)

messages = chat("它有什么优势？", messages)
print(messages[-1].content)
```

---
