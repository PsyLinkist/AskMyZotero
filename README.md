# 完成基本交互脚本
现在是直接调用底层的indexer.py进行检索测试功能后续与agent进行对接

##更新明细

- 新增**config.yaml配置文件**方便修改测试信息，比如api和pdf文件地址，同步修改了config.py中的resolve_config函数用于读取配置文件
- 新增**server.py**脚本用于前后端交互，目前直接调用底层功能indexer.py中的函数进行测试
- 新增indexer.py中**api_ask_once函数**用于一次性获取回答
- 新增**schemas.py**脚本进行前后端的数据结构设定和检查，ps：除了目前仅有问询和回答，其他根据前端渲染文献卡片的信息设置，但后端还并未同步得到这些信息的功能。
- 修改zotero_rag_ui.html中的runSearch函数，新增**renderResults函数**用于展示回答
- 测试方法：
- - 配置好config.yaml
  - 运行server.py脚本
  - 进入网页输入在你的文件里面能搜到的问题

- 测试结果：
  <img width="482" height="411" alt="image" src="https://github.com/user-attachments/assets/9803f85b-0455-4ef4-825f-17fe25c0a756" />
