# 运行环境
整个项目用uv做依赖管理，虚拟环境在根目录.venv中，项目后续的安装依赖包、运行代码都必须遵循如下准则
- 首选用命令`source .venv/bin/activate`初始化python虚拟环境
- 其次用uv工具安装依赖包，比如`uv add langchain chromadb`
- 最后用uv工具运行代码，比如`uv run main.py`