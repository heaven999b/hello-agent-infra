PY ?= .venv/bin/python

.PHONY: setup test test-solutions module lab check-links

setup:            ## 创建虚拟环境并安装依赖（只需要 numpy + pytest，M1/8GB 笔记本即可）
	python3 -m venv .venv
	.venv/bin/pip install -q -e ".[dev]"
	@echo "✅ 安装完成。试试：make lab N=01"

test:             ## 跑模拟器测试 + 你的练习（练习没写完会失败，这是正常的）
	$(PY) -m pytest

test-solutions:   ## 用参考答案跑所有练习测试（CI 用）
	INFRA_SOLUTION=1 $(PY) -m pytest

module:           ## 跑某个模块的练习测试：make module N=05
	$(PY) -m pytest modules/$(N)_*

lab:              ## 跑某个模块的实验：make lab N=05
	$(PY) modules/$(N)_*/lab.py

check-links:      ## 检查所有 Markdown 的相对链接与锚点
	$(PY) scripts/check_links.py
