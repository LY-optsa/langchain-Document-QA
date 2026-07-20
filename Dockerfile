# 使用Python 3.12作为基础镜像
FROM harbor-hw.gw-greenenergy.com/infra/python:3.12-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENAI_API_KEY="sk-4X5IqFBEGH5GdHyp0f79649b3a8a4c7fB0957a95B45c4b31" \
    PYTHONPATH=/app

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .


# 安装Python依赖
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade setuptools wheel
#RUN pip install  -r requirements.txt -i http://10.111.82.171:8081/repository/pypi-group/simple --trusted-host 10.111.82.171
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt


# 复制项目代码
COPY . .
# 暴露端口
EXPOSE 8000
# 启动命令
CMD ["python", "fastapi_application.py"]

