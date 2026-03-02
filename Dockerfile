# 用 Python 精简镜像（体积小，部署快）
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存加速
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# ClawCloud Run 会自动分配端口到 PORT 环境变量
# 默认 8000
ENV PORT=8000

# 启动网关
CMD ["python", "main.py"]
