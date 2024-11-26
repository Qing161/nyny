FROM python:3.8.5

WORKDIR /app

COPY . .

RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip3 config set global.extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --upgrade pip
# 更新Conda
RUN pip3 install -r requirements.txt
CMD ["python", "run.py"]