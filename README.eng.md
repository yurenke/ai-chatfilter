# AI-Chatting-Filter

## Installation Requirements for System (Centos ver)

### 1. python3.7 / pip3 and some dependencies
> for Redhat Linux (centos)

Install Python3.7 and pip3

```Shell
sudo yum update
sudo yum -y install zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gcc make libffi-devel
sudo yum -y install epel-release
sudo yum -y install python-pip
sudo pip install wget

wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
tar -zxvf Python-3.7.5.tgz
cd Python-3.7.5
./configure prefix=/usr/local/python3
sudo make install

sudo ln -s /usr/local/python3/bin/python3.7 /usr/bin/python3
sudo ln -s /usr/local/python3/bin/pip3.7 /usr/bin/pip3
python3 -V

sudo yum -y install python-devel python3-devel python-Levenshtein
```


### 2. postgresql 10.11.x
> for Redhat Linux (centos)
```Shell
sudo rpm -Uvh https://yum.postgresql.org/10/redhat/rhel-7-x86_64/pgdg-centos10-10-2.noarch.rpm
sudo yum -y install postgresql10-server postgresql10
sudo yum -y install postgresql-contrib postgresql-libs
sudo /usr/pgsql-10/bin/postgresql-10-setup initdb

systemctl start postgresql-10.service
systemctl enable postgresql-10.service
```

> create a new database and setting db users
```Shell
sudo su - postgres -c "psql"
\conninfo
\password postgres
CREATE DATABASE [name of database];
\q
```

> finally make sure pg_hba.conf is trust all localhost
```SQL
postgres=# show_hba_file;
```


### 3. redis server
> for Redhat Linux (centos)
```Shell
sudo yum -y install epel-release yum-utils
sudo yum -y install http://rpms.remirepo.net/enterprise/remi-release-7.rpm
sudo yum-config-manager --enable remi
sudo yum -y install redis
sudo systemctl start redis
sudo systemctl enable redis
sudo systemctl status redis
```


### 4. nginx
> for Redhat Linux (centos)
```Shell
sudo yum -y install nginx
sudo systemctl start nginx
```

> for both linux sysyem, allow 80 and 443 port in firewall
```Shell
sudo firewall-cmd --permanent --zone=public --add-service=http 
sudo firewall-cmd --permanent --zone=public --add-service=https
sudo firewall-cmd --reload
```


### 5. virtualenv
> for Redhat Linux (centos)
```Shell
sudo yum -y install python-virtualenv
```


### 6. tensorflow 2.0+

> make sure pip version > 19.0.x
```Shell
sudo pip -V
```


### 7. wsgi
> for both Linux system
```Shell
sudo pip install uwsgi
```



## Project installation Steps


### 1. prepare the project
> make project folder and clone the project.
**for example the project name is "ai-opt"**:

+ 1.1. Clone the project
```Shell
mkdir /ai-opt/chatfilter
cd /ai-opt/chatfilter
git clone ...
cd /ai-opt/chatfilter/main
```

+ 1.2. Setting nginx config
> copy and chang your network setting in nginx.conf file
```Shell
cp nginx.conf.example nginx.conf
```

>  make symbolic link to niginx configs
```Shell
sudo ln -s /ai-opt/chatfilter/main/nginx.conf /etc/nginx/sites-enabled/
or
sudo ln -s /ai-opt/chatfilter/main/nginx.conf /etc/nginx/conf.d/
```

+ 1.3. Setting Project config
> copy setting.ini and chagne config you need
```Shell
cp setting.ini.example setting.ini
nano setting.ini
```

> change "DATABASE" setting and "LANGUAGE_MODE"
```EditorConfig
[MAIN]
LANGUAGE_MODE = EN

[DATABASE]
DATABASE_NAME = DB_NAME
DATABASE_USER = DB_USER_NAME
DATABASE_PASSWORD = DB_PASSWORD
```

+ 1.4. Create logs directory in project and make sure the logs folder changeable for supervisor(python)
```Shell
mkdir /ai-opt/logs
chmod -R 777 /ai-opt/logs
```


### 2. build up virtual environment
> Create virtual environment named venv:
```Shell
cd /ai-opt
python3 -m venv venv
chmod -R 777 venv
source venv/bin/activate
python -V
pip -V
```
> should be seen the python version at least with 3.7.5 and pip is 19+


### 3. install tensorflow 2.0 - lastest
> before doing this you've make sure you already got "venv" environment
> install what python's need in "venv"
```Shell
pip install tf-nightly
pip install tensorflow_datasets
```


### 4. install python librarys
```Shell
pip install --upgrade pip
pip install -r requirement.txt
pip install psycopg2-binary
pip install websocket
pip install websocket-client
pip install zhconv
pip install xlwt
pip install django-import-export
pip install django-rq
pip install grpcio
pip install grpcio-tools
pip install --upgrade protobuf
```


### 5. do django framework initialize
> build up the database instruction
```Shell
python manage.py migrate
python manage.py loaddata service/seed/initial.json
python manage.py loaddata ai/json/knowledge.json
```

> create django admin superuser with following the guiding steps to finish
```Shell
python manage.py createsuperuser
```
..fill all the form

> collect and copy the static file in project to improve performance
```Shell
python manage.py collectstatic
```


### 6. training ai
> before you train you may need to check your vocabulary dictionary
```Shell
python manage.py knowledge -i ai/assets/english/dict.xls -lan EN -f 3
```

> start training
```Shell
python manage.py train -i ai/assets/textbook/json/english/2020-09-08.json -eng -t 1
```


### 7. firewall setting
> open tcp port for chatting socket if need
```Shell
sudo firewall-cmd --permanent --zone=public --add-port=8025/tcp
```




## For linux product deploy using supervisor
setting supervisor <http://supervisord.org/configuration.html>
(if you are in sourced on venv you have to deactivate it)
```Shell
deactivate
```

> for Redhat Linux (centos)
```Shell
sudo yum -y install supervisor

sudo systemctl start supervisord
sudo systemctl enable supervisord
sudo systemctl status supervisord
```

> copy and edit config
```Shell
cp supervisor.conf.example supervisor.conf
nano superviosr.conf
```
+ > change all directory `/opt/` to your project's folder
```EditorConfig
directory=/ai-opt/chatfilter/main
```
+ > change all `/opt/venv/bin/python` to your virtual environment python
```EditorConfig
command=/ai-opt/venv/bin/python tcpsocket/main.py -p 8025
```
+ > change all `/opt/logs/` to your logs folder
```EditorConfig
stdout_logfile=/ai-opt/logs/tcpsocket.log
```

> symbolic link to supervisor config
```Shell
sudo ln -s /ai-opt/chatfilter/main/supervisor.conf /etc/supervisord.d/ai-chatfilter-service.ini
```

> reload supervisor
```Shell
sudo supervisorctl reload
sudo supervisorctl reread
sudo supervisorctl update
```

> reload nginx
```Shell
sudo systemctl reload nginx
sudo systemctl restart nginx
```


## Others

> Check the SELinux and add policy to nginx or just disable it
```Shell
sestatus
```

*SELinux might block the socket connection between nginx and supervisord*



## Testing

### 1. website
> Test the django web site is working, type domain:port on browser for example: `http://127.0.0.1:8000/` you should see the screen with 404 not found page but has content like below
```
Using the URLconf defined in service.urls, Django tried these URL patterns, in this order:

chat/
admin/
auth/
auth/
The empty path didn't match any of these.
```

> that means the website is working fine and next we change url to `http://127.0.0.1:8000/chat/`, try typeing something to test websockets

### 2. tcp socket
> use tcpsocket client to test chatting binary packages.
```Shell
cd /ai-opt/chatfilter/main
python tcpsocket/client.py -h 127.0.0.1 -p 8025
```
> you will see
```
Please choose package type:
1.hearting
2.login
3.login response
4.chatting
5.chat response
Enter number:
```
*everything is fine!!*


### 3. command line
```shell
python manage.py predict -t [speaksome..] -s
python manage.py predict -i [ai/assets/..] -s
```

```shell
python manage.py testsocket -i [ai/assets/..] -p 8025 -h 127.0.0.1
```


## Maintaining
> some commod tips
> dump and restore blockword data
> do not use until you know all about database
```Shell
python manage.py dumpdata service > service/seed/initial.json

python manage.py loaddata service/seed/initial.json

python manage.py upsert -i ../excel file -model textbook

python manage.py parsexcel -i ai/assets/textbook/pinyin
python manage.py parsexcel -i ai/assets/textbook/grammar

python manage.py freq -i ai/assets/textbook/json/pinyin

python manage.py backupdatabase

```


## DOCKER (not finish)

```Shell
docker pull django
docker pull postgres
docker pull redis
```


### gRPC maintaining syntax

python -m grpc_tools.protoc -I./grpcservice/protos --python_out=./grpcservice/pb --grpc_python_out=./grpcservice/pb ./grpcservice/protos/learn.proto