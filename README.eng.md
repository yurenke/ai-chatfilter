# AI-Chat-Filter (EN/Multi-lingual)

## Installation Requirements for System (Centos ver)

### 1. python3.7 / pip3 and some dependencies (AI VMs)

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


### 2. postgresql 10.11.x (DB VMs)
> for master and slave DB VMs  
may be different machines from the AI VMs
```Shell
sudo rpm -Uvh https://yum.postgresql.org/10/redhat/rhel-7-x86_64/pgdg-centos10-10-2.noarch.rpm
sudo yum -y install postgresql10-server postgresql10
sudo yum -y install postgresql-contrib postgresql-libs
sudo /usr/pgsql-10/bin/postgresql-10-setup initdb

sudo ln -s /usr/pgsql-10/bin/psql /usr/bin/psql --force
sudo ln -s /usr/pgsql-10/bin/pg_dump /usr/bin/pg_dump --force
sudo ln -s /usr/pgsql-10/bin/pg_restore /usr/bin/pg_restore --force
```

> modify /var/lib/pgsql/10/data/postgresql.conf

```Shell
listen_addresses = '*'
```

> add both AI VMs and master/slave DB VMs entries into /var/lib/pgsql/10/data/pg_hba.conf

```SQL
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# "local" is for Unix domain socket connections only
local   all             all                                     trust
# IPv4 local connections:
host    all             all             127.0.0.1/32            trust
# add VM host addresse entries here
host    all             all             [VM1 IP address]/32     trust
host    all             all             [VM2 IP address]/32     trust
...
```

> start postgresql service

```Shell
systemctl start postgresql-10.service
systemctl enable postgresql-10.service
```

### 3. redis server (run on AI VMs locally)
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


### 4. nginx (nginx server)
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


### 5. virtualenv (AI VMs)
> for Redhat Linux (centos)
```Shell
sudo yum -y install python-virtualenv
```


### 6. tensorflow 2.0+ (AI VMs)

> make sure pip version > 19.0.x
```Shell
sudo pip -V
```


### 7. wsgi (AI VMs)
> for both Linux system
```Shell
sudo pip install uwsgi
```



## Project installation Steps (AI VMs)


### 1. prepare the project
> make project folder and copy the project files.
**for example the project name is "opt"**:

+ 1.1. [ on AI VMs ]  
Copy the packed project source files
```Shell
mkdir /opt/chatfilter/main
cd /opt/chatfilter/main
copy files into the folder
```

+ 1.2. [ on nginx server ]  
nginx config

> please refer to /etc/nginx/nginx.conf and /etc/nginx/conf.d/ai.conf in the chinese ai-chat-filter project.  

+ 1.3. [ on AI VMs ]  
Set Project config
> copy setting.ini and chagne config you need
```Shell
cp setting.ini.example setting.ini
nano setting.ini
```

> check and set corresponding entries  
please refer to https://en.wikipedia.org/wiki/List_of_tz_database_time_zones for TZ identifiers
```EditorConfig
[MAIN]
ALLOWED_HOSTS = *
TIME_ZONE = America/Costa_Rica
LANGUAGE_MODE = EN

[DATABASE]
DATABASE_HOST = [master DB address]

[TWICE]
TWICE_HOST = [another AI VM's address]
```

+ 1.4. [ on master DB ]  
create a new database and set db users.  
DB name is the same as DATABASE_NAME in setting.ini mentioned above

```Shell
sudo su - postgres -c "psql"
\conninfo
\password postgres
CREATE DATABASE [name of database];
\q
``` 

+ 1.5. [ on AI VMs ]  
Create logs directory in project and make sure the logs folder changeable for supervisor(python)
```Shell
mkdir /opt/logs
chmod -R 777 /opt/logs
```


### 2. build up virtual environment (AI VMs)
> Create virtual environment named venv:
```Shell
cd /opt
python3 -m venv venv
chmod -R 777 venv
source /opt/venv/bin/activate
python -V
pip -V
```
> the python version should be at least 3.7.5 and pip version should be 19+


### 3. install python packages (AI VMs)
> make sure the "venv" has been activated
```Shell
cd /opt/chatfilter/main
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. initizlize django framework initialize (AI VMs)
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
..fill all the form and save the django admin superuser infos

> collect and copy the static file in project to improve performance
```Shell
python manage.py collectstatic
```
> A folder called static will be generated under /opt/chatfilter/main/. Move it to somewhere on the nginx server, then modify corresponding nginx.conf.

```
server {
    ...

    # Django static file
    location /static {
        alias [the path of the static folder mentioned above];
    }


```

```
# then restart nginx (nginx server)

sudo nginx -t
sudo systemctl restart nginx
or
sudo nginx -s reload

```

### 7. firewall setting
> open tcp port for chatting socket if need
```Shell
sudo firewall-cmd --permanent --zone=public --add-port=8025/tcp
```




## For linux product deploy using supervisor (AI VMs)
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
```


> symbolic link to supervisor config
```Shell
sudo ln -s /opt/chatfilter/main/supervisor.conf /etc/supervisord.d/ai.conf.ini
```

> reload supervisor
```Shell
sudo supervisorctl reload
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start all
```

## Others

> Check the SELinux and add policy to nginx or just disable it
```Shell
sestatus
setsebool -P httpd_can_network_connect 1
```

*SELinux might block the socket connection between nginx and supervisord*
