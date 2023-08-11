
from configparser import RawConfigParser
import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Application definition

INSTALLED_APPS = [
    'channels',
    'import_export',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework.authtoken',
    'djoser',
    'corsheaders',
    'chat',
    'ai',
    'dataparser',
    'service',
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework.authentication.TokenAuthentication',
    ),
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
    ],
}

STREAM_API_KEY = '3fmx98fcw2xm'
STREAM_API_SECRET = 'sqgvg59aujuj2arc42tf65w25qxyj7spwurappu9jz8vx2ufwhpp4766h9kw8mu3'
DJOSER = {
    'SERIALIZERS': {
        'token': 'auth.serializers.StreamTokenSerializer',
    }
}


MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    
]

ROOT_URLCONF = 'service.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# WSGI_APPLICATION = 'service.wsgi.application'
ASGI_APPLICATION = "trainingservice.routing.application"
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', config.get('CHANNEL', 'CHANNEL_PORT'))],
        },
        # "ROUTING": "service.routing",
    },
}

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'aitrainservice',
        'USER': 'postgreslocal',
        'PASSWORD': 'postgreslocal',
        'HOST': 'localhost',
        'PORT': 5432
    },
    # 'postgresql': {
    #     'ENGINE': 'django.db.backends.postgresql_psycopg2',
    # },
    # 'mysql': {
    #     'ENGINE': 'django.db.backends.mysql',
    # }
}


# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

# AUTH_PASSWORD_VALIDATORS = [
#     {
#         'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
#     },
#     {
#         'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
#     },
#     {
#         'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
#     },
#     {
#         'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
#     },
# ]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'America/Puerto_Rico'
LANGUAGE_MODE = 'CH'

USE_I18N = True
USE_L10N = True
USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, "static/")

CORS_ORIGIN_ALLOW_ALL  = True

IMPORT_EXPORT_USE_TRANSACTIONS = True
IMPORT_EXPORT_IMPORT_PERMISSION_CODE = 'add'

CACHES = {
    'default': {
        'BACKEND': 'redis_cache.cache.RedisCache',
        'LOCATION': 'localhost:6379:1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'MAX_ENTRIES': 5000,
        },
    },
}
