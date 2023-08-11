
from django.contrib import admin
from dataparser.models import CustomDictionaryWord

class CustomDictionaryWordAdmin(admin.ModelAdmin):
    fields = ['text', 'pinyin']
    list_display = ['text', 'pinyin', 'date']
    empty_value_display = '---'


admin.site.register(CustomDictionaryWord, CustomDictionaryWordAdmin)