from django.contrib import admin
from ai.models import AbstractMeaning, Language, PartOfSpeech, Vocabulary, SoundVocabulary, DigitalVocabulary, NewVocabulary, TextbookSentense, NicknameTextbook


class AbstractMeaningAdmin(admin.ModelAdmin):
    fields = ['meaning']
    list_display = ['meaning']
    empty_value_display = '---'


class LanguageAdmin(admin.ModelAdmin):
    fields = ['code', 'chinese']
    list_display = ['code', 'chinese']
    empty_value_display = '---'


class PartOfSpeechAdmin(admin.ModelAdmin):
    fields = ['code', 'chinese']
    list_display = ['code', 'chinese']
    empty_value_display = '---'


class VocabularyAdmin(admin.ModelAdmin):
    fields = ['context', 'language', 'status', 'freq', 'meaning', 'part', 'abstract']
    list_display = ['context', 'status', 'freq', 'meaning', 'date']
    readonly_fields = ('context',)
    sortable_by = ('date',)
    empty_value_display = '---'

    search_fields = ('context',)


class SoundVocabularyAdmin(admin.ModelAdmin):
    # fields = ['pinyin', 'type', 'status']
    fields = ['pinyin', 'type', 'freq', 'status', 'vocabulary']
    list_display = ['pinyin', 'type', 'freq', 'status']
    # list_editable = ['type', 'status']
    empty_value_display = '---'

    readonly_fields = ('vocabulary',)
    search_fields = ('pinyin',)


class DigitalVocabularyAdmin(admin.ModelAdmin):
    fields = ['digits', 'pinyin', 'type', 'freq', 'status', 'vocabulary']
    list_display = ['digits', 'pinyin', 'type', 'freq', 'status']
    empty_value_display = '---'

    readonly_fields = ('vocabulary',)
    search_fields = ('digits', 'pinyin')


class NewVocabularyAdmin(admin.ModelAdmin):
    fields = ['pinyin', 'text', 'type', 'freq', 'status']
    list_display = ['pinyin', 'text', 'freq']
    empty_value_display = '---'

    search_fields = ('pinyin', )


class TextbookSentenseAdmin(admin.ModelAdmin):
    fields = ['origin', 'text', 'keypoint', 'weight', 'status', 'reason']
    list_display = ['origin', 'text', 'status']
    empty_value_display = '---'

    search_fields = ('text', 'status')

class NicknameTextbookAdmin(admin.ModelAdmin):
    fields = ['origin', 'text', 'status']
    list_display = ['origin', 'text', 'status']
    empty_value_display = '---'

    search_fields = ('text', 'status')

    


admin.site.register(AbstractMeaning, AbstractMeaningAdmin)
admin.site.register(Language, LanguageAdmin)
admin.site.register(PartOfSpeech, PartOfSpeechAdmin)
admin.site.register(Vocabulary, VocabularyAdmin)
admin.site.register(SoundVocabulary, SoundVocabularyAdmin)
admin.site.register(DigitalVocabulary, DigitalVocabularyAdmin)
admin.site.register(NewVocabulary, NewVocabularyAdmin)
admin.site.register(TextbookSentense, TextbookSentenseAdmin)
admin.site.register(NicknameTextbook, NicknameTextbookAdmin)