from django.db import models


class AbstractMeaning(models.Model):
    meaning = models.CharField(max_length=1024)

    def __str__(self):
        return self.meaning


class Language(models.Model):
    code = models.CharField(max_length=8)
    chinese = models.CharField(max_length=8)

    def __str__(self):
        return self.code


class PartOfSpeech(models.Model):
    code = models.CharField(max_length=8)
    chinese = models.CharField(max_length=16)

    def __str__(self):
        return self.code


class Vocabulary(models.Model):
    context = models.CharField(max_length=255)
    language = models.ForeignKey(Language, on_delete=models.SET_NULL, null=True)
    status = models.IntegerField(default=1)
    freq = models.IntegerField(default=1)
    meaning = models.CharField(max_length=512, blank=True, default="") 
    part = models.ManyToManyField(PartOfSpeech)
    abstract = models.ManyToManyField(AbstractMeaning)
    similarity = models.ManyToManyField("self")
    date = models.DateTimeField(auto_now_add=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['language',]),
            models.Index(fields=['status',]),
        ]

    def __str__(self):
        return self.context


class SoundVocabulary(models.Model):
    pinyin = models.CharField(max_length=65)
    type = models.IntegerField(default=1)
    status = models.IntegerField(default=1)
    freq = models.IntegerField(default=1)
    vocabulary = models.ManyToManyField(Vocabulary)

    class Meta:
        indexes = [
            models.Index(fields=['pinyin',]),
        ]

    def __str__(self):
        return self.pinyin


class DigitalVocabulary(models.Model):
    digits = models.CharField(max_length=65)
    pinyin = models.CharField(max_length=65, default='')
    type = models.IntegerField(default=1)
    status = models.IntegerField(default=1)
    freq = models.IntegerField(default=10)
    vocabulary = models.ManyToManyField(Vocabulary)

    class Meta:
        indexes = [
            models.Index(fields=['digits',]),
        ]

    def __str__(self):
        return self.digits


class NewVocabulary(models.Model):
    pinyin = models.CharField(max_length=65, default='')
    text = models.CharField(max_length=65, default='')
    type = models.IntegerField(default=1)
    freq = models.IntegerField(default=20)
    status = models.IntegerField(default=1)

    class Meta:
        indexes = [
            models.Index(fields=['pinyin',]),
        ]

    def __str__(self):
        return self.pinyin


class TextbookSentense(models.Model):
    origin = models.CharField(max_length=512, default='')
    text = models.CharField(max_length=255, default='')
    keypoint = models.CharField(max_length=255, default='')
    weight = models.IntegerField(default=1)
    status = models.IntegerField(default=0)
    reason = models.IntegerField(default=0)

    class Meta:
        indexes = [
            models.Index(fields=['status',]),
            models.Index(fields=['reason',]),
            models.Index(fields=['weight',]),
        ]

class NicknameTextbook(models.Model):
    origin = models.CharField(max_length=512, default='')
    text = models.CharField(max_length=255, default='')
    status = models.IntegerField(default=0)

    class Meta:
        indexes = [
            models.Index(fields=['status',])
        ]

