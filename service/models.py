from django.db import models



class Blockword(models.Model):
    text = models.CharField(max_length=64)
    date = models.DateTimeField(auto_now_add=True, blank=True)


class Whiteword(models.Model):
    text = models.CharField(max_length=64)
    date = models.DateTimeField(auto_now_add=True, blank=True)


class BlockUser(models.Model):
    name = models.CharField(max_length=64)
    date = models.DateTimeField(auto_now_add=True, blank=True)


class BlockedSentence(models.Model):
    message = models.CharField(max_length=96)
    text = models.CharField(max_length=64, null=True, blank=True)
    reason = models.CharField(max_length=64, null=True, blank=True)
    type = models.IntegerField(default=1)
    status = models.IntegerField(default=1)
    date = models.DateTimeField(auto_now=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['-date',]),
            models.Index(fields=['status',]),
        ]


class GoodSentence(models.Model):
    message = models.CharField(max_length=96)
    text = models.CharField(max_length=64, null=True, blank=True)
    type = models.IntegerField(default=1)
    status = models.IntegerField(default=0)
    date = models.DateTimeField(auto_now=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['-date',]),
        ]
    

class AnalyzingData(models.Model):
    year = models.PositiveSmallIntegerField()
    month = models.PositiveSmallIntegerField()
    day = models.PositiveSmallIntegerField()
    good_sentence = models.PositiveIntegerField(default=0)
    blocked_sentence = models.PositiveIntegerField(default=0)
    json_blocked_detail = models.CharField(max_length=2048, blank=True)
    json_addition = models.CharField(max_length=1024, blank=True)


class UnknownWord(models.Model):
    unknown = models.CharField(max_length=64)
    text = models.CharField(max_length=64, null=True, blank=True)
    status = models.IntegerField(default=1)
    date = models.DateTimeField(auto_now=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['-date',]),
            models.Index(fields=['status',]),
        ]


class Textbook(models.Model):
    message = models.CharField(max_length=255)
    text = models.CharField(max_length=64)
    type = models.IntegerField(default=1)  # 1= normal, 2= check right
    status = models.IntegerField(default=1)
    model = models.IntegerField(default=1)  # 0= all, 1= pinyin, 2= grammar
    date = models.DateTimeField(auto_now=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['-date',]),
            models.Index(fields=['status',]),
            models.Index(fields=['model',]),
        ]


class ChangeNicknameRequest(models.Model):
    nickname = models.CharField(max_length=63)
    status = models.IntegerField(default=1)
    date = models.DateTimeField(auto_now=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['status',]),
        ]


class DynamicPinyinBlock(models.Model):
    text = models.CharField(max_length=16)
    pinyin = models.CharField(max_length=64)
    type = models.IntegerField(default=1)
    date = models.DateTimeField(auto_now=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['type',]),
            models.Index(fields=['-date',]),
        ]

class DynamicNicknamePinyinBlock(models.Model):
    text = models.CharField(max_length=16)
    pinyin = models.CharField(max_length=64)
    date = models.DateTimeField(auto_now=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['-date',]),
        ]

class DynamicAlertWords(models.Model):
    text = models.CharField(max_length=16)
    date = models.DateTimeField(auto_now_add=True, blank=True)