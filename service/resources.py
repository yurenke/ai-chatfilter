from import_export import resources
from .models import BlockedSentence, GoodSentence, ChangeNicknameRequest

class BlockedSentenceResource(resources.ModelResource):
    class Meta:
        model = BlockedSentence
        fields = ('message', 'date', 'status',)

class GoodSentenceResource(resources.ModelResource):
    class Meta:
        model = GoodSentence
        fields = ('message', 'date', 'status',)

class ChangeNicknameRequestResource(resources.ModelResource):
    class Meta:
        model = ChangeNicknameRequest
        fields = ('nickname', 'date', 'status',)