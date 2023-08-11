from django.contrib import admin
from django.http import HttpResponse
from django.template.response import TemplateResponse
from django.utils.translation import gettext_lazy as _
from django import forms
from import_export.admin import ImportExportModelAdmin, ImportExportActionModelAdmin, ExportMixin
from import_export.signals import post_export
from service.resources import BlockedSentenceResource, GoodSentenceResource, ChangeNicknameRequestResource
from service.models import Blockword, Whiteword, BlockUser, BlockedSentence, GoodSentence, AnalyzingData, UnknownWord, Textbook, ChangeNicknameRequest, DynamicPinyinBlock, DynamicNicknamePinyinBlock
import datetime



class BlockwordAdmin(admin.ModelAdmin):
    fields = ['text']
    list_display = ['text', 'date']
    empty_value_display = '---'

class WhitewordAdmin(admin.ModelAdmin):
    fields = ['text']
    list_display = ['text', 'date']
    empty_value_display = '---'

class BlockUserAdmin(admin.ModelAdmin):
    fields = ['name']
    list_display = ['name', 'date']
    empty_value_display = '---'

class BlockedSentenceAdmin(ExportMixin, admin.ModelAdmin):
    fields = ['message', 'text', 'reason', 'type', 'status']
    list_display = ['message', 'text', 'reason', 'type', 'status', 'date']
    list_filter = ('status', )
    # search_fields = ['message', 'status', 'date']
    search_fields = ['message']
    empty_value_display = '---'
    resource_class = BlockedSentenceResource

    # override
    def export_action(self, request, *args, **kwargs):
        if not self.has_export_permission(request):
            raise PermissionDenied

        formats = self.get_export_formats()
        form = BsExportForm(formats, request.POST or None)
        if form.is_valid():
            file_format = formats[
                int(form.cleaned_data['file_format'])
            ]()

            queryset = self.get_export_queryset(request)

            q_date = form.cleaned_data['date_where']
            # print('q_date: ', q_date)
            q_date_list = str(q_date).split('-')

            if len(q_date_list) == 3:
                _year = int(q_date_list[0])
                _month = int(q_date_list[1])
                _day = int(q_date_list[2])
                # print('ymd: ', _year, _month, _day)
                _gte_date = datetime.date(_year, _month, _day)
                _lte_date = _gte_date + datetime.timedelta(days=1)
            
                queryset = queryset.filter(date__gte=_gte_date, date__lte=_lte_date)

            
            export_data = self.get_export_data(file_format, queryset, request=request)
            content_type = file_format.get_content_type()
            response = HttpResponse(export_data, content_type=content_type)
            response['Content-Disposition'] = 'attachment; filename="%s"' % (
                self.get_export_filename(request, queryset, file_format),
            )

            post_export.send(sender=None, model=self.model)
            return response

        context = self.get_export_context_data()

        context.update(self.admin_site.each_context(request))

        context['title'] = _("Export")
        context['form'] = form
        context['opts'] = self.model._meta
        request.current_app = self.admin_site.name
        return TemplateResponse(request, [self.export_template_name],
                                context)

   
class BsExportForm(forms.Form):
    file_format = forms.ChoiceField(
        label=_('Format'),
        choices=(),
    )

    date_where = forms.DateField(label=_('Date'),)

    def __init__(self, formats, *args, **kwargs):
        super().__init__(*args, **kwargs)
        choices = []
        for i, f in enumerate(formats):
            choices.append((str(i), f().get_title(),))
        if len(formats) > 1:
            choices.insert(0, ('', '---'))

        self.fields['file_format'].choices = choices



class GoodSentenceAdmin(ExportMixin, admin.ModelAdmin):
    fields = ['message', 'text', 'type', 'status']
    list_display = ['message', 'text', 'type', 'status', 'date']
    search_fields = ['message']
    empty_value_display = '---'

    resource_class = GoodSentenceResource

    # override
    def export_action(self, request, *args, **kwargs):
        if not self.has_export_permission(request):
            raise PermissionDenied

        formats = self.get_export_formats()
        form = BsExportForm(formats, request.POST or None)
        if form.is_valid():
            file_format = formats[
                int(form.cleaned_data['file_format'])
            ]()

            queryset = self.get_export_queryset(request)

            q_date = form.cleaned_data['date_where']
            # print('q_date: ', q_date)
            q_date_list = str(q_date).split('-')

            if len(q_date_list) == 3:
                _year = int(q_date_list[0])
                _month = int(q_date_list[1])
                _day = int(q_date_list[2])
                # print('ymd: ', _year, _month, _day)
                _gte_date = datetime.date(_year, _month, _day)
                _lte_date = _gte_date + datetime.timedelta(days=1)
            
                queryset = queryset.filter(date__gte=_gte_date, date__lte=_lte_date)

            
            export_data = self.get_export_data(file_format, queryset, request=request)
            content_type = file_format.get_content_type()
            response = HttpResponse(export_data, content_type=content_type)
            response['Content-Disposition'] = 'attachment; filename="%s"' % (
                self.get_export_filename(request, queryset, file_format),
            )

            post_export.send(sender=None, model=self.model)
            return response

        context = self.get_export_context_data()

        context.update(self.admin_site.each_context(request))

        context['title'] = _("Export")
        context['form'] = form
        context['opts'] = self.model._meta
        request.current_app = self.admin_site.name
        return TemplateResponse(request, [self.export_template_name],
                                context)



class AnalyzingDataAdmin(admin.ModelAdmin):
    fields = ['year', 'month', 'day', 'good_sentence', 'blocked_sentence', 'json_blocked_detail', 'json_addition']
    list_display = ['year', 'month', 'day', 'good_sentence', 'blocked_sentence']
    empty_value_display = '---'

class UnknownWordAdmin(admin.ModelAdmin):
    fields = ['unknown', 'text', 'status']
    list_display = ['unknown', 'text', 'status', 'date']
    search_fields = ['unknown']
    empty_value_display = '---'

class TextbookAdmin(admin.ModelAdmin):
    fields = ['message', 'text', 'status', 'type', 'model']
    list_display = ['text', 'status', 'type', 'model', 'date']
    search_fields = ['text']
    empty_value_display = '---'

class ChangeNicknameRequestAdmin(ExportMixin, admin.ModelAdmin):
    fields = ['nickname', 'status']
    list_display = ['nickname', 'status', 'date']
    search_fields = ['nickname']
    empty_value_display = '---'

    resource_class = ChangeNicknameRequestResource

    # override
    def export_action(self, request, *args, **kwargs):
        if not self.has_export_permission(request):
            raise PermissionDenied

        formats = self.get_export_formats()
        form = BsExportForm(formats, request.POST or None)
        if form.is_valid():
            file_format = formats[
                int(form.cleaned_data['file_format'])
            ]()

            queryset = self.get_export_queryset(request)

            q_date = form.cleaned_data['date_where']
            # print('q_date: ', q_date)
            q_date_list = str(q_date).split('-')

            if len(q_date_list) == 3:
                _year = int(q_date_list[0])
                _month = int(q_date_list[1])
                _day = int(q_date_list[2])
                # print('ymd: ', _year, _month, _day)
                _gte_date = datetime.date(_year, _month, _day)
                _lte_date = _gte_date + datetime.timedelta(days=1)
            
                queryset = queryset.filter(date__gte=_gte_date, date__lte=_lte_date)

            
            export_data = self.get_export_data(file_format, queryset, request=request)
            content_type = file_format.get_content_type()
            response = HttpResponse(export_data, content_type=content_type)
            response['Content-Disposition'] = 'attachment; filename="%s"' % (
                self.get_export_filename(request, queryset, file_format),
            )

            post_export.send(sender=None, model=self.model)
            return response

        context = self.get_export_context_data()

        context.update(self.admin_site.each_context(request))

        context['title'] = _("Export")
        context['form'] = form
        context['opts'] = self.model._meta
        request.current_app = self.admin_site.name
        return TemplateResponse(request, [self.export_template_name],
                                context)

class DynamicPinyinBlockAdmin(admin.ModelAdmin):
    fields = ['text', 'pinyin']
    list_display = ['text', 'pinyin', 'date']
    search_fields = ['text']
    empty_value_display = '---'

class DynamicNicknamePinyinBlockAdmin(admin.ModelAdmin):
    fields = ['text', 'pinyin']
    list_display = ['text', 'pinyin', 'date']
    search_fields = ['text']
    empty_value_display = '---'
    

admin.site.register(Blockword, BlockwordAdmin)
admin.site.register(BlockUser, BlockUserAdmin)
admin.site.register(Whiteword, WhitewordAdmin)
admin.site.register(BlockedSentence, BlockedSentenceAdmin)
admin.site.register(GoodSentence, GoodSentenceAdmin)
admin.site.register(AnalyzingData, AnalyzingDataAdmin)
admin.site.register(UnknownWord, UnknownWordAdmin)
admin.site.register(Textbook, TextbookAdmin)
admin.site.register(ChangeNicknameRequest, ChangeNicknameRequestAdmin)
admin.site.register(DynamicPinyinBlock, DynamicPinyinBlockAdmin)
admin.site.register(DynamicNicknamePinyinBlock, DynamicNicknamePinyinBlockAdmin)
