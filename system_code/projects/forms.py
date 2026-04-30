from django import forms
from .models import Project, Dataset


class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'source_path', 'format_type', 'description']
        widgets = {
            'source_path': forms.TextInput(attrs={'placeholder': '例如: /home/user/datasets/mydata', 'class': 'form-control'}),
        }


class DatasetImportForm(forms.Form):
    name = forms.CharField(label='数据集名称', max_length=100)
    path = forms.CharField(label='数据集路径', max_length=500, help_text='数据集的绝对路径，包含images/sparse等子目录')
    format_type = forms.ChoiceField(label='数据格式', choices=Dataset.FORMAT_CHOICES, initial='colmap')
    description = forms.CharField(label='描述', widget=forms.Textarea, required=False)


class ProjectCreateForm(forms.ModelForm):
    dataset = forms.ModelChoiceField(
        label='选择数据集',
        queryset=Dataset.objects.none(),
        required=False,
        help_text='从已有数据集中选择，或在下文手动输入路径'
    )
    source_path_manual = forms.CharField(
        label='或手动输入数据路径',
        max_length=500,
        required=False,
        widget=forms.TextInput(attrs={'placeholder': '例如: /home/user/datasets/mydata'})
    )

    class Meta:
        model = Project
        fields = [
            'name', 'description',
            'exp_name', 'resolution', 'eval_mode', 'llffhold', 'white_background',
            'manhattan', 'platform', 'pos', 'rot',
            'm_region', 'n_region', 'extend_rate', 'visible_rate',
            'iterations', 'quiet',
        ]
        widgets = {
            'pos': forms.TextInput(attrs={'placeholder': '25.607364654541 0.0 -12.012700080872'}),
            'rot': forms.TextInput(attrs={'placeholder': '0.923 0.0 0.384 0.0 1.0 0.0 -0.384 0.0 0.923'}),
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'exp_name': forms.TextInput(attrs={'class': 'form-control'}),
        }

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        if user:
            self.fields['dataset'].queryset = Dataset.objects.filter(created_by=user)
        self.fields['name'].help_text = '项目显示名称'
        self.fields['exp_name'].help_text = '实验名称，用于输出文件夹命名'

    def clean(self):
        cd = super().clean()
        dataset = cd.get('dataset')
        manual_path = cd.get('source_path_manual')
        if dataset:
            cd['source_path'] = dataset.source_path
        elif manual_path:
            cd['source_path'] = manual_path
        else:
            raise forms.ValidationError('请选择数据集或手动输入数据路径')
        return cd

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.source_path = self.cleaned_data['source_path']
        if commit:
            instance.save()
        return instance


class TrainingConfigForm(forms.ModelForm):
    """Advanced training parameter configuration"""

    class Meta:
        model = Project
        fields = [
            'iterations', 'resolution', 'llffhold',
            'm_region', 'n_region', 'extend_rate', 'visible_rate',
            'manhattan', 'platform', 'pos', 'rot',
            'eval_mode', 'white_background', 'quiet',
        ]
        widgets = {
            'pos': forms.TextInput(attrs={'class': 'form-control'}),
            'rot': forms.TextInput(attrs={'class': 'form-control'}),
        }


class RenderConfigForm(forms.Form):
    load_iteration = forms.IntegerField(label='加载迭代次数', initial=60_000, min_value=1)
    skip_train = forms.BooleanField(label='跳过训练集渲染', initial=True, required=False)
    skip_test = forms.BooleanField(label='跳过测试集渲染', initial=False, required=False)
