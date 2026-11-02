"""中文图表字体定位；不假设另一台Windows的系统盘符。"""
import os
import shutil
from pathlib import Path


def chinese_font(explicit=None):
    explicit = explicit or os.environ.get('OAS_CJK_FONT')
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f'中文字体不存在：{p}')
        return p
    windows = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'
    candidates = [windows/'msyh.ttc', windows/'simhei.ttf', windows/'simsun.ttc',
                  Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError('未找到中文字体；请在n17_server.yaml设置font_path，不会自动下载字体')


def configure_plot_fonts():
    from matplotlib import font_manager, rcParams
    p = chinese_font()
    font_manager.fontManager.addfont(str(p))
    rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=str(p)).get_name(), 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False


def prepare_font_cache(directory, explicit=None):
    """用已提供/系统字体填充框架本地缓存别名，不触发字体网络下载。"""
    directory=Path(directory)/'Ultralytics';directory.mkdir(parents=True,exist_ok=True)
    font=chinese_font(explicit)
    for name in ('Arial.ttf','Arial.Unicode.ttf'):
        target=directory/name
        if not target.exists():shutil.copy2(font,target)
    return font
