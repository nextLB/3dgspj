#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import platform


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vastgs_system.settings')

    # ── Windows autoreload fix ──────────────────────────────────────────────
    # WinError 1450 (系统资源不足) occurs when autoreload's filesystem polling
    # (os.stat) exhausts kernel resources under memory pressure from 3D training.
    # Disable autoreload by default on Windows to prevent this.
    if platform.system() == 'Windows':
        if len(sys.argv) >= 2 and sys.argv[1] == 'runserver':
            has_noreload = any(arg in sys.argv for arg in ('--noreload', '--no-reload', '-n'))
            if not has_noreload:
                sys.argv.append('--noreload')
                print("╔════════════════════════════════════════════════════════════╗")
                print("║ [通知] Windows 系统检测 - 已自动禁用自动重载 (--noreload)    ║")
                print("║ 运行 3DGS 训练时，自动重载会轮询扫描文件系统，              ║")
                print("║ 容易导致系统资源耗尽错误 (WinError 1450)。                  ║")
                print("║ 如需启用自动重载，请手动添加 --reload 参数。                ║")
                print("╚════════════════════════════════════════════════════════════╝")

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
