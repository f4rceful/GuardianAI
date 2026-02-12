import flet as ft
import sys
import os
import traceback

# Добавляем корень проекта в путь (path)
sys.path.append(os.getcwd())

from src.ui.views import GuardianApp

def main(page: ft.Page):
    try:
        page.title = "GuardianAI"
        app = GuardianApp(page)
        page.update()
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА В MAIN: {e}")
        traceback.print_exc()
        page.add(ft.Text(f"Ошибка при запуске приложения: {e}", color="red"))
        page.update()

if __name__ == "__main__":
    print("Запуск GuardianAI Web App...")
    print("Пожалуйста, подождите открытия браузера...")
    try:
        # Попытка использования просмотра в веб-браузере
        ft.app(target=main, view=ft.AppView.WEB_BROWSER)
    except Exception as e:
        print(f"НЕ УДАЛОСЬ ЗАПУСТИТЬ FLET APP: {e}")
        traceback.print_exc()
    print("Выполнение приложения завершено.")
