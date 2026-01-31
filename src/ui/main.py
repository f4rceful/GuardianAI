import flet as ft
import sys
import os
import traceback

# Add project root to path
sys.path.append(os.getcwd())

from src.ui.views import GuardianApp

def main(page: ft.Page):
    try:
        page.title = "GuardianAI"
        app = GuardianApp(page)
        page.update()
    except Exception as e:
        print(f"CRITICAL ERROR IN MAIN: {e}")
        traceback.print_exc()
        page.add(ft.Text(f"Error starting app: {e}", color="red"))
        page.update()

if __name__ == "__main__":
    print("Запуск GuardianAI Web App...")
    print("Please wait for the browser to open...")
    try:
        # Try using web browser view
        ft.app(target=main, view=ft.AppView.WEB_BROWSER)
    except Exception as e:
        print(f"FAILED TO START FLET APP: {e}")
        traceback.print_exc()
    print("App execution finished.")
