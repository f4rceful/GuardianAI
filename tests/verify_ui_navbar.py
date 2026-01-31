import flet as ft

try:
    nb = ft.NavigationBar(
        destinations=[
            ft.NavigationDestination(icon=ft.icons.HOME, label="Home"),
            ft.NavigationDestination(icon=ft.icons.SETTINGS, label="Settings"),
        ],
        selected_index=0
    )
    print("NavigationBar init success")
except Exception as e:
    print("NavigationBar init error:", e)
