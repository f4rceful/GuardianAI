import flet as ft
import sys

# Redirect stdout to a file to capture full help
with open("flet_help.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    help(ft.Tab)
