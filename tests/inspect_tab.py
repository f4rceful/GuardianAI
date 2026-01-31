import flet as ft
try:
    t = ft.Tab(label="Test")
    print("Tab Created with label")
    print("Tab Dir:", dir(t))
    print("Tab Vars:", vars(t))
except Exception as e:
    print("Error creating Tab with label:", e)

try:
    t = ft.Tab(text="Test")
    print("Tab Created with text")
except Exception as e:
    print("Error creating Tab with text:", e)
