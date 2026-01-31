import flet as ft
import inspect

print("ft.TabBar arguments:", inspect.signature(ft.TabBar.__init__))
print("ft.TabBar dir:", dir(ft.TabBar))
try:
    tb = ft.TabBar(tabs=[ft.Tab(label="T1")])
    print("TabBar init success")
    print("Has selected_index?", hasattr(tb, "selected_index"))
except Exception as e:
    print("TabBar init error:", e)
