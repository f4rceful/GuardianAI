import flet as ft
import inspect

print("--- Tab Annotations ---")
try:
    print(ft.Tab.__init__.__annotations__)
except:
    print("No annotations")

print("\n--- Tabs Annotations ---")
try:
    print(ft.Tabs.__init__.__annotations__)
except:
    print("No annotations")

t = ft.Tab(label="Test")
print("\nHas content?", hasattr(t, "content"))
print("Has controls?", hasattr(t, "controls"))
