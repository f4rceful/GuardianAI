import flet as ft

print("Checking ft.Tabs...")
try:
    ft.Tabs(tabs=[ft.Tab(label="T1")])
    print("SUCCESS with 'tabs'")
except Exception as e:
    print("FAILED with 'tabs':", e)

try:
    ft.Tabs(controls=[ft.Tab(label="T1")])
    print("SUCCESS with 'controls'")
except Exception as e:
    print("FAILED with 'controls':", e)

try:
    ft.Tabs([ft.Tab(label="T1")])
    print("SUCCESS with POSITIONAL tabs")
except Exception as e:
    print("FAILED with POSITIONAL tabs:", e)


print("\nChecking available attributes in ft module...")
print([x for x in dir(ft) if "Tab" in x])
