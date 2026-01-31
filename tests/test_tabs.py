import flet as ft

def main(page: ft.Page):
    t1_content = ft.Text("View 1")
    t2_content = ft.Text("View 2")
    
    body = ft.Container(content=t1_content)
    
    def on_change(e):
        if e.control.selected_index == 0:
            body.content = t1_content
        else:
            body.content = t2_content
        body.update()

    tabs = ft.Tabs(
        selected_index=0,
        on_change=on_change,
        tabs=[
            ft.Tab(label="Tab 1", icon="home"),
            ft.Tab(label="Tab 2", icon="settings"),
        ]
    )
    
    page.add(tabs, body)

ft.app(target=main)
