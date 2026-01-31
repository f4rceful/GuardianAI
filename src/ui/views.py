import flet as ft
from src.core.classifier import GuardianClassifier
from src.core.history import HistoryManager
import os

class GuardianApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.history_manager = HistoryManager()

        self.page.title = "GuardianAI: Цифровой телохранитель"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        
        # Инициализация ML мозга
        self.classifier = GuardianClassifier()
        if not self.classifier.is_trained:
             print("ВНИМАНИЕ: Модель не найдена. Запустите 'python train.py'!")
        
        # Состояние
        self.messages = []
        self.current_tab_index = 0
        
        # UI Элементы
        self.init_components()
        self.load_history_to_ui()
        self.navigate_to_dashboard()

    def load_history_to_ui(self):
        for entry in self.history_manager.history:
            self.add_message_to_list_ui_only(entry)

    def add_message_to_list_ui_only(self, result):
        # Reuse logic for UI tile creation without saving to history again
        score = result.get('ml_score', 0)
        is_scam = result.get('is_scam', False)
        
        if is_scam:
            icon = "dangerous"
            color = "red"
            verdict = "DANGER"
            bgcolor = "red50"
        elif score > 0.4:
            icon = "warning"
            color = "orange"
            verdict = "SUSPICIOUS"
            bgcolor = "orange50"
        else:
            icon = "check_circle"
            color = "green"
            verdict = "SAFE"
            bgcolor = "green50"
            
        tile = ft.Container(
            content=ft.Row([
                ft.Icon(icon, color=color, size=30),
                ft.Column([
                    ft.Text(result['text'], max_lines=2, overflow=ft.TextOverflow.ELLIPSIS, weight=ft.FontWeight.BOLD),
                    ft.Text(f"{verdict} • Confidence: {score:.0%}", color=color, size=12)
                ], expand=True),
                ft.Row([
                     ft.Container(content=ft.Icon("verified_user", color="green"), tooltip="В белый список", on_click=lambda e, t=result['text']: self.add_to_whitelist(t), padding=5),
                     ft.Container(content=ft.Icon("info_outline", color="grey"), padding=5)
                ])
            ]),
            padding=10,
            bgcolor=bgcolor,
            border_radius=10,
            margin=ft.margin.only(bottom=5)
        )
        self.messages_list.controls.insert(0, tile) 


    def init_components(self):
        # 1. Dashboard Components
        self.status_icon = ft.Icon("lock", color="green", size=50)
        self.status_text = ft.Text("Система активна. Мониторинг...", size=20, weight=ft.FontWeight.BOLD)
        self.messages_list = ft.ListView(expand=True, spacing=10, padding=20)
        
        # 2. Simulation (Control Panel) Components
        self.msg_input = ft.TextField(label="Текст сообщения", hint_text="Введите текст для проверки...", expand=True)
        self.send_btn = ft.ElevatedButton("Симуляция входящего SMS", icon="send", on_click=self.on_simulate_message)
        
        # 3. Alert Components
        self.alert_msg = ft.Text("", size=18, color="white", italic=True)
        self.alert_reason = ft.Text("", color="yellow", weight=ft.FontWeight.BOLD)
        
        self.alert_container = ft.Container(
            content=ft.Column([
                ft.Icon("warning", color="white", size=100),
                ft.Text("ВНИМАНИЕ!", size=40, color="white", weight=ft.FontWeight.BOLD),
                ft.Text("ОБНАРУЖЕНА УГРОЗА МОШЕННИЧЕСТВА", size=20, color="white"),
                ft.Divider(color="white"),
                ft.Text("Текст сообщения:", color="white70"),
                self.alert_msg,
                self.alert_reason,
                ft.Container(height=20),
                ft.ElevatedButton("Я ПОНЯЛ, ОТБОЙ", color="red", bgcolor="white", on_click=self.dismiss_alert)
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor="red",
            expand=True,
            padding=20,
            visible=False
        )

    def navigate_to_dashboard(self):
        self.page.clean()
        
        # --- Define Views (Content) ---
        
        # Stats
        stats = self.history_manager.get_stats()
        self.txt_total = ft.Text(str(stats['total']), size=30, weight=ft.FontWeight.BOLD)
        self.txt_scam = ft.Text(str(stats['scam']), size=30, color="red", weight=ft.FontWeight.BOLD)
        
        stats_row = ft.Row([
            ft.Container(
                content=ft.Column([ft.Text("Всего проверено"), self.txt_total], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor="blue50", padding=10, border_radius=10, expand=True
            ),
            ft.Container(
                content=ft.Column([ft.Text("Угроз отражено"), self.txt_scam], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor="red50", padding=10, border_radius=10, expand=True
            )
        ])

        # View 1: Grandma's Screen
        self.dashboard_view = ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Row([self.status_icon, self.status_text], alignment=ft.MainAxisAlignment.CENTER),
                    padding=20,
                    bgcolor="green", 
                    border_radius=10
                ),
                ft.Container(height=10),
                stats_row,
                ft.Text("История сообщений:", size=16),
                self.messages_list
            ]),
            padding=10,
            visible=True
        )
        
        # View 2: Hacker Control Panel
        self.control_panel_view = ft.Container(
            content=ft.Column([
                ft.Text("Симуляция атаки", size=20, weight=ft.FontWeight.BOLD),
                ft.Text("Введи текст, который якобы пришел на телефон:", color="grey"),
                ft.Row([self.msg_input, self.send_btn]),
                ft.Divider(),
                ft.Text("Подсказки для теста:", weight=ft.FontWeight.BOLD),
                ft.Chip(label="Мам, скинь денег", on_click=lambda e: self.fill_input(e.control.label)),
                ft.Chip(label="Ваш аккаунт взломан", on_click=lambda e: self.fill_input(e.control.label)),
                ft.Chip(label="Привет, как дела?", on_click=lambda e: self.fill_input(e.control.label)),
            ]),
            padding=20,
            visible=False
        )

        # View 3: Settings
        self.settings_view = ft.Container(
            content=ft.Column([
                ft.Text("Настройки защиты", size=20, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Row([
                    ft.Icon("security", size=30),
                    ft.Column([
                        ft.Text("Строгий режим (Strict Mode)", weight=ft.FontWeight.BOLD),
                        ft.Text("Блокировать ВСЕ сообщения с кодами и паролями", size=12, color="grey")
                    ], expand=True),
                    ft.Switch(label="", on_change=self.on_strict_mode_change)
                ]),
                ft.Text("Если выключено (Smart Mode), система будет игнорировать официальные уведомления от банков и Госуслуг.", size=12, color="blue", italic=True)
            ]),
            padding=20,
            visible=False
        )

        # Place to hold current body
        self.body_container = ft.Container(
            content=ft.Column([self.dashboard_view, self.control_panel_view, self.settings_view]),
            expand=True
        )

        # --- Custom Tab Bar (Using Buttons) ---
        self.tab_btn1 = ft.ElevatedButton("Главная", icon="home", on_click=lambda _: self.set_tab(0), bgcolor="blue", color="white")
        self.tab_btn2 = ft.TextButton("Тест", icon="bug_report", on_click=lambda _: self.set_tab(1))
        self.tab_btn3 = ft.TextButton("Настройки", icon="settings", on_click=lambda _: self.set_tab(2))

        custom_tabs = ft.Container(
            content=ft.Row(
                controls=[self.tab_btn1, self.tab_btn2, self.tab_btn3],
                alignment=ft.MainAxisAlignment.SPACE_EVENLY
            ),
            padding=10,
            bgcolor="blue50"
        )
        
        self.page.add(
            ft.Stack([
                ft.Column([custom_tabs, self.body_container], expand=True),
                self.alert_container 
            ], expand=True)
        )
        self.strict_mode = False

    def on_strict_mode_change(self, e):
        self.strict_mode = e.control.value
        mode = "СТРОГИЙ" if self.strict_mode else "УМНЫЙ"
        self.page.snack_bar = ft.SnackBar(ft.Text(f"Режим защиты: {mode}"), bgcolor="blue")
        self.page.snack_bar.open = True
        self.page.update()

    def set_tab(self, index):
        self.current_tab_index = index
        self.dashboard_view.visible = (index == 0)
        self.control_panel_view.visible = (index == 1)
        self.settings_view.visible = (index == 2)
        
        self._update_tab_styles(index)
        self.page.update()

    def _update_tab_styles(self, selected_index):
        # Reset all
        for btn in [self.tab_btn1, self.tab_btn2, self.tab_btn3]:
            btn.bgcolor = None
            btn.color = "black"
            
        # Set active
        active_btn = [self.tab_btn1, self.tab_btn2, self.tab_btn3][selected_index]
        active_btn.bgcolor = "blue"
        active_btn.color = "white"

    def fill_input(self, text):
        self.msg_input.value = text
        self.msg_input.update()

    async def on_simulate_message(self, e):
        text = self.msg_input.value
        if not text:
            return
            
        self.msg_input.value = ""
        self.msg_input.update()
        
        # Get context (last 3 messages)
        context = self.history_manager.get_recent_context(limit=3)

        # Async analysis (Pass strict_mode from UI state + Context)
        result = await self.classifier.predict_async(text, self.strict_mode, context)
        
        # Add to UI
        self.add_message_to_list(result)
        
        # Check danger
        if result['is_scam']:
            await self.show_alert(result)
        else:
            self.page.snack_bar = ft.SnackBar(ft.Text("Сообщение доставлено (Безопасно)"), bgcolor="green")
            self.page.snack_bar.open = True
            self.page.update()

    def add_message_to_list(self, result):
        # Save to history
        self.history_manager.add_entry(result)
        
        # Update Stats UI
        stats = self.history_manager.get_stats()
        self.txt_total.value = str(stats['total'])
        self.txt_scam.value = str(stats['scam'])
        self.txt_total.update()
        self.txt_scam.update()
        
        score = result['ml_score']
        
        # Visual Severity Logic
        if result['is_scam']:
            icon = "dangerous"
            color = "red"
            verdict = "DANGER"
            bgcolor = "red50"
        elif score > 0.4:
            icon = "warning"
            color = "orange"
            verdict = "SUSPICIOUS"
            bgcolor = "orange50"
        else:
            icon = "check_circle"
            color = "green"
            verdict = "SAFE"
            bgcolor = "green50"
        
        # Entities Chips
        entity_controls = []
        if 'entities' in result and result['entities']:
            for cat, items in result['entities'].items():
                entity_icon = "label"
                entity_color = "grey"
                if cat == "AUTHORITY": entity_icon, entity_color = "local_police", "red"
                elif cat == "FINANCE": entity_icon, entity_color = "attach_money", "amber"
                elif cat == "URGENCY": entity_icon, entity_color = "warning", "orange"
                elif cat == "SENSITIVE": entity_icon, entity_color = "vpn_key", "purple"
                elif cat == "RELATIVE": entity_icon, entity_color = "family_restroom", "blue"
                
                for item in items:
                    entity_controls.append(
                        ft.Chip(label=ft.Text(item, size=10), leading=ft.Icon(entity_icon, size=12, color=entity_color), height=24)
                    )

        self.messages_list.controls.insert(0, ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(icon, color=color),
                    ft.Text(result['text'], expand=True, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START),
                
                # Show Entity Chips
                ft.Row(entity_controls, wrap=True) if entity_controls else ft.Container(),
                
                ft.Row([
                    ft.Text(f"{verdict} • Confidence: {score:.0%}", color=color, size=12)
                ], expand=True),
                ft.Row([
                     ft.Container(content=ft.Icon("verified_user", color="green"), tooltip="В белый список", on_click=lambda e: self.add_to_whitelist(result['text']), padding=5),
                     ft.Container(content=ft.Icon("bug_report", color="orange"), tooltip="Сообщить об ошибке", on_click=lambda e: self.report_error(result), padding=5),
                     ft.Container(content=ft.Icon("info_outline", color="grey"), padding=5)
                ])
            ]),
            padding=10,
            border=ft.border.all(1, color),
            border_radius=5,
            bgcolor=bgcolor,
            margin=ft.margin.only(bottom=5)
        ))
        self.messages_list.update()

    def add_to_whitelist(self, text):
        self.classifier.whitelist.add(text)
        self.page.snack_bar = ft.SnackBar(ft.Text("Добавлено в белый список!"), bgcolor="green")
        self.page.snack_bar.open = True
        self.page.update()

    def report_error(self, result):
        try:
             with open("dataset/user_feedback.txt", "a", encoding="utf-8") as f:
                 # Format: LABEL | TEXT (Label is opposite of prediction)
                 correct_label = "SAFE" if result['is_scam'] else "SCAM"
                 f.write(f"{correct_label} | {result['text']}\n")
                 
             self.page.snack_bar = ft.SnackBar(ft.Text("Спасибо! Мы учтем эту ошибку."), bgcolor="blue")
             self.page.snack_bar.open = True
             self.page.update()
        except Exception as e:
            print(f"Error saving feedback: {e}")

    async def show_alert(self, result):
        self.alert_msg.value = f"\"{result['text']}\""
        self.alert_reason.value = f"Причина: {result['reason']}"
        self.alert_container.visible = True
        self.page.update()
        
    def dismiss_alert(self, e):
        self.alert_container.visible = False
        self.page.update()
