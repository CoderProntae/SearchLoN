"""PySide6 desktop UI for local chat, live web research, history, and quotas."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QKeyEvent, QPalette
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig, ConfigError, get_data_dir, load_config, save_config
from .limits import LimitStore, LimitStoreError
from .ollama import OllamaClient, OllamaError, estimate_tokens
from .rendering import render_markdown
from .research import SYSTEM_PROMPT, QuotaExceeded, ResearchAgent
from .storage import DataStore, local_day
from .web import WebSearchClient

APP_TITLE = "DeepSearch & Research AI Chat"


class ChatInput(QPlainTextEdit):
    submitRequested = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.submitRequested.emit()
            return
        super().keyPressEvent(event)


class MessageCard(QFrame):
    def __init__(self, role: str, content: str = "", streaming: bool = False) -> None:
        super().__init__()
        self.role = role
        self.streaming = streaming
        self.setObjectName("messageCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            "QFrame#messageCard { background: #171a21; border: 1px solid #282d38; "
            "border-radius: 12px; }"
            if role == "assistant"
            else "QFrame#messageCard { background: #202733; border: 1px solid #303b4c; "
            "border-radius: 12px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 11, 16, 12)
        layout.setSpacing(7)
        self.heading = QLabel("DeepSearch AI" if role == "assistant" else "Sen")
        self.heading.setObjectName("messageHeading")
        layout.addWidget(self.heading)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.browser.anchorClicked.connect(self._open_link)
        self.browser.setFrameShape(QFrame.Shape.NoFrame)
        self.browser.setStyleSheet(
            "QTextBrowser { background: transparent; border: 0; color: #e5e7eb; "
            "selection-background-color: #3565a8; }"
            "QTextBrowser pre { background: #10141b; padding: 8px; border-radius: 6px; }"
            "QTextBrowser code { font-family: Consolas, 'Cascadia Code', monospace; }"
        )
        self.browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.browser.setMinimumHeight(34)
        self.browser.setMaximumHeight(420)
        layout.addWidget(self.browser)
        self.set_content(content, markdown=not streaming)

    def set_content(self, text: str, *, markdown: bool = True) -> None:
        if markdown:
            self.browser.setHtml(render_markdown(text))
        else:
            self.browser.setPlainText(text)
        self.browser.document().adjustSize()
        natural_height = int(self.browser.document().size().height()) + 18
        self.browser.setMinimumHeight(max(34, min(natural_height, 420)))
        self.browser.updateGeometry()

    def _open_link(self, url: QUrl) -> None:
        if url.scheme().lower() in {"http", "https"}:
            QDesktopServices.openUrl(url)


class SearchEventCard(QFrame):
    def __init__(self, event: dict[str, str]) -> None:
        super().__init__()
        self.setObjectName("eventCard")
        self.setStyleSheet(
            "QFrame#eventCard { background: #111c28; border: 1px solid #29435a; "
            "border-left: 3px solid #43a3d3; border-radius: 8px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 7, 11, 7)
        layout.setSpacing(3)
        kind = event.get("kind", "info")
        labels = {
            "searching": "🔎 CANLI ARAMA",
            "result": "📚 KAYNAK BULUNDU",
            "fetching": "🌐 SİTE TARANIYOR",
            "fetched": "✓ SAYFA OKUNDU",
            "empty": "ℹ ARAMA",
            "error": "⚠ ARAŞTIRMA UYARISI",
        }
        tag = QLabel(labels.get(kind, "• ARAŞTIRMA"))
        tag.setStyleSheet("color: #6bc5ef; font-size: 10px; font-weight: 700;")
        layout.addWidget(tag)
        title = QLabel(event.get("title", ""))
        title.setWordWrap(True)
        title.setStyleSheet("color: #e6edf3; font-size: 12px;")
        layout.addWidget(title)
        detail = event.get("detail", "")
        if detail:
            snippet = QLabel(detail[:280])
            snippet.setWordWrap(True)
            snippet.setStyleSheet("color: #a7b2c0; font-size: 11px;")
            layout.addWidget(snippet)
        url = event.get("url", "")
        if url.startswith(("https://", "http://")):
            link = QPushButton(url[:140])
            link.setCursor(Qt.CursorShape.PointingHandCursor)
            link.setStyleSheet(
                "QPushButton { color: #71b8ff; border: 0; text-align: left; padding: 0; }"
                "QPushButton:hover { color: #acd5ff; text-decoration: underline; }"
            )
            link.clicked.connect(lambda _checked=False, value=url: QDesktopServices.openUrl(QUrl(value)))
            layout.addWidget(link)


class ResearchWorker(QThread):
    delta = Signal(str)
    resetDraft = Signal()
    researchEvent = Signal(dict)
    completed = Signal(str, int, int, float, str)
    failed = Signal(str, float, str)

    def __init__(
        self,
        *,
        config: AppConfig,
        history: list[dict[str, str]],
        model: str,
        store: DataStore,
        limit_store: LimitStore,
    ) -> None:
        super().__init__()
        self.config = config
        self.history = history
        self.model = model
        self.store = store
        self.limit_store = limit_store

    def run(self) -> None:
        start = time.monotonic()
        day = ""
        reserved = False
        try:
            # Limits are re-read for every request so edits made in UsageLimitEditor
            # take effect without restarting the chat application.
            limits = self.limit_store.load()
            estimated_text = SYSTEM_PROMPT + json.dumps(self.history[-14:], ensure_ascii=False)
            estimated_input = estimate_tokens(estimated_text) + 180
            snapshot = self.store.get_usage()
            decision = self.store.reserve_request(limits, estimated_input)
            day = decision.day
            if not decision.allowed:
                self.failed.emit(decision.reason, 0.0, day)
                return
            reserved = True
            available = max(1, limits.daily_token_limit - snapshot.total_tokens)
            client = OllamaClient(
                self.config.ollama_url,
                timeout_seconds=self.config.request_timeout_seconds,
                retry_count=self.config.retry_count,
            )
            web = WebSearchClient(
                endpoint=self.config.search_endpoint,
                timeout_seconds=self.config.search_timeout_seconds,
                retry_count=min(self.config.retry_count, 2),
            )
            agent = ResearchAgent(client, web)
            outcome = agent.respond(
                history=self.history,
                model=self.model,
                config=self.config,
                limits=limits,
                store=self.store,
                usage_day=day,
                num_predict=self.config.max_output_tokens,
                remaining_token_budget=available,
                on_delta=self.delta.emit,
                on_reset=self.resetDraft.emit,
                on_event=self.researchEvent.emit,
                on_usage=lambda incoming, outgoing: self.store.record_tokens(incoming, outgoing, day),
            )
            elapsed = time.monotonic() - start
            self.completed.emit(
                outcome.answer, outcome.input_tokens, outcome.output_tokens, elapsed, day
            )
        except (LimitStoreError, ConfigError, OllamaError, QuotaExceeded, ValueError) as exc:
            elapsed = time.monotonic() - start
            self.failed.emit(str(exc), elapsed if reserved else 0.0, day)
        except Exception as exc:  # noqa: BLE001 - surface unexpected worker failures in the UI.
            elapsed = time.monotonic() - start
            self.failed.emit(f"Beklenmeyen hata: {exc}", elapsed if reserved else 0.0, day)


class ModelsWorker(QThread):
    modelsLoaded = Signal(list)
    loadFailed = Signal(str)

    def __init__(self, base_url: str, timeout: int, retries: int) -> None:
        super().__init__()
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries

    def run(self) -> None:
        try:
            models = OllamaClient(self.base_url, self.timeout, self.retries).list_models()
            self.modelsLoaded.emit(models)
        except (OllamaError, ValueError) as exc:
            self.loadFailed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - prevent an unhandled QThread exception.
            self.loadFailed.emit(f"Beklenmeyen bağlantı hatası: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1510, 940)
        self.setMinimumSize(1100, 700)
        self.data_dir = get_data_dir()
        self.store = DataStore(self.data_dir / "deepsearch.sqlite3")
        self.limit_store = LimitStore(self.data_dir)
        self.config = load_config(self.data_dir / "settings.json")
        self.limits = self.limit_store.load()
        self.current_session_id: str | None = None
        self._assistant_card: MessageCard | None = None
        self._streamed_text = ""
        self._worker: ResearchWorker | None = None
        self._models_worker: ModelsWorker | None = None
        self._activity_widgets: list[QWidget] = []
        self._pending_active_seconds = 0
        self._active_day = local_day()
        self._quota_integrity_ok = True

        self._build_ui()
        self._connect_signals()
        self._active_timer = QTimer(self)
        self._active_timer.setInterval(1000)
        self._active_timer.timeout.connect(self._tick_active_time)
        self._active_timer.start()
        self._sync_settings_widgets()
        self._refresh_usage()
        self._refresh_history()
        sessions = self.store.list_sessions(limit=1)
        if sessions:
            self._load_session(sessions[0]["session_id"])
        else:
            self._show_empty_state()
        QTimer.singleShot(250, self.refresh_models)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        self.sidebar = self._build_sidebar()
        self.sidebar.setFixedWidth(250)
        root_layout.addWidget(self.sidebar)

        center = self._build_center()
        root_layout.addWidget(center, 1)

        self.settings_panel = self._build_settings()
        self.settings_panel.setFixedWidth(302)
        root_layout.addWidget(self.settings_panel)

        self.setCentralWidget(root)
        self.statusBar().showMessage("Hazır")

    def _build_sidebar(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 15, 14, 14)
        layout.setSpacing(11)
        title = QLabel("DEEPSEARCH")
        title.setStyleSheet("font-size: 15px; font-weight: 800; color: #82c8ff; letter-spacing: 1px;")
        layout.addWidget(title)
        subtitle = QLabel("Yerel yapay zekâ · Web araştırması")
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self.new_chat_button = QPushButton("＋  Yeni sohbet")
        self.new_chat_button.setObjectName("primaryButton")
        layout.addWidget(self.new_chat_button)

        history_heading = QLabel("SOHBET GEÇMİŞİ")
        history_heading.setObjectName("sectionHeading")
        layout.addWidget(history_heading)
        self.history_list = QListWidget()
        self.history_list.setObjectName("historyList")
        self.history_list.setSpacing(3)
        layout.addWidget(self.history_list, 1)

        usage_heading = QLabel("GÜNLÜK KULLANIM")
        usage_heading.setObjectName("sectionHeading")
        layout.addWidget(usage_heading)
        self.requests_label = QLabel("İstekler: 0 / 40")
        layout.addWidget(self.requests_label)
        self.requests_bar = QProgressBar()
        self.requests_bar.setTextVisible(False)
        self.requests_bar.setFixedHeight(7)
        layout.addWidget(self.requests_bar)
        self.tokens_label = QLabel("Token: 0 / 100.000")
        layout.addWidget(self.tokens_label)
        self.token_split_label = QLabel("Giriş: 0 · Çıkış: 0")
        self.token_split_label.setObjectName("muted")
        layout.addWidget(self.token_split_label)
        self.tokens_bar = QProgressBar()
        self.tokens_bar.setTextVisible(False)
        self.tokens_bar.setFixedHeight(7)
        layout.addWidget(self.tokens_bar)
        self.searches_label = QLabel("Aramalar: 0 / 50")
        layout.addWidget(self.searches_label)
        self.searches_bar = QProgressBar()
        self.searches_bar.setTextVisible(False)
        self.searches_bar.setFixedHeight(7)
        layout.addWidget(self.searches_bar)
        self.active_label = QLabel("Aktif süre: 0 dk")
        self.active_label.setObjectName("muted")
        layout.addWidget(self.active_label)
        self.quota_status_label = QLabel("Kotalar etkin")
        self.quota_status_label.setObjectName("quotaStatus")
        layout.addWidget(self.quota_status_label)
        return panel

    def _build_center(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)
        header = QFrame()
        header.setObjectName("topPanel")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(15, 9, 15, 9)
        title_box = QVBoxLayout()
        header_title = QLabel("DeepSearch & Research AI Chat")
        header_title.setObjectName("centerTitle")
        title_box.addWidget(header_title)
        self.model_status = QLabel("Model yükleniyor…")
        self.model_status.setObjectName("muted")
        title_box.addWidget(self.model_status)
        header_layout.addLayout(title_box, 1)
        self.refresh_usage_button = QPushButton("↻")
        self.refresh_usage_button.setToolTip("Kota bilgilerini yenile")
        self.refresh_usage_button.setFixedWidth(40)
        header_layout.addWidget(self.refresh_usage_button)
        layout.addWidget(header)

        self.activity_frame = QFrame()
        self.activity_frame.setObjectName("activityFrame")
        activity_frame_layout = QVBoxLayout(self.activity_frame)
        activity_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.activity_scroll = QScrollArea()
        self.activity_scroll.setWidgetResizable(True)
        self.activity_scroll.setMaximumHeight(185)
        self.activity_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.activity_content = QWidget()
        self.activity_layout = QVBoxLayout(self.activity_content)
        self.activity_layout.setContentsMargins(0, 0, 0, 0)
        self.activity_layout.setSpacing(5)
        self.activity_scroll.setWidget(self.activity_content)
        activity_frame_layout.addWidget(self.activity_scroll)
        self.activity_frame.hide()
        layout.addWidget(self.activity_frame)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.chat_scroll.setObjectName("chatScroll")
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setContentsMargins(3, 8, 5, 8)
        self.messages_layout.setSpacing(12)
        self.messages_layout.addStretch(1)
        self.chat_scroll.setWidget(self.messages_widget)
        layout.addWidget(self.chat_scroll, 1)

        composer = QFrame()
        composer.setObjectName("composer")
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(11, 9, 11, 10)
        composer_layout.setSpacing(7)
        self.prompt_input = ChatInput()
        self.prompt_input.setObjectName("promptInput")
        self.prompt_input.setPlaceholderText(
            "Bir soru sorun… Örn. ‘En güncel Qwen ve Gemini modellerini kaynaklarıyla karşılaştır.’"
        )
        self.prompt_input.setMinimumHeight(84)
        self.prompt_input.setMaximumHeight(150)
        composer_layout.addWidget(self.prompt_input)
        controls = QHBoxLayout()
        self.attach_file_button = QPushButton("＋ Dosya")
        self.attach_link_button = QPushButton("🔗 Bağlantı")
        controls.addWidget(self.attach_file_button)
        controls.addWidget(self.attach_link_button)
        self.web_toggle = QCheckBox("İnternette araştır")
        controls.addWidget(self.web_toggle)
        controls.addStretch(1)
        tip = QLabel("Ctrl + Enter gönderir")
        tip.setObjectName("muted")
        controls.addWidget(tip)
        self.send_button = QPushButton("Araştır ve yanıtla  ➤")
        self.send_button.setObjectName("primaryButton")
        controls.addWidget(self.send_button)
        composer_layout.addLayout(controls)
        layout.addWidget(composer)
        return panel

    def _build_settings(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings_content = QWidget()
        layout = QVBoxLayout(settings_content)
        layout.setContentsMargins(14, 15, 14, 14)
        layout.setSpacing(8)
        heading = QLabel("AYARLAR")
        heading.setObjectName("sectionHeading")
        layout.addWidget(heading)

        layout.addWidget(self._field_label("Ollama sunucu adresi"))
        self.ollama_url_input = QLineEdit()
        self.ollama_url_input.setPlaceholderText("http://localhost:11435")
        layout.addWidget(self.ollama_url_input)
        warning = QLabel("Uzak bir Ollama adresi seçerseniz sohbet metni o sunucuya gönderilir.")
        warning.setObjectName("muted")
        warning.setWordWrap(True)
        layout.addWidget(warning)

        layout.addWidget(self._field_label("Yerel Ollama modeli"))
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(False)
        self.model_combo.setMinimumWidth(130)
        model_row.addWidget(self.model_combo, 1)
        self.refresh_models_button = QPushButton("Yenile")
        model_row.addWidget(self.refresh_models_button)
        layout.addLayout(model_row)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName("separator")
        layout.addWidget(line)
        search_heading = QLabel("WEB ARAŞTIRMASI")
        search_heading.setObjectName("sectionHeading")
        layout.addWidget(search_heading)
        self.settings_web_checkbox = QCheckBox("Arama varsayılan olarak açık")
        layout.addWidget(self.settings_web_checkbox)
        layout.addWidget(self._field_label("DuckDuckGo HTML uç noktası"))
        self.search_endpoint_input = QLineEdit()
        layout.addWidget(self.search_endpoint_input)
        layout.addWidget(self._field_label("Arama zaman aşımı (saniye)"))
        self.search_timeout_spin = self._spin(3, 60)
        layout.addWidget(self.search_timeout_spin)
        layout.addWidget(self._field_label("Sonuç sayısı"))
        self.result_count_spin = self._spin(1, 12)
        layout.addWidget(self.result_count_spin)
        layout.addWidget(self._field_label("Araştırma derinliği (sayfa)"))
        self.search_depth_spin = self._spin(1, 5)
        layout.addWidget(self.search_depth_spin)

        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setObjectName("separator")
        layout.addWidget(line2)
        advanced = QLabel("OLLAMA / AKIŞ")
        advanced.setObjectName("sectionHeading")
        layout.addWidget(advanced)
        layout.addWidget(self._field_label("İstek zaman aşımı (saniye)"))
        self.request_timeout_spin = self._spin(30, 900)
        layout.addWidget(self.request_timeout_spin)
        layout.addWidget(self._field_label("Maksimum yanıt tokenı"))
        self.max_output_spin = self._spin(64, 8192)
        layout.addWidget(self.max_output_spin)
        layout.addWidget(self._field_label("Yeniden deneme"))
        self.retry_spin = self._spin(0, 5)
        layout.addWidget(self.retry_spin)

        self.settings_status = QLabel("Ayarlar yükleniyor")
        self.settings_status.setObjectName("muted")
        self.settings_status.setWordWrap(True)
        layout.addWidget(self.settings_status)
        self.save_settings_button = QPushButton("Ayarları kaydet")
        self.save_settings_button.setObjectName("primaryButton")
        layout.addWidget(self.save_settings_button)
        data_path = QLabel(f"Yerel veriler: {get_data_dir()}")
        data_path.setObjectName("muted")
        data_path.setWordWrap(True)
        data_path.setToolTip(str(get_data_dir()))
        layout.addWidget(data_path)
        layout.addStretch(1)
        settings_scroll.setWidget(settings_content)
        panel_layout.addWidget(settings_scroll)
        return panel

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    @staticmethod
    def _spin(minimum: int, maximum: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        return spin

    def _connect_signals(self) -> None:
        self.new_chat_button.clicked.connect(self.new_chat)
        self.history_list.itemClicked.connect(self._on_history_clicked)
        self.send_button.clicked.connect(self.send_message)
        self.prompt_input.submitRequested.connect(self.send_message)
        self.refresh_models_button.clicked.connect(self.refresh_models)
        self.save_settings_button.clicked.connect(self.save_settings)
        self.web_toggle.toggled.connect(self._set_web_search)
        self.settings_web_checkbox.toggled.connect(self._set_web_search)
        self.attach_file_button.clicked.connect(self.attach_file)
        self.attach_link_button.clicked.connect(self.attach_link)
        self.refresh_usage_button.clicked.connect(self._refresh_usage)

    def _sync_settings_widgets(self) -> None:
        self.ollama_url_input.setText(self.config.ollama_url)
        self.search_endpoint_input.setText(self.config.search_endpoint)
        self.search_timeout_spin.setValue(self.config.search_timeout_seconds)
        self.result_count_spin.setValue(self.config.search_result_count)
        self.search_depth_spin.setValue(self.config.search_depth)
        self.request_timeout_spin.setValue(self.config.request_timeout_seconds)
        self.max_output_spin.setValue(self.config.max_output_tokens)
        self.retry_spin.setValue(self.config.retry_count)
        self.web_toggle.setChecked(self.config.web_search_enabled)
        self.settings_web_checkbox.setChecked(self.config.web_search_enabled)

    def _set_web_search(self, enabled: bool) -> None:
        # Keep the quick control and settings control in sync without recursively saving.
        for checkbox in (self.web_toggle, self.settings_web_checkbox):
            if checkbox.isChecked() != enabled:
                checkbox.blockSignals(True)
                checkbox.setChecked(enabled)
                checkbox.blockSignals(False)
        self.config = AppConfig(
            **{**self.config.__dict__, "web_search_enabled": enabled}
        )
        self.settings_status.setText("Web arama bu sohbet için açık." if enabled else "Web arama kapalı.")

    def save_settings(self) -> None:
        try:
            config = AppConfig(
                ollama_url=self.ollama_url_input.text().strip(),
                web_search_enabled=self.settings_web_checkbox.isChecked(),
                search_endpoint=self.search_endpoint_input.text().strip(),
                search_timeout_seconds=self.search_timeout_spin.value(),
                search_result_count=self.result_count_spin.value(),
                search_depth=self.search_depth_spin.value(),
                request_timeout_seconds=self.request_timeout_spin.value(),
                max_output_tokens=self.max_output_spin.value(),
                retry_count=self.retry_spin.value(),
            ).validate()
            save_config(config, self.data_dir / "settings.json")
        except (ConfigError, OSError) as exc:
            QMessageBox.warning(self, "Ayarlar kaydedilemedi", str(exc))
            return
        self.config = config
        self.settings_status.setText("Ayarlar kaydedildi.")
        self.statusBar().showMessage("Ayarlar kaydedildi", 3500)
        self.refresh_models()

    def refresh_models(self) -> None:
        base_url = self.ollama_url_input.text().strip() or self.config.ollama_url
        if self._models_worker and self._models_worker.isRunning():
            return
        self.model_status.setText(f"{base_url} adresindeki modeller aranıyor…")
        self.refresh_models_button.setEnabled(False)
        worker = ModelsWorker(
            base_url,
            self.request_timeout_spin.value() or self.config.request_timeout_seconds,
            self.retry_spin.value(),
        )
        self._models_worker = worker
        worker.modelsLoaded.connect(self._on_models_loaded)
        worker.loadFailed.connect(self._on_models_failed)
        worker.finished.connect(lambda current=worker: self._models_thread_finished(current))
        worker.start()

    def _models_thread_finished(self, worker: ModelsWorker) -> None:
        if self._models_worker is worker:
            self._models_worker = None
        self.refresh_models_button.setEnabled(True)
        worker.deleteLater()

    def _on_models_loaded(self, models: list[dict[str, Any]]) -> None:
        current = self.model_combo.currentText()
        self.model_combo.clear()
        for model in models:
            name = model.get("name", "")
            if name:
                self.model_combo.addItem(str(name))
        if current:
            index = self.model_combo.findText(current)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        if models:
            self.model_status.setText(f"{len(models)} yerel model bulundu · {self.model_combo.currentText()}")
            self.settings_status.setText("Ollama bağlantısı başarılı.")
        else:
            self.model_status.setText("Ollama bağlı; yüklü model bulunamadı.")
            self.settings_status.setText("Ollama'ya model indirin: ollama pull qwen3:8b")

    def _on_models_failed(self, message: str) -> None:
        self.model_status.setText("Ollama bağlantısı kurulamadı.")
        self.settings_status.setText(message)

    def new_chat(self) -> None:
        self.current_session_id = None
        self.history_list.clearSelection()
        self._clear_messages()
        self._clear_activity()
        self._show_empty_state()
        self.prompt_input.setFocus()
        self.statusBar().showMessage("Yeni sohbet hazır", 2500)

    def _show_empty_state(self) -> None:
        if self.messages_layout.count() > 1:
            return
        empty = QLabel(
            "Bir araştırma sorusu yazın.\n\n"
            "Web araştırması açıksa kaynaklar aranır ve sayfalar okunur; yanıtlar seçtiğiniz "
            "Ollama modeli tarafından yerel olarak üretilir."
        )
        empty.setWordWrap(True)
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.setObjectName("emptyState")
        empty.setMinimumHeight(190)
        self._insert_message_widget(empty)

    def _clear_messages(self) -> None:
        self._assistant_card = None
        self._streamed_text = ""
        while self.messages_layout.count():
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.messages_layout.addStretch(1)

    def _clear_activity(self) -> None:
        while self.activity_layout.count():
            item = self.activity_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._activity_widgets.clear()
        self.activity_frame.hide()

    def _insert_message_widget(self, widget: QWidget) -> None:
        index = max(0, self.messages_layout.count() - 1)
        self.messages_layout.insertWidget(index, widget)
        QTimer.singleShot(40, self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        bar = self.chat_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _refresh_history(self) -> None:
        selected = self.current_session_id
        self.history_list.blockSignals(True)
        self.history_list.clear()
        for session in self.store.list_sessions(limit=100):
            item = QListWidgetItem(session["title"] or "Yeni sohbet")
            item.setData(Qt.ItemDataRole.UserRole, session["session_id"])
            item.setToolTip(session["updated_at"])
            self.history_list.addItem(item)
            if session["session_id"] == selected:
                self.history_list.setCurrentItem(item)
        self.history_list.blockSignals(False)

    def _on_history_clicked(self, item: QListWidgetItem) -> None:
        session_id = item.data(Qt.ItemDataRole.UserRole)
        if session_id:
            self._load_session(str(session_id))

    def _load_session(self, session_id: str) -> None:
        self.current_session_id = session_id
        self._clear_messages()
        self._clear_activity()
        messages = self.store.get_messages(session_id)
        if not messages:
            self._show_empty_state()
        for message in messages:
            self._insert_message_widget(MessageCard(message["role"], message["content"]))
        self._refresh_history()
        self._refresh_usage()

    def send_message(self) -> None:
        text = self.prompt_input.toPlainText().strip()
        if not text:
            return
        if self._worker and self._worker.isRunning():
            return
        model = self.model_combo.currentText().strip()
        if not model:
            QMessageBox.information(
                self,
                "Model seçilmedi",
                "Önce Ollama sunucusuna bağlanıp Ayarlar panelinden yüklü bir model seçin.",
            )
            return
        self.config = AppConfig(
            **{**self.config.__dict__, "web_search_enabled": self.web_toggle.isChecked()}
        )
        if self.current_session_id is None:
            self._clear_messages()
            self.current_session_id = self.store.create_session()
        self.store.save_message(self.current_session_id, "user", text)
        self._insert_message_widget(MessageCard("user", text))
        self.prompt_input.clear()
        self._refresh_history()
        self._clear_activity()
        self._streamed_text = ""
        self._assistant_card = MessageCard("assistant", "Yanıt hazırlanıyor…", streaming=True)
        self._insert_message_widget(self._assistant_card)
        self.send_button.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self.new_chat_button.setEnabled(False)
        self.history_list.setEnabled(False)
        self.send_button.setText("Araştırılıyor…")
        self.statusBar().showMessage("Kota kontrolü ve araştırma başlatılıyor…")
        history = self.store.get_messages(self.current_session_id)
        worker = ResearchWorker(
            config=self.config,
            history=history,
            model=model,
            store=self.store,
            limit_store=self.limit_store,
        )
        self._worker = worker
        worker.delta.connect(self._on_delta)
        worker.resetDraft.connect(self._on_reset_draft)
        worker.researchEvent.connect(self._on_research_event)
        worker.completed.connect(self._on_completed)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(lambda current=worker: self._research_thread_finished(current))
        worker.start()

    def _on_delta(self, text: str) -> None:
        self._streamed_text += text
        if self._assistant_card is not None:
            self._assistant_card.set_content(self._streamed_text, markdown=False)
        self._scroll_to_bottom()

    def _on_reset_draft(self) -> None:
        self._streamed_text = ""
        if self._assistant_card is not None:
            self._assistant_card.set_content("Araçlar çalıştırılıyor…", markdown=False)

    def _on_research_event(self, event: dict[str, str]) -> None:
        card = SearchEventCard(event)
        self._activity_widgets.append(card)
        self.activity_layout.addWidget(card)
        while len(self._activity_widgets) > 5:
            old = self._activity_widgets.pop(0)
            self.activity_layout.removeWidget(old)
            old.deleteLater()
        self.activity_frame.show()
        QTimer.singleShot(
            30,
            lambda: self.activity_scroll.verticalScrollBar().setValue(
                self.activity_scroll.verticalScrollBar().maximum()
            ),
        )

    def _on_completed(
        self, answer: str, input_tokens: int, output_tokens: int, elapsed: float, day: str
    ) -> None:
        if self._assistant_card is not None:
            self._assistant_card.streaming = False
            self._assistant_card.set_content(answer, markdown=True)
        if self.current_session_id:
            self.store.save_message(self.current_session_id, "assistant", answer)
        self._refresh_history()
        self._refresh_usage()
        self.statusBar().showMessage(
            f"Yanıt tamamlandı · {input_tokens + output_tokens:,} token · {elapsed:.1f} sn".replace(
                ",", "."
            ),
            7000,
        )

    def _on_failed(self, message: str, elapsed: float, day: str) -> None:
        if self._assistant_card is not None:
            self._assistant_card.streaming = False
            self._assistant_card.set_content(f"⚠️ {message}", markdown=False)
        self.statusBar().showMessage(message, 10_000)
        self._refresh_usage()

    def _research_thread_finished(self, worker: ResearchWorker) -> None:
        if self._worker is worker:
            self._worker = None
        self.send_button.setEnabled(True)
        self.prompt_input.setEnabled(True)
        self.new_chat_button.setEnabled(True)
        self.history_list.setEnabled(True)
        self.send_button.setText("Araştır ve yanıtla  ➤")
        self.prompt_input.setFocus()
        self._refresh_usage()
        worker.deleteLater()

    def _refresh_usage(self) -> None:
        try:
            self.limits = self.limit_store.load()
            usage = self.store.get_usage()
        except (LimitStoreError, OSError) as exc:
            self._quota_integrity_ok = False
            self.send_button.setEnabled(False)
            self.prompt_input.setEnabled(False)
            self.quota_status_label.setText("Kota dosyası hatalı — uygulama kilitli")
            self.quota_status_label.setStyleSheet("color: #ff7777; font-weight: 700;")
            self.statusBar().showMessage(str(exc), 10_000)
            return
        self._quota_integrity_ok = True
        req_percent = min(100, int(usage.requests * 100 / self.limits.daily_requests))
        token_percent = min(100, int(usage.total_tokens * 100 / self.limits.daily_token_limit))
        search_percent = min(100, int(usage.searches * 100 / self.limits.daily_search_limit))
        self.requests_label.setText(f"İstekler: {usage.requests} / {self.limits.daily_requests}")
        self.tokens_label.setText(
            f"Token: {usage.total_tokens:,} / {self.limits.daily_token_limit:,}".replace(",", ".")
        )
        self.token_split_label.setText(
            f"Giriş: {usage.input_tokens:,} · Çıkış: {usage.output_tokens:,}".replace(",", ".")
        )
        self.searches_label.setText(f"Aramalar: {usage.searches} / {self.limits.daily_search_limit}")
        self.requests_bar.setValue(req_percent)
        self.tokens_bar.setValue(token_percent)
        self.searches_bar.setValue(search_percent)
        current_pending = self._pending_active_seconds if usage.day == self._active_day else 0
        minutes, seconds = divmod(usage.active_seconds + current_pending, 60)
        self.active_label.setText(f"Aktif süre: {minutes} dk {seconds} sn")
        at_limit = (
            usage.requests >= self.limits.daily_requests
            or usage.total_tokens >= self.limits.daily_token_limit
            or usage.searches >= self.limits.daily_search_limit
        )
        if at_limit:
            self.quota_status_label.setText("Günlük kota doldu — yarın yeniden açılır")
            self.quota_status_label.setStyleSheet("color: #ff9d6c; font-weight: 700;")
        else:
            suffix = " · geliştirici modu" if self.limits.developer_mode else ""
            self.quota_status_label.setText(f"Kotalar etkin{suffix}")
            self.quota_status_label.setStyleSheet("color: #8fdaa7; font-weight: 700;")
        if not self._worker or not self._worker.isRunning():
            can_send = self._quota_integrity_ok and not at_limit
            self.send_button.setEnabled(can_send)
            self.prompt_input.setEnabled(can_send)

    def _tick_active_time(self) -> None:
        today = local_day()
        if today != self._active_day:
            self._flush_active_time()
            self._active_day = today
            self._pending_active_seconds = 0
            self._refresh_usage()
        if not self.isVisible() or not self.isActiveWindow():
            return
        self._pending_active_seconds += 1
        if self._pending_active_seconds >= 15:
            self._flush_active_time()
        self._update_active_label()

    def _flush_active_time(self) -> None:
        if self._pending_active_seconds <= 0:
            return
        try:
            self.store.record_active_seconds(self._pending_active_seconds, self._active_day)
        except (OSError, sqlite3.Error) as exc:
            self.statusBar().showMessage(f"Aktif süre kaydedilemedi: {exc}", 5000)
            return
        self._pending_active_seconds = 0

    def _update_active_label(self) -> None:
        usage = self.store.get_usage(self._active_day)
        total = usage.active_seconds + self._pending_active_seconds
        minutes, seconds = divmod(total, 60)
        self.active_label.setText(f"Aktif süre: {minutes} dk {seconds} sn")

    def closeEvent(self, event) -> None:
        research_running = self._worker and self._worker.isRunning()
        model_query_running = self._models_worker and self._models_worker.isRunning()
        if research_running or model_query_running:
            QMessageBox.information(
                self,
                "İşlem devam ediyor",
                "Arka plan işlemi tamamlanınca uygulamayı kapatın; bu, yarım kayıtları önler.",
            )
            event.ignore()
            return
        if hasattr(self, "_active_timer"):
            self._active_timer.stop()
        self._flush_active_time()
        super().closeEvent(event)

    def attach_file(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Metin dosyası ekle",
            str(Path.home()),
            "Metin dosyaları (*.txt *.md *.py *.json *.csv *.log);;Tüm dosyalar (*)",
        )
        if not file_name:
            return
        path = Path(file_name)
        try:
            raw = path.read_bytes()
            if len(raw) > 65_536:
                QMessageBox.warning(self, "Dosya büyük", "Ek başına en fazla 64 KiB metin kabul edilir.")
                return
            content = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            QMessageBox.warning(self, "Dosya okunamadı", f"UTF-8 metin dosyası seçin.\n{exc}")
            return
        current = self.prompt_input.toPlainText().rstrip()
        attached = f"\n\n[Dosya: {path.name}]\n```text\n{content}\n```"
        self.prompt_input.setPlainText((current + attached).strip())
        self.prompt_input.setFocus()

    def attach_link(self) -> None:
        link, accepted = QInputDialog.getText(
            self, "Bağlantı ekle", "İncelenecek http(s) URL:"
        )
        if not accepted or not link.strip():
            return
        current = self.prompt_input.toPlainText().rstrip()
        self.prompt_input.setPlainText(
            (current + f"\n\nİncelenecek bağlantı: {link.strip()}").strip()
        )
        self.prompt_input.setFocus()


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#101217"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e5e7eb"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#141820"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1b202a"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#202733"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#f3f4f6"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e5e7eb"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#202631"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e5e7eb"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ff6b6b"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2364a5"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(
        "QWidget#root { background: #101217; }"
        "QFrame#sidePanel, QFrame#topPanel, QFrame#composer { background: #151820; "
        "border: 1px solid #282d38; border-radius: 12px; }"
        "QLabel#centerTitle { font-size: 16px; font-weight: 700; color: #f4f6f8; }"
        "QLabel#muted { color: #8f9baa; font-size: 11px; }"
        "QLabel#sectionHeading { color: #91a0b3; font-size: 10px; font-weight: 800; "
        "letter-spacing: 1px; margin-top: 6px; }"
        "QLabel#fieldLabel { color: #c7d0dc; font-size: 11px; margin-top: 3px; }"
        "QLabel#messageHeading { color: #9db4cb; font-size: 11px; font-weight: 700; }"
        "QLabel#emptyState { color: #8f9baa; font-size: 14px; padding: 26px; }"
        "QLineEdit, QComboBox, QSpinBox, QPlainTextEdit { background: #10141b; "
        "border: 1px solid #333b49; border-radius: 7px; padding: 7px; color: #e6eaf0; }"
        "QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus { "
        "border: 1px solid #438dc9; }"
        "QComboBox QAbstractItemView { background: #171c25; selection-background-color: #285d8e; }"
        "QPushButton { background: #222936; border: 1px solid #343d4c; color: #dbe3ed; "
        "border-radius: 7px; padding: 7px 10px; }"
        "QPushButton:hover { background: #2b3544; border-color: #4d657c; }"
        "QPushButton:disabled { color: #687382; background: #191d25; }"
        "QPushButton#primaryButton { background: #2368a4; border-color: #347db8; "
        "color: white; font-weight: 700; padding: 8px 12px; }"
        "QPushButton#primaryButton:hover { background: #2f7cb9; }"
        "QListWidget { background: transparent; border: 0; outline: 0; color: #ccd5df; }"
        "QListWidget::item { border-radius: 6px; padding: 8px 7px; }"
        "QListWidget::item:selected { background: #253449; color: #fff; }"
        "QListWidget::item:hover { background: #1d2632; }"
        "QProgressBar { background: #252b35; border: 0; border-radius: 3px; }"
        "QProgressBar::chunk { background: #41a4d7; border-radius: 3px; }"
        "QCheckBox { color: #c8d1dc; spacing: 7px; }"
        "QCheckBox::indicator { width: 15px; height: 15px; }"
        "QFrame#separator { color: #303743; max-height: 1px; }"
        "QScrollArea { background: transparent; }"
        "QStatusBar { background: #12151c; color: #aab3bf; }"
        "QToolTip { background: #202733; color: #f4f6f8; border: 1px solid #394453; }"
    )


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setOrganizationName("SearchLoN")
    apply_dark_theme(app)
    try:
        window = MainWindow()
    except (ConfigError, LimitStoreError, OSError) as exc:
        QMessageBox.critical(
            None,
            "DeepSearch başlatılamadı",
            f"Ayarlar veya kota dosyası açılamadı. Güvenlik nedeniyle uygulama başlatılmadı.\n\n{exc}",
        )
        return 2
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
