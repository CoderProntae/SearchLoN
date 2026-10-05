"""Standalone GUI editor for encrypted per-user usage limits."""

from __future__ import annotations

import sys

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from deepsearch.config import get_data_dir
from deepsearch.limits import LimitStore, LimitStoreError, UsageLimits


class UsageLimitEditor(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Usage Limit Editor · DeepSearch")
        self.setMinimumWidth(440)
        self.store = LimitStore(get_data_dir())
        self._build_ui()
        self._load_current()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        heading = QLabel("Kullanım Kota Yöneticisi")
        heading.setStyleSheet("font-size: 19px; font-weight: 700; color: #85c8ff;")
        layout.addWidget(heading)
        subtitle = QLabel("Ana uygulama bu sınırları her istekte yeniden okur.")
        subtitle.setStyleSheet("color: #9aa6b5;")
        layout.addWidget(subtitle)

        form = QFormLayout()
        form.setVerticalSpacing(13)
        self.requests_spin = self._spin(1, 100_000)
        self.tokens_spin = self._spin(1, 10_000_000)
        self.searches_spin = self._spin(1, 100_000)
        form.addRow("Günlük istek üst sınırı", self.requests_spin)
        form.addRow("Günlük toplam token üst sınırı", self.tokens_spin)
        form.addRow("Günlük web araması üst sınırı", self.searches_spin)
        layout.addLayout(form)

        self.developer_checkbox = QCheckBox("Geliştirici modu (tanılama bilgilerini gösterir)")
        layout.addWidget(self.developer_checkbox)
        warning = QLabel(
            "Geliştirici modu kotaları devre dışı bırakmaz. Sınırlar her zaman etkindir.\n\n"
            "Ayarlar kullanıcı profilinde şifreli olarak saklanır. Yerel şifreleme anahtarı "
            "aynı işletim sistemi hesabına bağlıdır; bu araç kurumsal/uzak lisans denetimi veya "
            "aynı hesabın sahibine karşı aşılamaz koruma sağlamaz."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(
            "background: #241f16; color: #e8c982; border: 1px solid #594a2a; "
            "border-radius: 8px; padding: 10px;"
        )
        layout.addWidget(warning)
        self.path_label = QLabel(f"Veri dizini: {get_data_dir()}")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #8f9baa; font-size: 10px;")
        layout.addWidget(self.path_label)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.save_button = QPushButton("Kota ayarlarını kaydet")
        self.save_button.setStyleSheet(
            "QPushButton { background: #2368a4; color: white; border: 0; border-radius: 7px; "
            "padding: 10px; font-weight: 700; } QPushButton:hover { background: #2f7cb9; }"
        )
        self.save_button.clicked.connect(self._save)
        layout.addWidget(self.save_button)

    @staticmethod
    def _spin(minimum: int, maximum: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setGroupSeparatorShown(True)
        return spin

    def _load_current(self) -> None:
        try:
            limits = self.store.load()
        except LimitStoreError as exc:
            self.save_button.setEnabled(False)
            self.status.setText(f"Kota dosyası açılamadı: {exc}")
            self.status.setStyleSheet("color: #ff7777;")
            return
        self.requests_spin.setValue(limits.daily_requests)
        self.tokens_spin.setValue(limits.daily_token_limit)
        self.searches_spin.setValue(limits.daily_search_limit)
        self.developer_checkbox.setChecked(limits.developer_mode)
        self.status.setText("Mevcut kota ayarları yüklendi.")
        self.status.setStyleSheet("color: #8fdaa7;")

    def _save(self) -> None:
        limits = UsageLimits(
            daily_requests=self.requests_spin.value(),
            daily_token_limit=self.tokens_spin.value(),
            daily_search_limit=self.searches_spin.value(),
            developer_mode=self.developer_checkbox.isChecked(),
        )
        try:
            self.store.save(limits)
        except (LimitStoreError, OSError, ValueError) as exc:
            self.status.setText(f"Kaydedilemedi: {exc}")
            self.status.setStyleSheet("color: #ff7777;")
            return
        self.status.setText("Güvenli kota ayarları kaydedildi. Ana uygulamada hemen geçerli olur.")
        self.status.setStyleSheet("color: #8fdaa7;")
        QMessageBox.information(self, "Kaydedildi", "Limit ayarları ana uygulamaya uygulandı.")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("UsageLimitEditor")
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#101217"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e5e7eb"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#141820"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e5e7eb"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#202631"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e5e7eb"))
    app.setPalette(palette)
    app.setStyleSheet(
        "QWidget { background: #101217; color: #e5e7eb; }"
        "QSpinBox { background: #151a22; border: 1px solid #333b49; border-radius: 6px; padding: 6px; }"
        "QCheckBox { spacing: 8px; }"
    )
    try:
        window = UsageLimitEditor()
    except (LimitStoreError, OSError) as exc:
        QMessageBox.critical(None, "Usage Limit Editor", f"Başlatılamadı:\n{exc}")
        return 2
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
