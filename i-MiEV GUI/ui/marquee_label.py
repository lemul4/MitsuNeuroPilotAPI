from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer, QRect
from PySide6.QtGui import QPainter, QPalette
from PySide6.QtWidgets import QLabel, QPushButton, QStyle, QStyleOptionButton


class _SmoothMarqueeMixin:
    """Pixel-based marquee helper used by labels and buttons.

    The previous implementation changed the visible text every tick. That made
    Qt recompute text layout and could make compact panels visually jerk while
    the string was scrolling. This helper keeps the stored text unchanged and
    only repaints it with a sub-pixel/pixel offset.
    """

    _marquee_speed_px_s = 32.0
    _marquee_gap_px = 44
    _marquee_timer_ms = 16
    _marquee_start_pause_s = 0.75

    def _marquee_init(self, text: str = "") -> None:
        self._marquee_text = str(text or "")
        self._marquee_offset_px = 0.0
        self._marquee_last_t = time.monotonic()
        self._marquee_pause_until = self._marquee_last_t + self._marquee_start_pause_s
        self._marquee_timer = QTimer(self)
        self._marquee_timer.setInterval(self._marquee_timer_ms)
        self._marquee_timer.timeout.connect(self._marquee_tick)
        self.setToolTip(self._marquee_text)

    def _marquee_text_width(self) -> int:
        try:
            return int(self.fontMetrics().horizontalAdvance(self._marquee_text))
        except Exception:
            return 0

    def _marquee_available_width(self, rect: QRect | None = None) -> int:
        if rect is None:
            rect = self.contentsRect()
        return max(0, int(rect.width()) - 4)

    def _marquee_overflows(self, rect: QRect | None = None) -> bool:
        if not self._marquee_text:
            return False
        return self._marquee_text_width() > self._marquee_available_width(rect)

    def _marquee_reset(self) -> None:
        now = time.monotonic()
        self._marquee_offset_px = 0.0
        self._marquee_last_t = now
        self._marquee_pause_until = now + self._marquee_start_pause_s

    def _marquee_sync_timer(self) -> None:
        try:
            if self._marquee_overflows() and self.isVisible():
                if not self._marquee_timer.isActive():
                    self._marquee_last_t = time.monotonic()
                    self._marquee_timer.start()
            else:
                if self._marquee_timer.isActive():
                    self._marquee_timer.stop()
                self._marquee_offset_px = 0.0
        except RuntimeError:
            pass

    def _marquee_tick(self) -> None:
        if not self._marquee_overflows():
            self._marquee_sync_timer()
            self.update()
            return

        now = time.monotonic()
        dt = max(0.0, min(0.08, now - self._marquee_last_t))
        self._marquee_last_t = now

        if now >= self._marquee_pause_until:
            self._marquee_offset_px += self._marquee_speed_px_s * dt
            cycle = max(1, self._marquee_text_width() + self._marquee_gap_px)
            if self._marquee_offset_px >= cycle:
                self._marquee_offset_px = 0.0
                self._marquee_pause_until = now + self._marquee_start_pause_s

        self.update()

    def _marquee_draw(self, painter: QPainter, rect: QRect, color, alignment: Qt.AlignmentFlag | Qt.Alignment = Qt.AlignVCenter | Qt.AlignLeft) -> None:
        text = self._marquee_text
        if not text:
            return

        painter.save()
        painter.setPen(color)
        painter.setFont(self.font())
        painter.setClipRect(rect)

        if not self._marquee_overflows(rect):
            painter.drawText(rect, int(alignment), text)
            painter.restore()
            return

        fm = self.fontMetrics()
        text_w = fm.horizontalAdvance(text)
        y_rect = QRect(rect.left(), rect.top(), rect.width(), rect.height())
        x = int(rect.left() - self._marquee_offset_px)
        painter.drawText(QRect(x, y_rect.top(), text_w + 8, y_rect.height()), int(Qt.AlignVCenter | Qt.AlignLeft), text)
        painter.drawText(QRect(x + text_w + self._marquee_gap_px, y_rect.top(), text_w + 8, y_rect.height()), int(Qt.AlignVCenter | Qt.AlignLeft), text)
        painter.restore()


class ScrollingLabel(QLabel, _SmoothMarqueeMixin):
    """Single-line QLabel with smooth pixel scrolling when text overflows."""

    def __init__(self, text: str = "", parent=None, interval_ms: int | None = None, step_chars: int | None = None):
        super().__init__("", parent)
        self._marquee_init(text)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setWordWrap(False)
        if interval_ms is not None:
            self._marquee_timer.setInterval(max(12, int(interval_ms)))
        if step_chars is not None:
            # Kept for backward compatibility with the old constructor. Pixel
            # speed is smoother than character stepping; this maps old values to
            # a reasonable pixel speed without changing call sites.
            self._marquee_speed_px_s = max(18.0, float(step_chars) * 32.0)

    def setText(self, text):  # noqa: N802 - Qt API
        self._marquee_text = str(text or "")
        self.setToolTip(self._marquee_text)
        self._marquee_reset()
        self._marquee_sync_timer()
        self.update()

    def text(self):  # noqa: N802 - Qt API
        return self._marquee_text

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._marquee_sync_timer()

    def showEvent(self, event):
        super().showEvent(event)
        self._marquee_sync_timer()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._marquee_sync_timer()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.contentsRect().adjusted(2, 0, -2, 0)
        color = self.palette().color(self.foregroundRole())
        alignment = self.alignment() or (Qt.AlignVCenter | Qt.AlignLeft)
        self._marquee_draw(painter, rect, color, alignment)

    def sizeHint(self):
        hint = super().sizeHint()
        # Keep sizeHint bounded so a long label cannot stretch the Navigator row.
        hint.setWidth(min(max(80, hint.width()), 220))
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(40)
        return hint


class MarqueeButton(QPushButton, _SmoothMarqueeMixin):
    """QPushButton with smooth scrolling text when the caption is too long."""

    def __init__(self, text: str = "", parent=None):
        super().__init__("", parent)
        self._marquee_init(text)
        self.setText(text)

    def setText(self, text):  # noqa: N802 - Qt API
        self._marquee_text = str(text or "")
        self.setToolTip(self._marquee_text)
        self._marquee_reset()
        self._marquee_sync_timer()
        self.update()

    def text(self):  # noqa: N802 - Qt API
        return self._marquee_text

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._marquee_sync_timer()

    def showEvent(self, event):
        super().showEvent(event)
        self._marquee_sync_timer()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._marquee_sync_timer()

    def paintEvent(self, event):
        painter = QPainter(self)
        option = QStyleOptionButton()
        option.initFrom(self)
        if self.isDown():
            option.state |= QStyle.State_Sunken
        if self.isChecked():
            option.state |= QStyle.State_On
        if self.isDefault():
            try:
                option.features |= QStyleOptionButton.ButtonFeature.DefaultButton
            except Exception:
                pass
        if self.menu() is not None:
            try:
                option.features |= QStyleOptionButton.ButtonFeature.HasMenu
            except Exception:
                pass
        option.text = ""
        self.style().drawControl(QStyle.CE_PushButton, option, painter, self)

        content = self.style().subElementRect(QStyle.SE_PushButtonContents, option, self)
        content = content.adjusted(8, 0, -8, 0)
        if self.menu() is not None:
            content.adjust(0, 0, -12, 0)

        group = QPalette.ColorGroup.Disabled if not self.isEnabled() else QPalette.ColorGroup.Active
        color = self.palette().color(group, QPalette.ColorRole.ButtonText)
        self._marquee_draw(painter, content, color, Qt.AlignCenter)

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(min(max(96, self.fontMetrics().horizontalAdvance(self._marquee_text) + 36), 190))
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(86)
        return hint
