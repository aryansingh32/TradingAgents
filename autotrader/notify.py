"""Notifications and remote control, as freqtrade's Telegram bot made standard.

Telegram: create a bot with @BotFather, put its token in TELEGRAM_BOT_TOKEN
and your chat id in TELEGRAM_CHAT_ID. Commands are accepted only from that
chat: /status, /pause, /resume, /panic (close everything and pause), /daily,
/help. Discord: put a channel webhook URL in DISCORD_WEBHOOK_URL
(notifications only).
"""

from __future__ import annotations

import logging
import os

import requests

from autotrader.config import NotifyConfig

log = logging.getLogger(__name__)

COMMANDS = ("status", "pause", "resume", "panic", "daily", "help")


class Notifier:
    timeout = 8

    def __init__(self, cfg: NotifyConfig, session: requests.Session | None = None):
        self.cfg = cfg
        self.session = session or requests.Session()
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")
        self._offset = None

    @property
    def telegram_on(self) -> bool:
        return self.cfg.telegram and bool(self.token and self.chat_id)

    @property
    def discord_on(self) -> bool:
        return self.cfg.discord and bool(self.webhook)

    def send(self, event: str, text: str) -> None:
        if event not in self.cfg.events:
            return
        self.say(text)

    def say(self, text: str) -> None:
        text = text[:3900]
        if self.telegram_on:
            try:
                self.session.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                                  json={"chat_id": self.chat_id, "text": text}, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("telegram send failed: %s", exc)
        if self.discord_on:
            try:
                self.session.post(self.webhook, json={"content": text[:1900]}, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("discord send failed: %s", exc)

    def poll_commands(self) -> list[str]:
        """New commands from the configured chat since the last poll."""
        if not (self.telegram_on and self.cfg.telegram_commands):
            return []
        params = {"timeout": 0, "allowed_updates": '["message"]'}
        if self._offset is not None:
            params["offset"] = self._offset
        try:
            data = self.session.get(f"https://api.telegram.org/bot{self.token}/getUpdates",
                                    params=params, timeout=self.timeout).json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("telegram poll failed: %s", exc)
            return []
        out = []
        for update in data.get("result", []):
            self._offset = update["update_id"] + 1
            msg = update.get("message") or {}
            if str((msg.get("chat") or {}).get("id")) != str(self.chat_id):
                continue  # only the owner's chat may control the bot
            text = (msg.get("text") or "").strip().lower()
            if text.startswith("/"):
                cmd = text[1:].split("@")[0].split()[0] if len(text) > 1 else ""
                if cmd in COMMANDS:
                    out.append(cmd)
        return out
