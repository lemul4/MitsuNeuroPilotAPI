#!/usr/bin/env bash

# Используем прямой вызов через cmd.exe, чтобы не мучиться с путями Bash
CARLA_EXE="E:\основы программирования\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe"

echo "Запускаю CARLA через системный вызов Windows..."

# Команда запуска (используем кавычки для защиты пробелов)
cmd.exe /c "$CARLA_EXE" -quality-level=Poor -world-port=2000 -resx=800 -resy=600 -nosound