# Запуск

Поднимаем Neo4j:

```
docker compose up -d
```

Ставим uv:

Linux/Mac:

```
wget -qO- https://astral.sh/uv/install.sh | sh
```

Windows:
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Подробнее здесь https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

Устанавливаем пакеты:

```
uv sync
````

Делаем файлик .env туда пишем:


```
OPENAI_API_KEY = "ключ ChatGPT"
````

Запускаем:

```
python3 main.py
```
