# OasisAC ML Backend — самохостинг

Твой личный замена облачного сервера `oasisac.biz`.
Принимает данные от плагина, анализирует движения мышью и возвращает `confidence`.

---

## Быстрый старт (5 минут)

### 1. Требования
- VPS с Linux (Ubuntu 20.04+)
- Docker + Docker Compose

```bash
# Установка Docker (если нет)
curl -fsSL https://get.docker.com | sh
apt install docker-compose -y
```

### 2. Запуск сервера

```bash
# Клонируй / скопируй папку на VPS
scp -r oasis-server/ user@YOUR_VPS_IP:/opt/oasis-server/
ssh user@YOUR_VPS_IP

cd /opt/oasis-server

# ВАЖНО: задай свой API ключ в docker-compose.yml
nano docker-compose.yml
# Измени строку: OASIS_API_KEY=your_secret_key_here

# Запуск
docker-compose up -d

# Проверка
curl http://localhost:8080/health
# → {"status": "ok"}
```

При первом запуске сервер **автоматически обучит модель** на синтетических данных (~10 сек).
Логи: `docker logs oasis-backend -f`

---

### 3. Настройка плагина

Открой `plugins/oasisAC/config.yml` на игровом сервере:

```yaml
ml-api:
  endpoint: "YOUR_VPS_IP:8080"   # или домен: "ac.myserver.ru"
  api-key: "your_secret_key_here"  # тот же ключ что в docker-compose
  timeout: 10
  retry-attempts: 3
```

Перезагрузи плагин: `/oasis reload`

---

## Улучшение модели на реальных данных

Синтетическая модель работает, но со временем можно улучшить точность
на реальных игроках с твоего сервера.

### Шаг 1: Сбор данных через плагин

```
# Для честного игрока (легитимный)
/oasis datacollect start legit <ник>
# Пусть игрок поиграет 5-10 минут
/oasis datacollect stop <ник>

# Для читера (если поймал)
/oasis datacollect start cheat <ник>
/oasis datacollect stop <ник>
```

CSV-файлы сохраняются в `plugins/oasisAC/datacollect/`

### Шаг 2: Копирование данных на VPS

```bash
scp plugins/oasisAC/datacollect/*.csv user@YOUR_VPS_IP:/opt/oasis-server/data/datacollect/
```

### Шаг 3: Переобучение

```bash
docker exec oasis-backend python retrain.py
docker restart oasis-backend
```

Скрипт покажет качество модели (ROC-AUC). Чем ближе к 1.0 — тем лучше.
Рекомендуется: минимум 50+ записей читеров и 200+ честных игроков.

---

## Структура проекта

```
oasis-server/
├── app/
│   ├── main.py       # FastAPI сервер, эндпоинт /analyze
│   └── model.py      # Извлечение фич + sklearn модель
├── retrain.py        # Скрипт переобучения на реальных данных
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── data/             # Создаётся автоматически
    ├── model.pkl     # Обученная модель
    └── datacollect/  # Сюда кладёшь CSV с сервера
```

---

## Переменные окружения

| Переменная       | По умолчанию          | Описание                          |
|------------------|-----------------------|-----------------------------------|
| `OASIS_API_KEY`  | `your_secret_key_here`| Секретный ключ (совпадает с config.yml) |
| `MODEL_PATH`     | `/data/model.pkl`     | Путь к сохранённой модели         |
| `DATACOLLECT_DIR`| `/data/datacollect`   | Папка с CSV для дообучения        |

---

## Nginx + SSL (опционально, для продакшена)

Если хочешь `https://ac.myserver.ru` вместо IP:

```nginx
server {
    listen 443 ssl;
    server_name ac.myserver.ru;

    ssl_certificate     /etc/letsencrypt/live/ac.myserver.ru/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ac.myserver.ru/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
    }
}
```

```bash
certbot --nginx -d ac.myserver.ru
```

Тогда в config.yml плагина:
```yaml
endpoint: "ac.myserver.ru"
```
