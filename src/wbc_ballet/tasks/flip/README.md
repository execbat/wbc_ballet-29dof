# Flip task

Самостоятельная папка task без импортов и наследования из `tasks/ballet`.
Вся локальная MDP-логика находится в штатных файлах `mdp/commands.py`,
`curriculums.py`, `events.py`, `metrics.py`, `observations.py`, `rewards.py`
и `terminations.py`.

## Обучение

```bash
uv run train Mjlab-Flip-Flat-Unitree-G1-29DoF
```

## Play

```bash
uv run play Mjlab-Flip-Flat-Unitree-G1-29DoF --checkpoint-file /absolute/path/to/model.pt
```

Flip в training выбирается 50/50 на каждом reset и перерандомизируется через
9–11 секунд. В play flip поступает из UDP-геймпада, а база и суставы при каждом
reset рандомизируются так же, как в training.

Разрешённые опоры на земле: стопы и звенья `wrist_roll`, `wrist_pitch`,
`wrist_yaw`. Контакт любой другой частью тела после двухсекундного spawn grace
немедленно завершает эпизод. Отдельный `fell_over` срабатывает после накопленных
двух секунд высоты таза ниже 0,2 м; короткий отскок не очищает таймер, для
очистки требуется две секунды корректного восстановления выше 0,35 м.

Стопы и кисти имеют startup-рандомизацию трения 0,3–1,2. Для кистей в сцене
явно включён `condim=3`. `contact_sensor_maxmatch=1024` установлен и для
training, и для play.

Папка использует общие инфраструктурные модули проекта (`robots`, `teleop`,
`utils`) и MJLab, но не зависит от соседнего task `ballet`.

Регистрация остаётся в родительском `tasks/__init__.py`. Если папка `ballet`
удаляется физически, из этого файла также нужно удалить два импорта/регистрации
`Mjlab-Ballet-*`, оставив регистрацию `Mjlab-Flip-Flat-Unitree-G1-29DoF`.
