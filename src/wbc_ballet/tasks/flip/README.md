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

## Posture + locomotion reward changes

Эта версия закрывает два локальных optimum, которые наблюдались после ~15k PPO
iterations: policy использовала waist почти на механических лимитах, а нулевая
velocity-команда приводила к кривой статической позе; одновременно locomotion
команда была слишком слабо отличима от "просто стоять".

### `waist_zero_pose_penalty`

Работает в `flip=0` и `flip=1`, на любой скорости. Для каждой незамаскированной
оси waist используется **сырая ошибка в радианах**, без нормализации на полный
mechanical range:

```text
waist_yaw^2 + waist_roll^2 + waist_pitch^2
```

Замаскированная ось исключается отдельно. Curriculum:

```text
-1.0 -> -2.0 -> -3.0 -> -4.0
```

Это намеренно намного сильнее старого penalty: bend порядка 0.5 rad больше не
является дешёвым способом компенсировать перекошенный pelvis.

### `pelvis_orientation_penalty`

Работает **в обоих flip modes и на любой velocity**:

- `flip=0`: local +Z pelvis должен смотреть в world +Z;
- `flip=1`: local +Z pelvis должен смотреть в world -Z;
- yaw остаётся свободным.

Это не позволяет получить хороший `torso_link upright`, сильно наклонив pelvis
и компенсировав наклон экстремальным waist bend. Curriculum:

```text
-0.5 -> -1.0 -> -1.5 -> -2.0
```

### `feet_stand_pose_hold`

Работает только когда:

- `flip == 0`;
- `max(abs(vx), abs(vy), abs(yaw_rate)) < 0.05`;
- на обеих ногах нет ballet masks.

Все 12 leg joints тянутся к default standing pose. Ошибка теперь нормализуется
не на огромный mechanical range, а на posture tolerances:

```text
hip_pitch      0.25 rad
hip_roll/yaw   0.15 rad
knee           0.25 rad
ankle_pitch    0.15 rad
ankle_roll     0.12 rad
```

После нормализации используется Huber loss. Curriculum:

```text
-0.25 -> -0.50 -> -0.75 -> -1.00
```

Во время ходьбы этот term полностью выключен.

### `feet_flatness_penalty`

Upright + zero velocity. Каждая сторона mask-aware отдельно. Штрафует tilt
локальной Z-оси стопы относительно world Z, то есть стояние на носке/пятке или
на ребре. Curriculum:

```text
-0.10 -> -0.20 -> -0.35 -> -0.50
```

### `handstand_leg_pose_penalty`

При `flip=1` незамаскированные leg axes тянутся не к standing home, а к
`HANDSTAND_LEGS`. Это устраняет прежний конфликт между action reference и
`unmasked_home_tracking`. Используются те же физические tolerances + Huber.
Curriculum:

```text
-0.25 -> -0.50 -> -1.00 -> -1.50
```

`unmasked_home_tracking` в inverted mode теперь не тянет ноги к обычной
standing pose.

### `HANDSTAND_WAIST`

Handstand action reference теперь имеет строго нулевой waist:

```text
waist_yaw   = 0
waist_roll  = 0
waist_pitch = 0
```

### Velocity tracking

Линейное tracking стало существенно более discriminative:

```text
track_linear_velocity: weight +3.0, std 0.22
linear_velocity_error_penalty: weight -1.5
```

Раньше при `std=0.5` policy могла полностью игнорировать `vx=0.4` и всё равно
получать более половины максимального tracking reward. Теперь стоять на месте
при ненулевой команде существенно невыгодно.

### `com_support_projection`

При ненулевой velocity и отсутствии single-support ballet mask этот term теперь
**выключен**. Раньше он притягивал COM к midpoint двух стоп/кистей именно во
время движения и тем самым подавлял перенос веса, необходимый для шага.

При zero velocity или при явной single-support mask он остаётся активным.

### Upright gait helpers

`upright_foot_clearance` (`-0.5`) использует уже существующий foot-height sensor.
Во время upright locomotion moving foot penalized за отклонение от ~8 cm swing
height; stance foot почти не штрафуется, потому что ошибка умножается на
planar foot speed.

`upright_support_switch` (`+0.30`) даёт небольшой reward pulse, когда при
ненулевой velocity single-support переключается left -> right или right ->
left. Это stateful event reward без скрытой gait phase и без изменения actor
observation ABI.

### Curriculum timing

Posture curricula начинаются с iteration 0 и меняют stage каждые 1000 PPO
iterations. Mask/target curriculum остаётся прежним: первые 4000 iterations
`mask_probability=0` и `target_scale=0`, затем target scale растёт до 0.9 за
1000 iterations, mask probability до 0.15 за 4000 iterations.

Важно: `flip` всё ещё 50/50 с начала обучения; это не "4000 iterations только
upright walking".

### Play / eval

`FlipPlayRewardsCfg` сразу использует финальные curriculum weights:

```text
feet_stand_pose_hold       -1.0
feet_flatness_penalty      -0.5
pelvis_orientation_penalty -2.0
waist_zero_pose_penalty    -4.0
handstand_leg_pose_penalty -1.5
```

Поэтому play не откатывается к stage-0 весам.
