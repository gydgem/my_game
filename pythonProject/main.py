import pygame
import random
import math
import os
from collections import deque

# Инициализация Pygame
pygame.init()

# Константы
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 40

FPS = 60
UI_HEIGHT = 100
RESOURCE_GENERATION_RADIUS = 20  # радиус спавна ресурсов вокруг базы
RESOURCE_COUNT = 100             # число ресурсных клеток
COLORS = {
    'light': (200, 200, 200),
    'gray': (100, 100, 100),
    'dark': (50, 50, 50),
    'resource': (0, 255, 0),
    'enemy': (255, 0, 0),    # враги — красные треугольники
    'base': (0, 0, 255),     # База — синий
    'turret': (255, 0, 0),   # Турель — красный
    'lantern': (255, 255, 0),# Фонарь — жёлтый
    'drill': (165, 42, 42),  # Буровая — коричневый
    'ui_bg': (30, 30, 30),
    'text': (255, 255, 255),
    'menu_bg': (20, 20, 20),
    'button': (70, 70, 70),
    'button_hover': (100, 200, 100),
}

# Стоимость построек
BUILDINGS_COST = {'lantern': 10, 'turret': 20, 'drill': 15}

RECORD_FILE = "record.txt"


class Projectile:
    def __init__(self, start_x, start_y, target_x, target_y):
        self.x = start_x
        self.y = start_y
        self.target_x = target_x
        self.target_y = target_y
        self.speed = 200  # пикселей в секунду
        dx = target_x - start_x
        dy = target_y - start_y
        distance = math.hypot(dx, dy)
        if distance == 0:
            distance = 1
        self.vx = dx / distance * self.speed
        self.vy = dy / distance * self.speed

class Cell:
    def __init__(self, x, y, cell_type='dark'):
        self.x = x
        self.y = y
        self.type = cell_type  # 'light','gray','dark','resource'
        self.building = None


class Building:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.health = 100


class Headquarters(Building):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.light_radius = 5
        self.spawn_radius = 8


class Lantern(Building):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.light_radius = 3
        self.cost = BUILDINGS_COST['lantern']


class Turret(Building):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.range = 5
        self.damage = 50
        self.cost = BUILDINGS_COST['turret']
        self.fire_cooldown = 0  # Было 1


class Drill(Building):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.production_rate = 100  # единиц/сек
        self.cost = BUILDINGS_COST['drill']


class Enemy:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.health = 50
        self.speed = 1.5  # уменьшенная скорость


class Game:
    def __init__(self):
        # Основные компоненты Pygame
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Lanterns & Turrets")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        self.running = True

        # Состояние меню и игры
        self.state = 'MENU'  # MENU или PLAYING
        self.current_time = 0.0
        self.best_time = self.load_record()
        self.projectiles = []
        # Параметры мира
        self.world = {}
        self.buildings = []
        self.lantern_graph = {}
        self.gray_zones = set()
        self.enemies = []
        self.shots = []
        self.resources = 1000
        self.resource_acc = 0.0
        self.base = None
        self.selected_building = None
        self.spawn_timer = 0
        self.spawn_interval = 3000
        self.cam_x = 0
        self.cam_y = 0
        self.difficulty_level = 1
        self.last_difficulty_increase = 0
        self.base_spawn_interval = 3000  # Начальный интервал спавна
        self.textures = {
            'base': pygame.image.load('base.png').convert_alpha(),
            'lantern': pygame.image.load('lamp.png').convert_alpha(),
            'turret': pygame.image.load('tower.png').convert_alpha(),
            'drill': pygame.image.load('drill.png').convert_alpha(),
            'enemy': pygame.image.load('enemy.png').convert_alpha()  # новая текстура
        }

        # Масштабирование
        for key in self.textures:
            self.textures[key] = pygame.transform.scale(
                self.textures[key],
                (CELL_SIZE, CELL_SIZE)
            )

    # --- Загрузка и сохранение рекорда ---

    def load_record(self):
        if os.path.exists(RECORD_FILE):
            try:
                with open(RECORD_FILE, 'r') as f:
                    return float(f.read().strip())
            except:
                return 0.0
        return 0.0

    def save_record(self):
        with open(RECORD_FILE, 'w') as f:
            f.write(f"{self.best_time:.2f}")

    # --- Инициализация и перезапуск игры ---

    def restart(self):
        self.world.clear()
        self.buildings.clear()
        self.lantern_graph.clear()
        self.gray_zones.clear()
        self.enemies.clear()
        self.shots.clear()
        self.resources = 1000
        self.resource_acc = 0.0
        self.current_time = 0.0
        self.spawn_timer = 0
        # Устанавливаем камеру в центр
        self.cam_x = -(WIDTH // 2 - CELL_SIZE // 2)
        self.cam_y = -(HEIGHT // 2 - CELL_SIZE // 2)
        self.selected_building = None
        self.difficulty_level = 1
        self.last_difficulty_increase = 0
        self.spawn_interval = self.base_spawn_interval

        self.create_base(0, 0)
        self.state = 'PLAYING'



    # --- Работа с клетками и освещением ---

    def set_cell(self, x, y, ct):
        if (x, y) not in self.world:
            self.world[(x, y)] = Cell(x, y)
        self.world[(x, y)].type = ct

    def update_world(self, x, y, bld):
        if (x, y) not in self.world:
            self.world[(x, y)] = Cell(x, y)
        self.world[(x, y)].building = bld

    def can_place_lantern(self, x, y):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            c = self.world.get((x + dx, y + dy))
            if c and c.building and isinstance(c.building, (Lantern, Headquarters)):
                return True
        return False

    def create_base(self, x, y):
        hq = Headquarters(x, y)
        self.buildings.append(hq)
        self.base = hq
        self.update_world(x, y, hq)
        self.lantern_graph[(x, y)] = []
        self.recompute_lighting_and_zones()

    def add_lantern(self, x, y):
        prev_gray = self.gray_zones.copy()

        l = Lantern(x, y)
        self.buildings.append(l)
        self.update_world(x, y, l)
        self.lantern_graph[(x, y)] = []
        # обновляем граф связей
        for p in list(self.lantern_graph.keys()):
            nbrs = []
            x0, y0 = p
            for o in self.lantern_graph:
                if o != p and abs(o[0] - x0) + abs(o[1] - y0) == 1:
                    nbrs.append(o)
            self.lantern_graph[p] = nbrs
        self.recompute_lighting_and_zones()

        # Генерация ресурсов в бывших серых зонах
        new_light_in_gray = []
        for (gx, gy) in prev_gray:
            cell = self.world.get((gx, gy))
            if cell and cell.type == 'light':
                new_light_in_gray.append((gx, gy))

        for (gx, gy) in new_light_in_gray:
            if random.random() < 0.2:  # 20% шанс
                self.set_cell(gx, gy, 'resource')

    def remove_orphaned_lanterns(self):
        to_remove = []
        for p in list(self.lantern_graph.keys()):
            if p == (self.base.x, self.base.y):
                continue
            if not self.is_connected_to_base(p):
                to_remove.append(p)
        for p in to_remove:
            self.remove_building(*p)

    def is_connected_to_base(self, start):
        visited = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur == (self.base.x, self.base.y):
                return True
            for n in self.lantern_graph.get(cur, []):
                if n not in visited:
                    visited.add(n)
                    queue.append(n)
        return False

    def remove_building(self, x, y):
        c = self.world.get((x, y))
        if c and c.building:
            self.buildings = [b for b in self.buildings if not (b.x == x and b.y == y)]
            c.building = None
        if (x, y) in self.lantern_graph:
            del self.lantern_graph[(x, y)]
        for nbrs in self.lantern_graph.values():
            if (x, y) in nbrs:
                nbrs.remove((x, y))
        self.recompute_lighting_and_zones()

    def recompute_lighting_and_zones(self):
        # Сохраняем все ресурсы и постройки перед пересчетом
        resource_cells = [(x, y) for (x, y), c in self.world.items() if c.type == 'resource']

        # Сбрасываем только нересурсные клетки
        for c in self.world.values():
            if c.type != 'resource':
                c.type = 'dark'

        bx, by = self.base.x, self.base.y
        lr, sr = self.base.light_radius, self.base.spawn_radius

        # Освещение базы (светлая + серая зона)
        for dx in range(-sr, sr + 1):
            for dy in range(-sr, sr + 1):
                d = dx * dx + dy * dy
                x, y = bx + dx, by + dy
                if d <= lr * lr:
                    self.set_cell(x, y, 'light')
                elif d <= sr * sr:
                    self.set_cell(x, y, 'gray')  # Серая зона базы

        # Освещение от фонарей
        for p in self.lantern_graph:
            if p != (bx, by) and self.is_connected_to_base(p):
                bld = self.world[p].building
                if isinstance(bld, Lantern):
                    self.illuminate_from(bld)

        # Восстанавливаем ресурсы везде кроме темных зон
        for x, y in resource_cells:
            cell = self.world.get((x, y))
            if cell and cell.type != 'dark':
                self.set_cell(x, y, 'resource')

        # Расширяем границы серой зоны
        extra_gray = set()
        for (x, y), c in self.world.items():
            if c.type == 'light':
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in self.world or self.world[(nx, ny)].type == 'dark':
                        extra_gray.add((nx, ny))

        for g in extra_gray:
            self.set_cell(g[0], g[1], 'gray')

        self.gray_zones = {p for p, c in self.world.items() if c.type == 'gray'}

        # Удаляем объекты ТОЛЬКО в темных зонах
        for (x, y), cell in list(self.world.items()):
            if cell.type == 'dark':
                # Удаляем ВСЕ постройки
                if cell.building:
                    self.buildings = [b for b in self.buildings if (b.x, b.y) != (x, y)]
                    cell.building = None

                # Удаляем ресурсы
                if cell.type == 'resource':
                    cell.type = 'dark'

    def illuminate_from(self, b):
        for dx in range(-b.light_radius, b.light_radius + 1):
            for dy in range(-b.light_radius, b.light_radius + 1):
                if dx * dx + dy * dy <= b.light_radius * b.light_radius:
                    x, y = b.x + dx, b.y + dy
                    if (x, y) in self.world:
                        self.set_cell(x, y, 'light')

    # --- Враги и оборона ---

    def spawn_enemy(self):
        if not self.gray_zones:
            return

        # Расчет количества врагов и их характеристик
        spawn_count = min(self.difficulty_level, 5)  # Не более 5 за спавн
        health_multiplier = 1 + 0.2 * (self.difficulty_level - 1)
        speed_multiplier = 1 + 0.1 * (self.difficulty_level - 1)

        for _ in range(spawn_count):
            x, y = random.choice(list(self.gray_zones))
            enemy = Enemy(x, y)
            enemy.health = int(enemy.health * health_multiplier)
            enemy.speed = enemy.speed * speed_multiplier
            self.enemies.append(enemy)

    def update_enemies(self, dt):
        for e in list(self.enemies):
            targets = [p for p in self.lantern_graph if self.is_connected_to_base(p)]
            if not targets:
                continue
            tx, ty = min(targets, key=lambda p: math.hypot(p[0] - e.x, p[1] - e.y))
            dx, dy = tx - e.x, ty - e.y
            dist = math.hypot(dx, dy)
            step = e.speed * (dt / 1000)
            if dist > 0:
                e.x += dx / dist * step
                e.y += dy / dist * step
            if dist < 0.2:
                self.remove_building(tx, ty)
                e.health = 0
        self.enemies = [e for e in self.enemies if e.health > 0]

    def update_turrets(self, dt):
        for bld in self.buildings:
            if isinstance(bld, Turret):
                bld.fire_cooldown -= dt
                if bld.fire_cooldown > 0:
                    continue

                in_range = [e for e in self.enemies if math.hypot(e.x - bld.x, e.y - bld.y) <= bld.range]
                if in_range:
                    target = min(in_range, key=lambda e: math.hypot(e.x - bld.x, e.y - bld.y))
                    start_x = bld.x * CELL_SIZE + CELL_SIZE // 2
                    start_y = bld.y * CELL_SIZE + CELL_SIZE // 2
                    end_x = target.x * CELL_SIZE + CELL_SIZE // 2
                    end_y = target.y * CELL_SIZE + CELL_SIZE // 2
                    self.projectiles.append(Projectile(start_x, start_y, end_x, end_y))
                    bld.fire_cooldown = 1000  # 1 выстрел в секунду

    def update_projectiles(self, dt):
        for projectile in self.projectiles[:]:
            projectile.x += projectile.vx * (dt / 1000)
            projectile.y += projectile.vy * (dt / 1000)

            hit = False
            # Проверяем всех врагов
            for enemy in self.enemies[:]:
                # Получаем координаты центра врага
                enemy_x = enemy.x * CELL_SIZE + CELL_SIZE // 2
                enemy_y = enemy.y * CELL_SIZE + CELL_SIZE // 2

                # Проверяем расстояние до снаряда
                if math.hypot(projectile.x - enemy_x, projectile.y - enemy_y) < 20:
                    enemy.health -= 50
                    if enemy.health <= 0:
                        self.enemies.remove(enemy)
                    hit = True
                    break

            # Удаляем снаряд при попадании или достижении цели
            if hit or math.hypot(projectile.x - projectile.target_x,
                                 projectile.y - projectile.target_y) < 10:
                self.projectiles.remove(projectile)

    def update_drills(self, dt):
        total_rate = sum(
            bld.production_rate for bld in self.buildings
            if isinstance(bld, Drill)
        )
        self.resource_acc += total_rate * (dt / 1000)
        gain = int(self.resource_acc)
        if gain > 0:
            self.resources += gain
            self.resource_acc -= gain

    # --- Обработка ввода ---

    def handle_menu_input(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                btn = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 25, 200, 50)
                if btn.collidepoint(mx, my):
                    self.restart()

    def handle_game_input(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_LEFT:
                    self.cam_x -= CELL_SIZE
                elif ev.key == pygame.K_RIGHT:
                    self.cam_x += CELL_SIZE
                elif ev.key == pygame.K_UP:
                    self.cam_y -= CELL_SIZE
                elif ev.key == pygame.K_DOWN:
                    self.cam_y += CELL_SIZE
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                # Выбор типа постройки в UI
                if my > HEIGHT - UI_HEIGHT:
                    if HEIGHT - UI_HEIGHT + 50 < my < HEIGHT - UI_HEIGHT + 90:
                        if 10 < mx < 130:
                            self.selected_building = 'lantern'
                        elif 140 < mx < 260:
                            self.selected_building = 'turret'
                        elif 270 < mx < 390:
                            self.selected_building = 'drill'
                else:
                    gx = (mx + self.cam_x) // CELL_SIZE
                    gy = (my + self.cam_y) // CELL_SIZE
                    cell = self.world.get((gx, gy))
                    cost = BUILDINGS_COST.get(self.selected_building, 0)
                    if not cell or cell.building or self.resources < cost:
                        continue
                    # Проверка ресурсов и типа клетки
                    if self.selected_building == 'drill':
                        if cell.type != 'resource':
                            continue
                    else:
                        if cell.type != 'light':
                            continue
                        if self.selected_building == 'lantern' and not self.can_place_lantern(gx, gy):
                            continue
                    # Строительство
                    self.resources -= cost
                    if self.selected_building == 'lantern':
                        self.add_lantern(gx, gy)
                    elif self.selected_building == 'turret':
                        t = Turret(gx, gy)
                        self.buildings.append(t)
                        self.update_world(gx, gy, t)
                    else:
                        d = Drill(gx, gy)
                        self.buildings.append(d)
                        self.update_world(gx, gy, d)

    # --- Отрисовка экранов ---

    def draw_menu(self):
        self.screen.fill(COLORS['menu_bg'])
        title = self.font.render("Lanterns & Turrets", True, COLORS['text'])
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 4))

        mx, my = pygame.mouse.get_pos()
        btn = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 25, 200, 50)
        color = COLORS['button_hover'] if btn.collidepoint(mx, my) else COLORS['button']
        pygame.draw.rect(self.screen, color, btn)
        txt = self.font.render("Play", True, COLORS['text'])
        self.screen.blit(txt, (
            btn.x + btn.width // 2 - txt.get_width() // 2,
            btn.y + btn.height // 2 - txt.get_height() // 2
        ))

        rec = self.font.render(f"Best time: {self.best_time:.1f} s", True, COLORS['text'])
        self.screen.blit(rec, (WIDTH // 2 - rec.get_width() // 2, HEIGHT // 2 + 50))

    def draw_timer(self):
        cur = self.font.render(f"Time: {self.current_time:.1f} s", True, COLORS['text'])
        rec = self.font.render(f"Record: {self.best_time:.1f} s", True, COLORS['text'])
        self.screen.blit(cur, (10, 10))
        self.screen.blit(rec, (10, 40))

    def draw_ui(self):
        ui = pygame.Rect(0, HEIGHT - UI_HEIGHT, WIDTH, UI_HEIGHT)
        pygame.draw.rect(self.screen, COLORS['ui_bg'], ui)

        # Отображение ресурсов
        txt = self.font.render(f'Resources: {self.resources}', True, COLORS['text'])
        self.screen.blit(txt, (10, HEIGHT - UI_HEIGHT + 10))

        # Кнопки построек
        x_pos = 10
        buttons = [
            ('lantern', self.textures['lantern'], 10),
            ('turret', self.textures['turret'], 20),
            ('drill', self.textures['drill'], 15)
        ]

        for building_type, texture, cost in buttons:
            btn_rect = pygame.Rect(x_pos, HEIGHT - UI_HEIGHT + 40, 80, 40)

            # Цвет фона кнопки
            color = COLORS['button_hover'] if self.selected_building == building_type else COLORS['button']
            pygame.draw.rect(self.screen, color, btn_rect)

            # Иконка
            self.screen.blit(texture, (x_pos + 5, HEIGHT - UI_HEIGHT + 45))

            # Цена
            cost_text = self.font.render(str(cost), True, COLORS['text'])
            self.screen.blit(cost_text, (x_pos + 50, HEIGHT - UI_HEIGHT + 55))

            x_pos += 100

    def draw_grid(self):
        self.screen.fill((0, 0, 0))

        # Отрисовка клеток
        for (x, y), cell in self.world.items():
            rect = pygame.Rect(
                x * CELL_SIZE - self.cam_x,
                y * CELL_SIZE - self.cam_y,
                CELL_SIZE,
                CELL_SIZE
            )

            # Цвета клеток
            if cell.type == 'resource':
                pygame.draw.rect(self.screen, COLORS['resource'], rect)
            else:
                pygame.draw.rect(self.screen, COLORS[cell.type], rect)

        # Отрисовка кабелей
        for p, nbrs in self.lantern_graph.items():
            for n in nbrs:
                if p < n:
                    x1, y1 = p
                    x2, y2 = n
                    start_pos = (
                        x1 * CELL_SIZE - self.cam_x + CELL_SIZE // 2,
                        y1 * CELL_SIZE - self.cam_y + CELL_SIZE // 2
                    )
                    end_pos = (
                        x2 * CELL_SIZE - self.cam_x + CELL_SIZE // 2,
                        y2 * CELL_SIZE - self.cam_y + CELL_SIZE // 2
                    )
                    pygame.draw.line(self.screen, (0, 0, 0), start_pos, end_pos, 3)

        # Отрисовка зданий
        for building in self.buildings:
            if isinstance(building, Lantern):
                # Отрисовка фонарей как желтых кругов
                center = (
                    building.x * CELL_SIZE - self.cam_x + CELL_SIZE // 2,
                    building.y * CELL_SIZE - self.cam_y + CELL_SIZE // 2
                )
                pygame.draw.circle(self.screen, COLORS['lantern'], center, CELL_SIZE // 3)
            else:
                # Отрисовка остальных построек текстурами
                texture = None
                if isinstance(building, Headquarters):
                    texture = self.textures['base']
                elif isinstance(building, Turret):
                    texture = self.textures['turret']
                elif isinstance(building, Drill):
                    texture = self.textures['drill']

                if texture:
                    pos = (
                        building.x * CELL_SIZE - self.cam_x,
                        building.y * CELL_SIZE - self.cam_y
                    )
                    self.screen.blit(texture, pos)

        # Отрисовка снарядов
        for projectile in self.projectiles:
            # Ядро снаряда
            pygame.draw.circle(self.screen, (255, 0, 0),
                               (int(projectile.x - self.cam_x),
                                int(projectile.y - self.cam_y)), 5)
            # Свечение
            pygame.draw.circle(self.screen, (255, 255, 0),
                               (int(projectile.x - self.cam_x),
                                int(projectile.y - self.cam_y)), 3)

        # Отрисовка врагов
        for enemy in self.enemies:
            pos = (
                enemy.x * CELL_SIZE - self.cam_x,
                enemy.y * CELL_SIZE - self.cam_y
            )
            self.screen.blit(self.textures['enemy'], pos)

        # Отрисовка выстрелов
        for sx, sy in self.shots:
            pygame.draw.circle(self.screen, (255, 255, 0), (int(sx), int(sy)), 3)
    # --- Основной цикл ---

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS)
            if self.state == 'MENU':
                self.handle_menu_input()
                self.draw_menu()

            elif self.state == 'PLAYING':
                self.update_projectiles(dt)
                self.current_time += dt / 1000.0

                # Увеличение сложности каждые 30 секунд
                if self.current_time - self.last_difficulty_increase > 30:
                    self.difficulty_level += 1
                    self.last_difficulty_increase = self.current_time
                    # Уменьшаем интервал спавна с ограничением
                    self.spawn_interval = max(1000, self.base_spawn_interval - 200 * self.difficulty_level)

                self.handle_game_input()
                self.remove_orphaned_lanterns()

                self.spawn_timer += dt
                if self.spawn_timer >= self.spawn_interval:
                    self.spawn_enemy()
                    self.spawn_timer = 0

                self.update_drills(dt)
                self.update_enemies(dt)
                self.update_turrets(dt)

                self.draw_grid()
                self.draw_timer()
                self.draw_ui()

                # Отображение уровня сложности
                difficulty_text = self.font.render(f"Wave: {self.difficulty_level}", True, COLORS['text'])
                self.screen.blit(difficulty_text, (WIDTH - 150, 10))

                # Проверка поражения
                if self.base not in self.buildings:
                    if self.current_time > self.best_time:
                        self.best_time = self.current_time
                        self.save_record()
                    self.state = 'MENU'

            pygame.display.flip()
        pygame.quit()


if __name__ == '__main__':
    Game().run()