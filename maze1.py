import pygame
from heapq import heappush, heappop
from collections import deque
import sys
import os

# ---------- CONFIG ----------
WIDTH, HEIGHT = 600, 600     # window size
ROWS, COLS = 30, 30          # grid size
CELL_SIZE = WIDTH // COLS

# Movement intervals (ms)
PAC_MOVE_INTERVAL = 120      # smaller = faster Pac-Man
GHOST_MOVE_INTERVAL = 260    # bigger = slower ghost

# Cell types
EMPTY = 0
WALL = 1
START = 2
GOAL = 3
OPEN = 4
CLOSED = 5
PATH = 6      # not used visually now, but kept for compatibility
REWARD = 7

# Neon / glowy colors
COLORS = {
    EMPTY:  (15, 15, 35),         # dark background
    WALL:   (30, 30, 60),         # base wall fill (outline added separately)
    START:  (0, 255, 180),        # neon teal
    GOAL:   (255, 80, 120),       # neon red/pink
    OPEN:   (80, 80, 200),
    CLOSED: (60, 60, 160),
    PATH:   (15, 15, 35),         # same as EMPTY now (no trail)
    REWARD: (180, 130, 255),      # base color; diamond drawn separately
}

pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Pac-Man Maze (Neon, Corner Ghosts, Timer, Levels)")
FONT_SMALL = pygame.font.SysFont("consolas", 20)
FONT_BIG = pygame.font.SysFont("consolas", 36)


# ---------- UTILS ----------

def corner_positions():
    """Return the four grid corners as (row, col)."""
    return [
        (0, 0),
        (0, COLS - 1),
        (ROWS - 1, 0),
        (ROWS - 1, COLS - 1),
    ]


# ---------- GRID & DRAWING ----------

def make_grid():
    return [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]


def draw_grid_lines(surface):
    # Soft glowing grid lines
    for r in range(ROWS):
        pygame.draw.line(surface, (60, 60, 120), (0, r * CELL_SIZE), (WIDTH, r * CELL_SIZE), 1)
    for c in range(COLS):
        pygame.draw.line(surface, (60, 60, 120), (c * CELL_SIZE, 0), (c * CELL_SIZE, HEIGHT), 1)


def draw_grid(surface, grid):
    surface.fill((5, 5, 20))  # deep dark blue background
    for r in range(ROWS):
        for c in range(COLS):
            cell_type = grid[r][c]
            color = COLORS.get(cell_type, (0, 0, 0))
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if cell_type == WALL:
                # Fill wall
                pygame.draw.rect(surface, COLORS[WALL], rect, border_radius=6)
                # Bright cyan outline to highlight walls
                pygame.draw.rect(surface, (0, 255, 255), rect, width=2, border_radius=6)
            else:
                pygame.draw.rect(surface, color, rect, border_radius=4)

    draw_grid_lines(surface)


def draw_hud(surface, score, start_time, end_time, game_over, final_score, reason):
    # Timer
    if start_time is not None:
        if game_over and end_time is not None:
            elapsed_ms = end_time - start_time
        else:
            elapsed_ms = pygame.time.get_ticks() - start_time
        elapsed_sec = elapsed_ms // 1000
        mins = elapsed_sec // 60
        secs = elapsed_sec % 60
        time_str = f"{mins:02d}:{secs:02d}"
    else:
        time_str = "00:00"

    # Neon HUD
    score_text = FONT_SMALL.render(f"♦ {score}", True, (255, 215, 0))
    time_text = FONT_SMALL.render(f"⏱ {time_str}", True, (0, 255, 200))
    surface.blit(score_text, (10, 10))
    surface.blit(time_text, (10, 35))

    if game_over:
        if reason == "win":
            msg = "YOU WIN!"
        elif reason == "ghost":
            msg = "GHOST CAUGHT YOU!"
        elif reason == "stuck":
            msg = "NO PATH TO GOAL!"
        else:
            msg = "GAME OVER"

        msg_surf = FONT_BIG.render(msg, True, (255, 80, 160))
        surface.blit(msg_surf, (WIDTH // 2 - msg_surf.get_width() // 2, 10 + 50))

        if final_score is not None:
            fs_surf = FONT_BIG.render(f"Final Score: {final_score}", True, (255, 255, 255))
            surface.blit(fs_surf, (WIDTH // 2 - fs_surf.get_width() // 2, 10 + 90))


def get_cell_from_mouse(pos):
    x, y = pos
    c = x // CELL_SIZE
    r = y // CELL_SIZE
    if 0 <= r < ROWS and 0 <= c < COLS:
        return r, c
    return None


# ---------- A* SEARCH (Pac-Man) ----------

def heuristic(a, b):
    (r1, c1) = a
    (r2, c2) = b
    return abs(r1 - r2) + abs(c1 - c2)


def reconstruct_path(came_from, current, start):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(grid, start, goal, blocked=None):
    """A* from start to goal; blocked is a set of cells to treat as walls (e.g. ghost positions)."""
    if start is None or goal is None:
        return None

    if blocked is None:
        blocked = set()

    g_score = {(r, c): float('inf') for r in range(ROWS) for c in range(COLS)}
    f_score = {(r, c): float('inf') for r in range(ROWS) for c in range(COLS)}

    g_score[start] = 0
    f_score[start] = heuristic(start, goal)

    open_set = []
    heappush(open_set, (f_score[start], 0, start))
    came_from = {}
    in_open = {start}
    counter = 0

    while open_set:
        _, _, current = heappop(open_set)
        in_open.discard(current)

        if current == goal:
            return reconstruct_path(came_from, goal, start)

        (r, c) = current

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if grid[nr][nc] == WALL:
                continue
            if (nr, nc) in blocked:
                continue

            neighbor = (nr, nc)
            tentative_g = g_score[current] + 1

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                if neighbor not in in_open:
                    counter += 1
                    heappush(open_set, (f_score[neighbor], counter, neighbor))
                    in_open.add(neighbor)

    return None  # no path


# ---------- BFS FOR GHOSTS (chasing Pac-Man) ----------

def bfs_path(grid, start, goal):
    """Simple BFS path for a ghost from start to Pac-Man."""
    if start is None or goal is None:
        return None
    if start == goal:
        return [start]

    q = deque([start])
    visited = {start}
    came_from = {}

    while q:
        current = q.popleft()
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        (r, c) = current
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if grid[nr][nc] == WALL:
                continue
            nxt = (nr, nc)
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = current
                q.append(nxt)

    return None


# ---------- PAC-MAN SPRITE ----------

def create_pacman_frames():
    frames = []
    for mouth_open in [True, False]:
        surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        center = (CELL_SIZE // 2, CELL_SIZE // 2)
        radius = CELL_SIZE // 2 - 4
        # Pac-Man body
        pygame.draw.circle(surf, (255, 255, 0), center, radius)
        # Small eye
        pygame.draw.circle(surf, (0, 0, 0), (center[0] - 3, center[1] - 4), 2)
        if mouth_open:
            # Mouth facing right
            p1 = center
            p2 = (center[0] + radius, center[1] - radius // 2)
            p3 = (center[0] + radius, center[1] + radius // 2)
            pygame.draw.polygon(surf, (0, 0, 0, 0), [p1, p2, p3])
        frames.append(surf)
    return frames


# ---------- LEVEL LOADING (centered, multiple ghosts) ----------

def load_level(filename):
    """
    Load and center a level from text file into the 30x30 grid.
    'X' in the file adds a ghost at that location.
    """
    grid = make_grid()
    start = None
    goal = None
    reward_cells = set()
    ghost_positions = []

    if not os.path.isfile(filename):
        print(f"Level file '{filename}' not found.")
        return grid, start, goal, reward_cells, ghost_positions

    with open(filename, "r") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    if not lines:
        return grid, start, goal, reward_cells, ghost_positions

    level_rows = len(lines)
    level_cols = max(len(line) for line in lines)

    row_offset = (ROWS - level_rows) // 2
    col_offset = (COLS - level_cols) // 2

    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            gr = r + row_offset
            gc = c + col_offset
            if not (0 <= gr < ROWS and 0 <= gc < COLS):
                continue

            if ch == "#":
                grid[gr][gc] = WALL
            elif ch == "S":
                grid[gr][gc] = START
                start = (gr, gc)
            elif ch == "G":
                grid[gr][gc] = GOAL
                goal = (gr, gc)
            elif ch == "D":
                grid[gr][gc] = REWARD
                reward_cells.add((gr, gc))
            elif ch == "X":
                ghost_positions.append((gr, gc))
            else:
                grid[gr][gc] = EMPTY

    return grid, start, goal, reward_cells, ghost_positions


# ---------- MAIN LOOP ----------

def main():
    grid = make_grid()
    start = None
    goal = None
    reward_cells = set()

    pac_frames = create_pacman_frames()
    pac_frame_idx = 0

    pac_pos = None
    # multiple ghosts now
    ghost_positions = []

    # by default, spawn ghosts at corners where not walls
    for (r, c) in corner_positions():
        if 0 <= r < ROWS and 0 <= c < COLS:
            if grid[r][c] != WALL:
                ghost_positions.append((r, c))

    playing = False
    game_over = False
    reason = None  # "win", "ghost", "stuck"
    score = 0
    final_score = None

    start_time = None
    end_time = None
    last_pac_move_time = 0
    last_ghost_move_time = 0

    clock = pygame.time.Clock()

    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Keyboard controls
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    # Clear everything, then respawn corner ghosts
                    grid = make_grid()
                    start = None
                    goal = None
                    reward_cells.clear()
                    pac_pos = None
                    ghost_positions = []
                    for (r, c) in corner_positions():
                        if 0 <= r < ROWS and 0 <= c < COLS:
                            if grid[r][c] != WALL:
                                ghost_positions.append((r, c))

                    playing = False
                    game_over = False
                    reason = None
                    score = 0
                    final_score = None
                    start_time = None
                    end_time = None

                if event.key == pygame.K_l:
                    # Load level1.txt (centered), then also add corner ghosts if possible
                    grid, start, goal, reward_cells, ghost_positions = load_level("level1.txt")
                    # ensure corner ghosts exist (on non-wall cells)
                    corners = corner_positions()
                    for (r, c) in corners:
                        if 0 <= r < ROWS and 0 <= c < COLS and grid[r][c] != WALL:
                            if (r, c) not in ghost_positions:
                                ghost_positions.append((r, c))

                    pac_pos = None
                    playing = False
                    game_over = False
                    reason = None
                    score = 0
                    final_score = None
                    start_time = None
                    end_time = None

                if event.key == pygame.K_SPACE and start and goal and not playing:
                    # Start game
                    playing = True
                    game_over = False
                    reason = None
                    pac_pos = start
                    score = 0
                    final_score = None
                    start_time = pygame.time.get_ticks()
                    end_time = None
                    last_pac_move_time = start_time
                    last_ghost_move_time = start_time

                if event.key == pygame.K_r:
                    # Reset run but keep maze and ghosts
                    playing = False
                    game_over = False
                    reason = None
                    final_score = None
                    pac_pos = None
                    start_time = None
                    end_time = None

        # Mouse editing only when not playing / not game over
        if not playing and not game_over:
            if pygame.mouse.get_pressed()[0]:  # left click
                pos = pygame.mouse.get_pos()
                cell = get_cell_from_mouse(pos)
                if cell:
                    r, c = cell
                    keys = pygame.key.get_pressed()

                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                        # Shift + click => diamond
                        if (r, c) != start and (r, c) != goal:
                            grid[r][c] = REWARD
                            reward_cells.add((r, c))

                    elif keys[pygame.K_g]:
                        # G + click => add ghost at clicked cell
                        if (r, c) not in ghost_positions:
                            ghost_positions.append((r, c))

                    else:
                        # Normal left click: start, goal, then walls
                        if not start:
                            start = (r, c)
                            grid[r][c] = START
                            reward_cells.discard((r, c))
                        elif not goal and (r, c) != start:
                            goal = (r, c)
                            grid[r][c] = GOAL
                            reward_cells.discard((r, c))
                        elif (r, c) != start and (r, c) != goal:
                            grid[r][c] = WALL
                            reward_cells.discard((r, c))

                            # if we place a wall on a ghost, remove that ghost
                            if (r, c) in ghost_positions:
                                ghost_positions.remove((r, c))

            elif pygame.mouse.get_pressed()[2]:  # right click to erase
                pos = pygame.mouse.get_pos()
                cell = get_cell_from_mouse(pos)
                if cell:
                    r, c = cell
                    if (r, c) == start:
                        start = None
                    elif (r, c) == goal:
                        goal = None
                    if (r, c) in ghost_positions:
                        ghost_positions.remove((r, c))
                    grid[r][c] = EMPTY
                    reward_cells.discard((r, c))

        # ---------- GAME LOGIC ----------
        if playing and not game_over and pac_pos is not None:
            now = pygame.time.get_ticks()

            # Pac-Man movement (A* replanning)
            if now - last_pac_move_time >= PAC_MOVE_INTERVAL:
                last_pac_move_time = now
                blocked = set(ghost_positions) if ghost_positions else set()
                path = astar(grid, pac_pos, goal, blocked=blocked)

                if path is None or len(path) < 2:
                    # No path to goal
                    playing = False
                    game_over = True
                    reason = "stuck"
                    final_score = score
                    end_time = now
                else:
                    next_cell = path[1]
                    pac_pos = next_cell

                    # Collect reward
                    if pac_pos in reward_cells:
                        score += 1
                        reward_cells.remove(pac_pos)

                    # Check win
                    if pac_pos == goal:
                        playing = False
                        game_over = True
                        reason = "win"
                        final_score = score
                        end_time = now

            # Ghosts movement (BFS chase)
            if ghost_positions and pac_pos is not None and now - last_ghost_move_time >= GHOST_MOVE_INTERVAL:
                last_ghost_move_time = now
                new_ghost_positions = []
                for gpos in ghost_positions:
                    path_g = bfs_path(grid, gpos, pac_pos)
                    if path_g is not None and len(path_g) >= 2:
                        new_ghost_positions.append(path_g[1])
                    else:
                        new_ghost_positions.append(gpos)
                ghost_positions = new_ghost_positions

            # Check collision with any ghost
            if pac_pos is not None and ghost_positions and pac_pos in ghost_positions:
                playing = False
                game_over = True
                reason = "ghost"
                final_score = score
                end_time = pygame.time.get_ticks()

        # ---------- DRAW ----------
        draw_grid(WIN, grid)

        # Draw diamonds as diamond shapes
        for (r, c) in reward_cells:
            cx = c * CELL_SIZE + CELL_SIZE // 2
            cy = r * CELL_SIZE + CELL_SIZE // 2
            size = CELL_SIZE // 3
            points = [
                (cx, cy - size),   # top
                (cx + size, cy),   # right
                (cx, cy + size),   # bottom
                (cx - size, cy),   # left
            ]
            pygame.draw.polygon(WIN, (255, 215, 0), points)
            pygame.draw.polygon(WIN, (255, 255, 255), points, 2)

        # Draw all ghosts (blue)
        for (gr, gc) in ghost_positions:
            gx = gc * CELL_SIZE + CELL_SIZE // 2
            gy = gr * CELL_SIZE + CELL_SIZE // 2
            radius = CELL_SIZE // 2 - 3
            pygame.draw.circle(WIN, (0, 102, 255), (gx, gy), radius)       # body
            pygame.draw.circle(WIN, (255, 255, 255), (gx - 5, gy - 4), 4)  # eyes
            pygame.draw.circle(WIN, (255, 255, 255), (gx + 5, gy - 4), 4)
            pygame.draw.circle(WIN, (0, 0, 0), (gx - 5, gy - 4), 2)
            pygame.draw.circle(WIN, (0, 0, 0), (gx + 5, gy - 4), 2)

        # Draw Pac-Man
        if pac_pos is not None:
            pac_frame = pac_frames[pac_frame_idx]
            pac_frame_idx = (pac_frame_idx + 1) % len(pac_frames)
            px = pac_pos[1] * CELL_SIZE
            py = pac_pos[0] * CELL_SIZE
            WIN.blit(pac_frame, (px, py))

        draw_hud(WIN, score, start_time, end_time, game_over, final_score, reason)

        hint = FONT_SMALL.render(
            "L: load level   C: clear   SPACE: start   R: reset   Click: S/G/Walls   Shift+Click: diamond   G+Click: ghost",
            True, (160, 160, 255),
        )
        WIN.blit(hint, (10, HEIGHT - 25))

        pygame.display.update()


if __name__ == "__main__":
    main()
