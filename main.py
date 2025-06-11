import pygame
import random
import copy
from collections import deque
import pygame_gui
import threading

"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""

pygame.font.init()

# GLOBALS VARS
s_width = 1200
s_height = 700
play_width = 300 
play_height = 600  
block_size = 30

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height

game_started = False

#BOT VARS
executing_path = False
best_path = []

#GUI
win = pygame.display.set_mode((s_width, s_height))
pygame.display.set_caption('Tetris')

manager = pygame_gui.UIManager((s_width, s_height))
clock = pygame.time.Clock()


# SHAPE FORMATS

S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
# index 0 - 6 represent shape
shape_bag = None


class Piece(object):
    rows = 20  # y
    columns = 10  # x

    def __init__(self, column, row, shape):
        self.x = column
        self.y = row
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0  # number from 0-3

class GameState:
    def __init__(self):
        self.manager = pygame_gui.UIManager((s_width, s_height))
        self.last_move_breakdown = {}  
        self.lines_cleared_over_time = []
        self.run = True
        self.game_started = False
        self.current_piece = None
        self.change_piece = None
        self.hold_used = None
        self.held_piece = None
        self.fall_time = 0
        self.level_time = 0
        self.fall_speed = 0.27
        self.score = 0
        self.locked_positions = {}
        self.grid = create_grid({})
        self.next_pieces = [get_shape() for _ in range(5)]
        self.best_move = None
        self.best_path = None
        self.precomputed_best_move = None
        self.precomputed_best_path = None
        self.new_piece_spawned = False
        self.clock = pygame.time.Clock()
        self.weight_inputs = {}
        self.is_continuous_play = False
        self.frame_count = 0
        self.is_recalculating = False
        self.pending_best_move = None
        self.ga_best_output_text = "Best Genome: "
        self.best_genome = None


def create_grid(locked_positions={}):
    grid = [[(0,0,0) for x in range(10)] for x in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j,i) in locked_positions:
                c = locked_positions[(j,i)]
                grid[i][j] = c
    return grid


def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(shape, grid):
    accepted_positions = [[(j, i) for j in range(10) if grid[i][j] == (0,0,0)] for i in range(20)]
    accepted_positions = [j for sub in accepted_positions for j in sub]
    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_positions:
            if pos[1] > -1:
                return False

    return True


def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False


def get_shape():
    global shape_bag, shapes
    if not shape_bag:
        local_shapes = copy.deepcopy(shapes)
        random.shuffle(local_shapes)
        shape_bag = deque(local_shapes)
    return Piece(5, 0, shape_bag.popleft())


def draw_text_middle(text, size, color, surface):
    lines = text.split("\n")
    try:
        font = pygame.font.Font("Roboto-Regular.ttf", size)
    except OSError:
        font = pygame.font.Font(pygame.font.get_default_font(), size)

    total_height = len(lines) * font.get_linesize()
    y_offset = top_left_y + play_height / 2 - total_height / 2

    for i, line in enumerate(lines):
        label = font.render(line, True, color)
        x = top_left_x + play_width / 2 - label.get_width() / 2
        y = y_offset + i * font.get_linesize()
        surface.blit(label, (x, y))


def draw_grid(surface, row, col):
    sx = top_left_x
    sy = top_left_y
    for i in range(row):
        pygame.draw.line(surface, (128,128,128), (sx, sy+ i*30), (sx + play_width, sy + i * 30))  # horizontal lines
        for j in range(col):
            pygame.draw.line(surface, (128,128,128), (sx + j * 30, sy), (sx + j * 30, sy + play_height))  # vertical lines


def clear_rows(grid, locked):
    cleared_rows = []
    for i in range(len(grid)-1, -1, -1):
        row = grid[i]
        if (0, 0, 0) not in row:
            cleared_rows.append(i)
            for j in range(len(row)):
                try:
                    del locked[(j, i)]
                except:
                    continue

    if cleared_rows:
        cleared_rows.sort()
        for row in cleared_rows:
            # Shift everything above the cleared row down by 1
            new_locked = {}
            for (x, y), color in locked.items():
                if y < row:
                    new_locked[(x, y + 1)] = color
                else:
                    new_locked[(x, y)] = color
            locked.clear()
            locked.update(new_locked)

    return len(cleared_rows)


def draw_next_shape(shape, surface):
    try:
        font = pygame.font.Font("Roboto-Regular.ttf", 30)
    except OSError:
        font = pygame.font.Font(pygame.font.get_default_font(), 20) 
    label = font.render('Next Shape', 1, (255, 255, 255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height / 2 - 100
    pygame.draw.rect(surface, (255, 255, 255), (sx - 10, sy - 10, 150, 150), 2)

    surface.blit(label, (sx - 10, sy - 45))

    format = shape.shape[shape.rotation % len(shape.shape)]

    # Calculate bounding box of the shape
    min_x, max_x = 5, 0
    min_y, max_y = 5, 0
    for i, line in enumerate(format):
        for j, char in enumerate(line):
            if char == '0':
                min_x = min(min_x, j)
                max_x = max(max_x, j)
                min_y = min(min_y, i)
                max_y = max(max_y, i)

    shape_width = max_x - min_x + 1
    shape_height = max_y - min_y + 1

    block_size = 30
    offset_x = ((5 - shape_width) / 2 - min_x) * block_size
    offset_y = ((5 - shape_height) / 2 - min_y) * block_size

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(
                    surface,
                    shape.color,
                    (sx - 10 + j * block_size + offset_x, sy - 10 + i * block_size + offset_y, block_size, block_size),
                    0
                )

def draw_held_piece(surface, held_piece):
    try:
        font = pygame.font.Font("Roboto-Regular.ttf", 40)
    except OSError:
        font = pygame.font.Font(pygame.font.get_default_font(), 20)

    sx = top_left_x - play_width + 100
    sy = top_left_y + 100

    label = font.render('Hold', True, (255, 255, 255))
    surface.blit(label, (sx + 30, sy + 60))
    pygame.draw.rect(surface, (255, 255, 255), (sx, sy + 100, 150, 150), 2)

    if held_piece is not None:
        format = held_piece.shape[held_piece.rotation % len(held_piece.shape)]

        # Calculate bounding box of the shape
        min_x, max_x = 5, 0
        min_y, max_y = 5, 0
        for i, line in enumerate(format):
            for j, char in enumerate(line):
                if char == '0':
                    min_x = min(min_x, j)
                    max_x = max(max_x, j)
                    min_y = min(min_y, i)
                    max_y = max(max_y, i)

        shape_width = max_x - min_x + 1
        shape_height = max_y - min_y + 1

        # Size of each block (same as grid)
        block_size = 30

        # Calculate offset to center the piece in the box
        offset_x = ((5 - shape_width) / 2 - min_x) * block_size
        offset_y = ((5 - shape_height) / 2 - min_y) * block_size

        for i, line in enumerate(format):
            row = list(line)
            for j, column in enumerate(row):
                if column == '0':
                    pygame.draw.rect(
                        surface,
                        held_piece.color,
                        (sx + j * block_size + offset_x, sy + 100 + i * block_size + offset_y, block_size, block_size),
                        0
                    )

def draw_window(surface, game_state):
    surface.fill((0,0,0))
    # Tetris Title
    try:
        font = pygame.font.Font("Roboto-Regular.ttf", 60)
    except OSError:
        font = pygame.font.Font(pygame.font.get_default_font(), 20) 

    label = font.render('TETRIS', 1, (255,255,255))

    surface.blit(label, (top_left_x + play_width / 2 - (label.get_width() / 2), 30))

    for i in range(len(game_state.grid)):
        for j in range(len(game_state.grid[i])):
            pygame.draw.rect(surface, game_state.grid[i][j], (top_left_x + j* 30, top_left_y + i * 30, 30, 30), 0)

    # draw grid and border
    draw_grid(surface, 20, 10)
    pygame.draw.rect(surface, (255, 0, 0), (top_left_x, top_left_y, play_width, play_height), 5)
    # pygame.display.update()

def evaluate_board(lines, aggregate_height, bumpiness, well_depth, holes, support, weights, active_heuristics):
    score = 0
    if active_heuristics.get("lines_cleared", True):
        score += weights["lines_cleared"] * lines
    if active_heuristics.get("aggregate_height", True):
        score += weights["aggregate_height"] * aggregate_height
    if active_heuristics.get("bumpiness", True):
        score += weights["bumpiness"] * bumpiness
    if active_heuristics.get("well_depth", True):
        score += weights["well_depth"] * well_depth
    if active_heuristics.get("holes", True):
        score += weights["holes"] * holes
    if active_heuristics.get("support", True):
        score += weights["support"] * support
    return score

def support_score(grid, piece_shape, x, y):
    support = 0
    for py in range(5):
        for px in range(5):
            if piece_shape[py][px] == '0':
                gx = x + px
                gy = y + py
                if 0 <= gx < len(grid[0]) and 0 <= gy + 1 < len(grid):
                    if grid[gy + 1][gx] != (0, 0, 0):
                        support += 1
                else:
                    support += 1
    return support

def max_column_height(grid):
    min_row_index = 20
    for j in range(len(grid[0])):
        for i in range(len(grid)):
            if grid[i][j] != (0, 0, 0):
                if i < min_row_index:
                    min_row_index = i
                    break
    return 20 - min_row_index

def place_piece_on_grid(piece, grid):
    new_grid = [row[:] for row in grid]
    possible_pos = convert_shape_format(piece)
    for i, j in possible_pos:
            if 0 <= j < 20 and 0 <= i < 10:
                new_grid[j][i] = piece.color
    return new_grid

def compute_column_metrics(grid):
    num_cols = len(grid[0])
    num_rows = len(grid)
    
    heights = [0] * num_cols
    holes = 0
    well_depth = 0
    
    for x in range(num_cols):
        found_block = False
        for y in range(num_rows):
            if grid[y][x] != (0, 0, 0):
                if not found_block:
                    heights[x] = num_rows - y
                    found_block = True
            elif found_block:
                holes += 1

    aggregate_height = sum(heights)
    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(num_cols - 1))

    for x in range(num_cols):
        for y in range(num_rows):
            if grid[y][x] != (0, 0, 0):
                break
            left_filled = (x == 0 or grid[y][x - 1] != (0, 0, 0))
            right_filled = (x == num_cols - 1 or grid[y][x + 1] != (0, 0, 0))
            if left_filled and right_filled:
                well_depth += 1

    return aggregate_height, bumpiness, well_depth, holes

def recalculate_best_move_async(
    locked_positions_snapshot, current_piece_snapshot, next_pieces_snapshot,
    weights, active_heuristics, lookahead_depth, beam_width, game_state
):
    # Recreate grid for snapshot
    grid_snapshot = create_grid(locked_positions_snapshot)
    
    best_move, _ = choose_best_move(
        current_piece_snapshot, grid_snapshot, weights, active_heuristics,
        lookahead_depth=lookahead_depth, beam_width=beam_width,
        next_pieces=next_pieces_snapshot
    )

    game_state.pending_best_move = best_move
    game_state.is_recalculating = False

def choose_best_move(piece, grid, weights, active_heuristics, lookahead_depth=2, next_pieces=None, beam_width=3):
    all_moves = []

    for i in range(len(piece.shape)):  # Loop through all rotations
        for j in range(len(grid[0])):  # Loop through all x positions
            sim_piece = copy.deepcopy(piece)
            sim_piece.rotation = i
            sim_piece.x = j
            sim_piece.y = 0
            rotated_shape = sim_piece.shape[i % len(sim_piece.shape)]

            # Skip if out of horizontal bounds after rotation
            positions = convert_shape_format(sim_piece)
            if any(x < 0 or x >= len(grid[0]) or y >= len(grid) for x, y in positions):
                continue

            # Drop the piece
            while valid_space(sim_piece, grid):
                sim_piece.y += 1
            sim_piece.y -= 1

            # Final spot must be valid
            if not valid_space(sim_piece, grid):
                continue

            # Place the piece and create new board state
            new_grid = place_piece_on_grid(sim_piece, copy.deepcopy(grid))

            # Evaluate final board
            lines = count_cleared_rows(new_grid)
            support = support_score(new_grid, rotated_shape, sim_piece.x, sim_piece.y)
            aggregate_height, bumpiness, well_depth, holes = compute_column_metrics(new_grid)

            score = evaluate_board(
                lines, aggregate_height, bumpiness, well_depth, holes, support,
                weights, active_heuristics
            )

            all_moves.append({
                "score": score,
                "move": (i, j),
                "new_grid": new_grid
            })

    # Sort all final states by their score (highest first)
    all_moves.sort(key=lambda x: x["score"], reverse=True)

    # Keep only the top N moves (beam width)
    top_moves = all_moves[:beam_width]

    # If lookahead depth is 1, return the best move directly
    if lookahead_depth == 1 or not next_pieces or len(next_pieces) == 0:
        best_move = top_moves[0]["move"] if top_moves else None
        best_score = top_moves[0]["score"] if top_moves else float('-inf')
        return best_move, best_score

    # Otherwise, recursively evaluate next pieces
    best_score = float('-inf')
    best_move = None
    for move_data in top_moves:
        next_piece = next_pieces[0]
        # Recursively find best move for the next piece
        _, future_score = choose_best_move(
            next_piece,
            move_data["new_grid"],
            weights,
            active_heuristics,
            lookahead_depth=lookahead_depth - 1,
            next_pieces=next_pieces[1:],
            beam_width=beam_width
        )
        if future_score > best_score:
            best_score = future_score
            best_move = move_data["move"]

    return best_move, best_score

def count_cleared_rows(grid):
    return sum(1 for row in grid if (0, 0, 0) not in row)

def get_drop_y(piece, grid):
    temp_piece = copy.deepcopy(piece)
    while valid_space(temp_piece, grid):
        temp_piece.y += 1
    temp_piece.y -= 1
    return temp_piece.y

def find_path_to_target(start_piece, target_x, target_y, target_rotation, grid):
    visited = set()
    queue = deque()
    max_iterations = 5000  # total states to explore before giving up

    start_state = (start_piece.x, start_piece.y, start_piece.rotation)
    queue.append((start_state, []))
    visited.add(start_state)

    while queue and max_iterations > 0:
        (x, y, rot), path = queue.popleft()
        max_iterations -= 1  # Decrement iteration counter

        # Check for goal
        if (x, y, rot) == (target_x, target_y, target_rotation):
            return path

        # Try each move
        for move, dx, dy, drot in [
            ('left', -1, 0, 0),
            ('right', 1, 0, 0),
            ('down', 0, 1, 0),
            ('rotate_cw', 0, 0, 1),
        ]:
            new_x = x + dx
            new_y = y + dy
            new_rot = (rot + drot) % 4

            test_piece = copy.deepcopy(start_piece)
            test_piece.x = new_x
            test_piece.y = new_y
            test_piece.rotation = new_rot

            state = (new_x, new_y, new_rot)

            if state not in visited and valid_space(test_piece, grid):
                if len(path) < 20:  # Limit max path length
                    visited.add(state)
                    queue.append((state, path + [move]))

    return None  # No path found or search too long


def create_heuristic_ui(game_state):
    manager = game_state.manager
    sliders, toggles, weight_inputs = {}, {}, {}
    heuristic_elements = []

    heuristics = [
        ("lines_cleared", 10),
        ("aggregate_height", -0.5),
        ("bumpiness", -2.0),
        ("well_depth", -5.0),
        ("holes", -10.0),
        ("support", 2.0)
    ]

    for i, (name, default) in enumerate(heuristics):
        y = 30 + i * 60
        x_offset = top_left_x + play_width + 200 

        label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((x_offset, y), (140, 20)),
            text=name.replace("_", " ").title(),
            manager=manager
        )
        heuristic_elements.append(label)

        weight_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((x_offset + 150, y), (50, 20)),
            manager=manager
        )
        weight_input.set_text(str(default))
        weight_inputs[name] = weight_input
        heuristic_elements.append(weight_input)

        slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((x_offset, y + 25), (180, 20)),
            start_value=default,
            value_range=(-20.0, 20.0),
            manager=manager
        )
        sliders[name] = slider
        heuristic_elements.append(slider)

        toggle = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((x_offset + 190, y + 25), (50, 20)),
            text="ON",
            manager=manager
        )
        toggles[name] = toggle
        heuristic_elements.append(toggle)

    lookahead_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_offset + 0, 400), (150, 20)),
        text="Lookahead Depth:",
        manager=manager
    )
    heuristic_elements.append(lookahead_label)

    lookahead_input = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((x_offset + 190, 400), (50, 20)),
        manager=manager
    )
    lookahead_input.set_text("1")
    heuristic_elements.append(lookahead_input)

    beam_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_offset - 15, 420), (150, 20)),
        text="Beam Width:",
        manager=manager
    )
    heuristic_elements.append(beam_label)

    beam_width_input = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((x_offset + 190, 420), (50, 20)),
        manager=manager
    )
    beam_width_input.set_text("3")
    heuristic_elements.append(beam_width_input)

    game_state.heuristic_elements = heuristic_elements

    return sliders, toggles, weight_inputs, lookahead_input, beam_width_input


def create_ga_ui(game_state):
    manager = game_state.manager
    ga_elements = []

    x_offset = top_left_x - play_width - 50 
    y_start = 450  # Adjust this if it overlaps with other elements

    # GA Population
    pop_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_offset - 110, y_start + 60), (150, 20)),
        text="GA Population:",
        manager=manager
    )
    ga_elements.append(pop_label)

    ga_population_input = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((x_offset + 75, y_start + 60), (50, 20)),
        manager=manager
    )
    ga_population_input.set_text("20")
    ga_elements.append(ga_population_input)
    game_state.ga_population_input = ga_population_input

    # GA Generations
    gen_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_offset - 110, y_start + 90), (150, 20)),
        text="GA Generations:",
        manager=manager
    )
    ga_elements.append(gen_label)

    ga_generations_input = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((x_offset + 75, y_start + 90), (50, 20)),
        manager=manager
    )
    ga_generations_input.set_text("15")
    ga_elements.append(ga_generations_input)
    game_state.ga_generations_input = ga_generations_input

    mut_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_offset - 110, y_start + 120), (150, 20)),
        text="GA Mutation:",
        manager=manager
    )
    ga_elements.append(mut_label)

    ga_mutation_input = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((x_offset + 75, y_start + 120), (50, 20)),
        manager=manager
    )
    ga_mutation_input.set_text("0.1")
    ga_elements.append(ga_mutation_input)
    game_state.ga_mutation_input = ga_mutation_input

    ga_best_output = pygame_gui.elements.UITextBox(
        relative_rect=pygame.Rect((x_offset + 125, y_start + 30), (200, 200)),
        html_text="Best Genome: \nNone",
        manager=manager
    )
    ga_best_output.background_colour = pygame.Color(0, 0, 0, 100)
    ga_elements.append(ga_best_output)
    game_state.ga_best_output = ga_best_output

    max_pieces_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_offset - 110, y_start + 150), (150, 20)),
        text="Max Pieces:",
        manager=manager
    )
    ga_elements.append(max_pieces_label)

    max_pieces_input = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((x_offset + 75, y_start + 150), (50, 20)),
        manager=manager
    )
    max_pieces_input.set_text("50")
    ga_elements.append(max_pieces_input)
    game_state.max_pieces_input = max_pieces_input
    
    ga_run_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((x_offset - 75, y_start + 200), (90, 30)),
        text="Run GA",
        manager=manager
    )
    ga_elements.append(ga_run_button)
    game_state.ga_run_button = ga_run_button

    ga_replay_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((x_offset + 15, y_start + 200), (90, 30)),
        text="Replay Best",
        manager=manager
    )
    ga_elements.append(ga_replay_button)
    game_state.ga_replay_button = ga_replay_button

    game_state.ga_elements = ga_elements

def create_ui(game_state):
    game_state.sliders, game_state.toggles, game_state.weight_inputs, \
    game_state.lookahead_input, game_state.beam_width_input, = create_heuristic_ui(game_state)

    create_ga_ui(game_state)

    game_state.heuristic_toggle_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((top_left_x + play_width + 25, 550), (100, 40)),
        text="Heuristics",
        manager=game_state.manager
    )

    game_state.ga_toggle_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((top_left_x + play_width + 125, 550), (100, 40)),
        text="GA",
        manager=game_state.manager
    )

    start_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((top_left_x + play_width + 25, 600), (100, 40)),
        text="Start Game",
        manager=game_state.manager
    )

    restart_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((top_left_x + play_width + 125, 600), (100, 40)),
        text="Restart Game",
        manager=game_state.manager
    )

    game_state.heuristics_ui_active = False
    game_state.ga_ui_active = False
    game_state.start_button = start_button
    game_state.restart_button = restart_button

def restart_game(game_state, weights, active_heuristics, lookahead_depth=1):
    print("Restarting game logic...")

    # Reset game state
    game_state.hold_used = False
    game_state.held_piece = None
    game_state.new_piece_spawned = False
    game_state.best_move = None
    game_state.best_path = None
    game_state.precomputed_best_move = None
    game_state.precomputed_best_path = None

    # Reset the board and pieces
    game_state.locked_positions = {}
    game_state.grid = create_grid(game_state.locked_positions)
    game_state.next_pieces = [get_shape() for _ in range(5)]
    game_state.current_piece = game_state.next_pieces.pop(0)
    game_state.next_pieces.append(get_shape())
    game_state.change_piece = False


    # Compute the best moves
    game_state.best_move, game_state.best_path = choose_best_move(game_state.current_piece, game_state.grid, weights, active_heuristics, 
                                                                                  lookahead_depth=lookahead_depth)
    game_state.precomputed_best_move, game_state.precomputed_best_path = choose_best_move( game_state.next_pieces[0], game_state.grid, weights, active_heuristics, 
                                                                                          lookahead_depth=lookahead_depth)

    print("Best move selected (initial):", game_state.best_move)

    # Reset timers and control flags
    game_state.clock = pygame.time.Clock()
    game_state.fall_time = 0
    game_state.level_time = 0
    game_state.fall_speed = 0.27
    game_state.score = 0
    game_state.run = True
    game_state.game_started = False

def draw_heuristic_bars(surface, breakdown, pos_x=30, pos_y= 30):
    if not breakdown:
        return

    max_width = 200
    bar_height = 20
    spacing = 10
    font = pygame.font.SysFont("Arial", 16)

    max_val = max(abs(v) for v in breakdown.values()) or 1

    for i, (key, value) in enumerate(breakdown.items()):
        bar_length = int((abs(value) / max_val) * max_width)
        color = (0, 255, 0) if value >= 0 else (255, 0, 0)
        pygame.draw.rect(surface, color, (pos_x, pos_y + i * (bar_height + spacing), bar_length, bar_height))
        label = font.render(f"{key}: {value:.2f}", True, (255, 255, 255))
        surface.blit(label, (pos_x + bar_length + 5, pos_y + i * (bar_height + spacing)))

def draw_performance_chart(surface, performance, pos_x=30, pos_y=210, width=150, height=100):
    if len(performance) < 2:
        return

    max_val = max(performance) or 1
    step_x = width / (len(performance) - 1)
    points = [(pos_x + i * step_x, pos_y + height - (score / max_val) * height) for i, score in enumerate(performance)]
    pygame.draw.lines(surface, (0, 255, 0), False, points, 2)
    pygame.draw.rect(surface, (255, 255, 255), (pos_x, pos_y, width, height), 1)

def draw_total_score(surface, score, pos_x=top_left_x + play_width + 10, pos_y=100):
    font = pygame.font.SysFont("Arial", 18)
    score_text = font.render(f"Total Score: {score}", True, (255, 255, 255))
    surface.blit(score_text, (pos_x, pos_y))

def get_weights(weight_inputs):
    weights = {}
    for name, input_box in weight_inputs.items():
        try:
            val = float(input_box.get_text())
            weights[name] = val
        except ValueError:
            weights[name] = 0.0  # fallback if input box is empty/invalid
    return weights


def get_active_heuristics(toggles):
    return {key: toggles[key].text == "ON" for key in toggles}


def simulate_game(weights, active_heuristics, max_pieces=100):
    game_state = GameState()
    restart_game(game_state, weights, active_heuristics)
    piece_count = 0
    while True:
        if not valid_space(game_state.current_piece, game_state.grid):
            break  # Game over

        # Let the bot pick and place the piece
        best_move, _ = choose_best_move(
            game_state.current_piece, game_state.grid, weights, active_heuristics,
            lookahead_depth=1
        )
        if best_move is None:
            break

        # Snap to final Y
        game_state.current_piece.rotation = best_move[0]
        game_state.current_piece.x = best_move[1]
        game_state.current_piece.y = get_drop_y(game_state.current_piece, game_state.grid)

        # Lock piece
        for pos in convert_shape_format(game_state.current_piece):
            if pos[1] > -1:
                game_state.locked_positions[(pos[0], pos[1])] = game_state.current_piece.color

        game_state.grid = create_grid(game_state.locked_positions)
        lines_cleared = clear_rows(game_state.grid, game_state.locked_positions)
        game_state.lines_cleared_over_time.append(lines_cleared)

        game_state.current_piece = game_state.next_pieces.pop(0)
        game_state.next_pieces.append(get_shape())
        piece_count += 1

        if piece_count >= max_pieces:
            break

    total_lines_cleared = sum(game_state.lines_cleared_over_time)
    return total_lines_cleared

def run_genetic_algorithm(game_state, generations=15, pop_size=20, mutation_rate=0.1):
    # Initial random population
    heuristics = ["lines_cleared", "aggregate_height", "bumpiness", "well_depth", "holes", "support"]
    population = []
    for _ in range(pop_size):
        genome = {h: random.uniform(-10, 10) for h in heuristics}
        genome["lines_cleared"] = random.uniform(1, 20)
        #genome["holes"] = random.uniform(-20, -2)
        #genome["bumpiness"] = random.uniform(-20, -1)
        #genome["support"] = random.uniform(1, 5)
        #genome["well_depth"] = random.uniform(-20, 10)
        population.append(genome)

    best_genome = None
    best_fitness = -1

    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}")

        fitness_scores = []
        for genome in population:
            active_heuristics = {h: True for h in heuristics} 
            fitness = simulate_game(genome, active_heuristics)
            fitness_scores.append((fitness, genome))
            if fitness > best_fitness:
                best_fitness = fitness
                best_genome = copy.deepcopy(genome)

        fitness_scores.sort(reverse=True, key=lambda x: x[0])

        next_gen = [copy.deepcopy(fitness_scores[0][1]), copy.deepcopy(fitness_scores[1][1])]

        # Generate rest via crossover and mutation
        while len(next_gen) < pop_size:
            parent1 = random.choice(fitness_scores[:10])[1]
            parent2 = random.choice(fitness_scores[:10])[1]
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_gen.append(child)
            #child["holes"] = min(max(child["holes"], -20), -2)
            #child["support"] = min(max(child["support"], 1), 5)
            #child["bumpiness"] = min(max(child["bumpiness"], -20), -1)
            #child["well_depth"] = min(max(child["well_depth"], -20), 2)

            if genome["lines_cleared"] < 0:
                genome["lines_cleared"] = abs(genome["lines_cleared"])

        population = next_gen

        best_display = f"<font size=4>Gen {gen+1} Best: {fitness_scores[0][0]} lines<br>"
        for k, v in fitness_scores[0][1].items():
            best_display += f"{k}: {v:.1f}<br>"
        best_display += "</font>"

        game_state.ga_best_output_text = best_display 
        print(f"Best fitness this generation: {fitness_scores[0][0]}", flush=True)
        print(f"Best genome: {fitness_scores[0][1]}", flush=True)

    print("\nFinal best genome and score:")
    print(best_genome, "Lines cleared:", best_fitness)
    game_state.best_genome = best_genome

def crossover(parent1, parent2):
    child = {}
    for h in parent1.keys():
        child[h] = random.choice([parent1[h], parent2[h]])
    return child

def mutate(genome, rate=0.1):
    for h in genome.keys():
        if random.random() < rate:
            genome[h] += random.uniform(-1.0, 1.0)
    return genome

def replay_best_genome(game_state, best_genome):
    print("\nReplaying best genome in GUI mode...")
    active_heuristics = {h: True for h in best_genome}
    restart_game(game_state, best_genome, {h: True for h in best_genome})
    game_state.game_started = True




def main(weights, active_heuristics):
    game_state = GameState()
    create_ui(game_state)

    try:
        lookahead_depth = int(game_state.lookahead_input.get_text())
        lookahead_depth = max(1, lookahead_depth)
    except ValueError:
        lookahead_depth = 1
    restart_game(game_state, weights, active_heuristics, lookahead_depth)

    try:
        beam_width = int(game_state.beam_width_input.get_text())
        beam_width = max(1, min(beam_width, 20))  # Clamp to sensible range
    except ValueError:
        beam_width = 3  # Fallback

    while game_state.run:

        time_delta = game_state.clock.tick(60) / 1000.0
        win.fill((0, 0, 0))

        if game_state.ga_best_output_text is not None:
            game_state.ga_best_output.set_text(game_state.ga_best_output_text)
            game_state.ga_best_output_text = None  # clear it so we don't overwrite scrolling

        try:
            lookahead_depth = int(game_state.lookahead_input.get_text())
            lookahead_depth = max(1, lookahead_depth)
        except ValueError:
            lookahead_depth = 1

        try:
            beam_width = int(game_state.beam_width_input.get_text())
            beam_width = max(1, min(beam_width, 20))  
        except ValueError:
            beam_width = 3  

        for name, input_box in game_state.weight_inputs.items():
            if input_box.is_focused:
                try:
                    val = float(input_box.get_text())
                    val_clamped = max(min(val, 20.0), -20.0)
                    game_state.sliders[name].set_current_value(val_clamped)
                    # Do not update input box text here
                except ValueError:
                    pass
            else:
                # Only update input box if it’s within slider range
                try:
                    val = float(input_box.get_text())
                    if -20.0 <= val <= 20.0:
                        slider_val = game_state.sliders[name].get_current_value()
                        input_box.set_text(f"{slider_val:.1f}")
                except ValueError:
                    slider_val = game_state.sliders[name].get_current_value()
                    input_box.set_text(f"{slider_val:.1f}")

        weights = get_weights(game_state.weight_inputs)
        active_heuristics = get_active_heuristics(game_state.toggles)


        for event in pygame.event.get():
            game_state.manager.process_events(event)
            if event.type == pygame.QUIT:
                pygame.display.quit()
                quit()
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == game_state.start_button:
                    print("Start button pressed!")
                    game_state.game_started = True

                    if game_state.current_piece is None:
                        game_state.current_piece = game_state.next_pieces.pop(0)
                        game_state.next_pieces.append(get_shape())

                    game_state.best_move, game_state.best_path = choose_best_move(game_state.current_piece, game_state.grid, weights, active_heuristics, 
                                                                                  lookahead_depth=lookahead_depth)
                    
                elif event.ui_element == game_state.restart_button:
                    restart_game(game_state, weights, active_heuristics)

                elif event.ui_element == game_state.ga_run_button:
                    pop_size = int(game_state.ga_population_input.get_text())
                    gens = int(game_state.ga_generations_input.get_text())
                    mut_rate = float(game_state.ga_mutation_input.get_text())

                    threading.Thread(
                        target=run_genetic_algorithm,
                        args=(game_state, gens, pop_size, mut_rate),
                        daemon=True
                    ).start()

                elif event.ui_element == game_state.ga_replay_button:
                    print("Replay button pressed")
                    if game_state.best_genome is not None:
                        print("Replaying best genome in GUI mode...")
                        active_heuristics = {h: True for h in game_state.best_genome}
                        weights = game_state.best_genome
                        active_heuristics = {h: True for h in weights}
                        restart_game(game_state, weights, active_heuristics)
                        game_state.best_move, game_state.best_path = choose_best_move(game_state.current_piece, game_state.grid, weights, active_heuristics, 
                                                                                      lookahead_depth=1)
                        game_state.game_started = True

                if event.ui_element == game_state.heuristic_toggle_button:
                    game_state.heuristics_ui_active = not game_state.heuristics_ui_active
                    for elem in game_state.heuristic_elements:
                        if game_state.heuristics_ui_active:
                            elem.show()
                        else:
                            elem.hide()

                elif event.ui_element == game_state.ga_toggle_button:
                    game_state.ga_ui_active = not game_state.ga_ui_active
                    for elem in game_state.ga_elements:
                        if game_state.ga_ui_active:
                            elem.show()
                        else:
                            elem.hide()

                for key, button in game_state.toggles.items():
                    if event.ui_element == button:
                        button.set_text("OFF" if button.text == "ON" else "ON")

            if event.type == pygame.KEYDOWN and event.key == pygame.K_LSHIFT:
                if game_state.hold_used == False:
                    if game_state.held_piece == None:
                        game_state.held_piece = game_state.current_piece
                        game_state.current_piece = game_state.next_pieces.pop(0)
                        game_state.next_pieces.append(get_shape())
                    else:
                        game_state.held_piece, game_state.current_piece = game_state.current_piece, game_state.held_piece
                        game_state.new_piece_spawned = True
                    game_state.hold_used = True

        shape_pos = convert_shape_format(game_state.current_piece)

        game_state.manager.update(time_delta)

        draw_window(win, game_state)
        draw_next_shape(game_state.next_pieces[0], win)
        draw_held_piece(win, game_state.held_piece)
        game_state.manager.draw_ui(win)
        draw_heuristic_bars(win, game_state.last_move_breakdown)
        draw_performance_chart(win, game_state.lines_cleared_over_time)
        draw_total_score(win, game_state.score)
        pygame.display.update()

        game_state.grid = create_grid(game_state.locked_positions)
        game_state.fall_time += game_state.clock.get_rawtime()
        game_state.level_time += game_state.clock.get_rawtime()
        game_state.clock.tick()

        if game_state.game_started:
            shape_pos = convert_shape_format(game_state.current_piece)
            if game_state.level_time/1000 > 4:
                game_state.level_time = 0
                if game_state.fall_speed > 0.15:
                    game_state.fall_speed -= 0.005
                

            # PIECE FALLING CODE
            if game_state.fall_time/1000 >= game_state.fall_speed:
                game_state.fall_time = 0
                game_state.current_piece.y += 1
                if not (valid_space(game_state.current_piece, game_state.grid)) and game_state.current_piece.y > 0:
                    game_state.current_piece.y -= 1
                    game_state.change_piece = True 

            if game_state.change_piece:
                    game_state.hold_used = False
                    if game_state.pending_best_move is not None:
                        # Update best_move only if thread finished
                        game_state.best_move = game_state.pending_best_move
                        game_state.pending_best_move = None  # Clear it
            # Only run once per new piece
            if game_state.best_move is None:
                game_state.best_move, _ = choose_best_move(game_state.current_piece, game_state.grid, weights, active_heuristics, 
                                                           lookahead_depth=lookahead_depth, beam_width=beam_width, next_pieces=game_state.next_pieces)
            if game_state.best_move is None and game_state.pending_best_move is not None:
                game_state.best_move = game_state.pending_best_move
                game_state.pending_best_move = None

            game_state.frame_count += 1
            if game_state.is_continuous_play:
                # If not already recalculating, start a new thread
                if not game_state.is_recalculating and game_state.frame_count % 5 == 0:
                    locked_positions_snapshot = copy.deepcopy(game_state.locked_positions)
                    current_piece_snapshot = copy.deepcopy(game_state.current_piece)
                    next_pieces_snapshot = copy.deepcopy(game_state.next_pieces)

                    game_state.is_recalculating = True
                    threading.Thread(
                        target=recalculate_best_move_async,
                        args=(
                            locked_positions_snapshot, current_piece_snapshot, next_pieces_snapshot,
                            weights, active_heuristics, lookahead_depth, beam_width,
                            game_state
                        )
                    ).start()
            else:
                # In thinking mode, calculate only once
                if game_state.best_move is None:
                    game_state.best_move, _ = choose_best_move(
                        game_state.current_piece, game_state.grid, weights, active_heuristics,
                        lookahead_depth=lookahead_depth, beam_width=beam_width,
                        next_pieces=game_state.next_pieces
                    )



            if game_state.best_move is not None:
                target_rot, target_x = game_state.best_move

                # Rotate toward target
                if game_state.current_piece.rotation != target_rot:
                    game_state.current_piece.rotation = (game_state.current_piece.rotation + 1) % len(game_state.current_piece.shape)
                    if not valid_space(game_state.current_piece, game_state.grid):
                        game_state.current_piece.rotation = (game_state.current_piece.rotation - 1) % len(game_state.current_piece.shape)

                # Move horizontally toward target
                elif game_state.current_piece.x < target_x:
                    game_state.current_piece.x += 1
                    if not valid_space(game_state.current_piece, game_state.grid):
                        game_state.current_piece.x -= 1
                elif game_state.current_piece.x > target_x:
                    game_state.current_piece.x -= 1
                    if not valid_space(game_state.current_piece, game_state.grid):
                        game_state.current_piece.x += 1

                # Drop once position and rotation are correct
                else:
                    drop_y = get_drop_y(game_state.current_piece, game_state.grid)
                    if game_state.current_piece.y < drop_y:
                        game_state.current_piece.y += 1
                    else:
                        drop_y = get_drop_y(game_state.current_piece, game_state.grid)
                        game_state.current_piece.y = drop_y  # SNAP directly

                        # Recalculate shape_pos after snapping
                        shape_pos = convert_shape_format(game_state.current_piece)

                        game_state.change_piece = True
                        game_state.best_move = None

                        # add piece to the grid for drawing (using updated shape_pos)
                        for x, y in shape_pos:
                            if y > -1:
                                game_state.grid[y][x] = game_state.current_piece.color

                    # add piece to the grid for drawing
                    for i in range(len(shape_pos)):
                        x, y = shape_pos[i]
                        if y > -1:
                            game_state.grid[y][x] = game_state.current_piece.color
            
            else:
                continue

            # IF PIECE HIT GROUND
            if game_state.change_piece:
                #Lock the current piece
                for pos in shape_pos:
                    p = (pos[0], pos[1])
                    game_state.locked_positions[p] = game_state.current_piece.color

                game_state.grid = create_grid(game_state.locked_positions)

                lines = count_cleared_rows(game_state.grid)
                support = support_score(game_state.grid, game_state.current_piece.shape[game_state.current_piece.rotation % len(game_state.current_piece.shape)],
                                        game_state.current_piece.x, game_state.current_piece.y)
                aggregate_height, bumpiness, well_depth, holes = compute_column_metrics(game_state.grid)

                # Store the breakdown
                game_state.last_move_breakdown = {
                    "lines_cleared": weights["lines_cleared"] * lines,
                    "aggregate_height": weights["aggregate_height"] * aggregate_height,
                    "bumpiness": weights["bumpiness"] * bumpiness,
                    "well_depth": weights["well_depth"] * well_depth,
                    "holes": weights["holes"] * holes,
                    "support": weights["support"] * support
                }

                # Track total lines cleared over time
                game_state.lines_cleared_over_time.append(game_state.score)

                #Precompute the move for the next piece using the updated grid
                game_state.precomputed_best_move, game_state.precomputed_best_path = choose_best_move(game_state.next_pieces[0], game_state.grid, weights, 
                                                                                                      active_heuristics, lookahead_depth=lookahead_depth)

                game_state.current_piece = game_state.next_pieces.pop(0)
                game_state.next_pieces.append(get_shape())

                game_state.best_move = None

                # game_state.best_move, game_state.best_path = game_state.precomputed_best_move, game_state.precomputed_best_path

                game_state.change_piece = False
                game_state.hold_used = False

                if game_state.best_move is not None:
                    game_state.current_piece.rotation = game_state.best_move[0]
                    game_state.current_piece.x = game_state.best_move[1]
                    game_state.current_piece.y = 0
                
                if not valid_space(game_state.current_piece, game_state.grid) or check_lost(game_state.locked_positions):
                    print("Game Over")
                    game_over_screen(game_state.score, game_state, weights, active_heuristics)


                #Check for cleared rows
                lines_cleared = clear_rows(game_state.grid, game_state.locked_positions)
                if lines_cleared > 0:
                    game_state.score += {1: 100, 2: 300, 3: 500, 4: 800}.get(lines_cleared, 0)
                    print(game_state.score)
                    draw_window(win, game_state)
                    draw_next_shape(game_state.next_pieces[0], win)
                    draw_held_piece(win, game_state.held_piece)
                    pygame.display.update()

                    # Check if user lost
                    if check_lost(game_state.locked_positions):
                        print("Game Over: Blocks reached the top.")
                        draw_window(win, game_state, game_state.grid, game_state.score)
                        draw_text_middle("You Lost", 40, (255, 255, 255), win)
                        pygame.display.update()
                        pygame.time.delay(2000)
                        game_state.run = False
                        return

# Show game over screen and wait for key press
def game_over_screen(score, game_state, weights, active_heuristics):
    draw_window(win, game_state)
    draw_text_middle(f"You Lost\nScore: {score}\nPress any key to restart", 20, (255, 255, 255), win)
    pygame.display.update()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            """if event.type == pygame.QUIT:
                pygame.quit()
                quit()"""
            if event.type == pygame.KEYDOWN:
                restart_game(game_state, weights, active_heuristics)
                waiting = False  


# Setup window
win = pygame.display.set_mode((s_width, s_height))
pygame.display.set_caption('Tetris')
manager = pygame_gui.UIManager((s_width, s_height))
clock = pygame.time.Clock()

def create_initial_heuristics():
    heuristics = [
        ("lines_cleared", 10),
        ("aggregate_height", -0.5),
        ("bumpiness", -2.0),
        ("well_depth", -5.0),
        ("holes", -10.0),
        ("support", 2.0)
    ]
    weights = {name: default for name, default in heuristics}
    active_heuristics = {name: True for name, _ in heuristics}
    return weights, active_heuristics

def main_menu():
    weights, active_heuristics = create_initial_heuristics()
    main(weights, active_heuristics)
    pygame.display.update()

# Call it
main_menu()






