#!/usr/bin/env python3
"""Render the Branching Rooms layout and a valid scripted walkthrough.

This utility intentionally has no Gymnasium or MiniGrid dependency. It reads the
fixed layout constants from ``branching_rooms_env.py`` so that environment
visuals can still be generated on analysis machines without the training stack.
The animated local view is a schematic of the 7x7 egocentric field of view; the
top-down map and agent transitions use the exact environment geometry.
"""

import argparse
import ast
from collections import deque
from pathlib import Path
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw, ImageFont


Position = Tuple[int, int]
Room = Tuple[int, int]
RoomEdge = Tuple[Room, Room]
Pose = Tuple[int, int, int]


ENV_SOURCE = Path(__file__).resolve().parents[1] / "zoo/minigrid/envs/branching_rooms_env.py"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "video/branching_rooms_visuals"

WALL_COLOR = (31, 36, 42)
GRID_COLOR = (65, 71, 78)
UNKNOWN_COLOR = (14, 17, 20)
DOOR_COLOR = (190, 195, 201)
ROUTE_COLOR = (20, 184, 166)
AGENT_COLOR = (250, 204, 21)
DEAD_END_COLOR = (220, 38, 38)
TEXT_COLOR = (235, 238, 241)
MUTED_TEXT_COLOR = (164, 171, 180)
PANEL_COLOR = (24, 28, 33)

FLOOR_COLORS = {
    "red": (177, 75, 72),
    "green": (70, 139, 96),
    "blue": (70, 112, 171),
    "purple": (135, 91, 164),
    "grey": (116, 123, 131),
    "yellow": (190, 156, 62),
}

DIRECTION_NAMES = ("E", "S", "W", "N")
DIRECTION_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))
ACTION_NAMES = {0: "Turn left", 1: "Turn right", 2: "Forward"}


def _font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _outlined_rectangle(
        draw: ImageDraw.ImageDraw,
        bounds: Sequence[int],
        color: Tuple[int, int, int],
        width: int,
) -> None:
    left, top, right, bottom = bounds
    for inset in range(max(1, width)):
        draw.rectangle(
            [left + inset, top + inset, right - inset, bottom - inset],
            outline=color,
        )


def _outlined_text(
        draw: ImageDraw.ImageDraw,
        position: Tuple[float, float],
        label: str,
        font,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        outline_width: int = 1,
) -> None:
    x, y = position
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), label, font=font, fill=outline)
    draw.text((x, y), label, font=font, fill=fill)


def _load_class_literals(source_path: Path) -> Dict[str, object]:
    wanted = {
        "size",
        "room_count_per_axis",
        "room_bounds",
        "room_centers",
        "start_room",
        "start_position",
        "start_direction",
        "room_edges",
        "doorway_positions",
        "room_color_pairs",
    }
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BranchingRoomsEnv"
    )
    values = {}
    for node in class_node.body:
        name = None
        value_node = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value_node = node.value
        if name in wanted and value_node is not None:
            values[name] = ast.literal_eval(value_node)

    missing = wanted.difference(values)
    if missing:
        raise RuntimeError("Could not read layout constants: {}".format(sorted(missing)))
    return values


class Layout:
    def __init__(self, values: Dict[str, object]):
        self.size = int(values["size"])
        self.room_count_per_axis = int(values["room_count_per_axis"])
        self.room_bounds = tuple(values["room_bounds"])
        self.room_centers = tuple(values["room_centers"])
        self.start_room = tuple(values["start_room"])
        self.start_position = tuple(values["start_position"])
        self.start_direction = int(values["start_direction"])
        self.room_edges = tuple(values["room_edges"])
        self.doorway_positions = dict(values["doorway_positions"])
        self.room_color_pairs = tuple(values["room_color_pairs"])

        self.room_by_position = {}
        self.primary_by_position = {}
        self.secondary_positions = set()
        self.floor_positions = set()
        for row, (y_min, y_max) in enumerate(self.room_bounds):
            for col, (x_min, x_max) in enumerate(self.room_bounds):
                room = (row, col)
                room_index = row * self.room_count_per_axis + col
                primary, _ = self.room_color_pairs[room_index]
                for x in range(x_min, x_max + 1):
                    for y in range(y_min, y_max + 1):
                        position = (x, y)
                        self.floor_positions.add(position)
                        self.room_by_position[position] = room
                        self.primary_by_position[position] = primary
                center_x, center_y = self.room_centers[room_index]
                self.secondary_positions.update({
                    (center_x - 1, center_y - 1),
                    (center_x, center_y - 1),
                    (center_x - 1, center_y),
                    (center_x, center_y),
                })

        self.door_positions = set(self.doorway_positions.values())
        self.floor_positions.update(self.door_positions)
        self.graph = {
            (row, col): set()
            for row in range(self.room_count_per_axis)
            for col in range(self.room_count_per_axis)
        }
        for room_a, room_b in self.room_edges:
            self.graph[tuple(room_a)].add(tuple(room_b))
            self.graph[tuple(room_b)].add(tuple(room_a))
        self.dead_end_rooms = {
            room for room, neighbors in self.graph.items() if len(neighbors) == 1
        }
        reachable = self._reachable_positions(self.start_position)
        if len(reachable) != 693:
            raise RuntimeError("Expected 693 reachable positions, found {}".format(len(reachable)))

    def _reachable_positions(self, start: Position) -> Set[Position]:
        reachable = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in self.floor_positions and neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        return reachable

    def room_center(self, room: Room) -> Position:
        index = room[0] * self.room_count_per_axis + room[1]
        return tuple(self.room_centers[index])

    def color_at(self, position: Position) -> Tuple[int, int, int]:
        if position in self.door_positions:
            return DOOR_COLOR
        room = self.room_by_position.get(position)
        if room is None:
            return WALL_COLOR
        index = room[0] * self.room_count_per_axis + room[1]
        primary, secondary = self.room_color_pairs[index]
        name = secondary if position in self.secondary_positions else primary
        return FLOOR_COLORS[name]


def _shortest_path(layout: Layout, start: Position, goal: Position) -> List[Position]:
    queue = deque([start])
    parent = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        x, y = current
        for neighbor in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            if neighbor in layout.floor_positions and neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)
    if goal not in parent:
        raise RuntimeError("No path between {} and {}".format(start, goal))
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def _room_dfs_walk(layout: Layout) -> List[Room]:
    walk = [layout.start_room]
    visited = {layout.start_room}

    def visit(room: Room) -> None:
        neighbors = sorted(layout.graph[room], key=lambda item: (item[0], item[1]))
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            walk.append(neighbor)
            visit(neighbor)
            walk.append(room)

    visit(layout.start_room)
    return walk


def _scripted_positions(layout: Layout) -> Tuple[List[Position], List[Room]]:
    room_walk = _room_dfs_walk(layout)
    positions = [layout.start_position]
    current = layout.start_position
    for room in room_walk[1:]:
        segment = _shortest_path(layout, current, layout.room_center(room))
        positions.extend(segment[1:])
        current = segment[-1]
    return positions, room_walk


def _poses_and_actions(layout: Layout, positions: Sequence[Position]) -> Tuple[List[Pose], List[Optional[int]]]:
    direction = layout.start_direction
    x, y = positions[0]
    poses = [(x, y, direction)]
    actions = [None]
    for next_x, next_y in positions[1:]:
        dx, dy = next_x - x, next_y - y
        desired_direction = DIRECTION_VECTORS.index((dx, dy))
        clockwise = (desired_direction - direction) % 4
        if clockwise == 3:
            direction = (direction - 1) % 4
            poses.append((x, y, direction))
            actions.append(0)
        else:
            for _ in range(clockwise):
                direction = (direction + 1) % 4
                poses.append((x, y, direction))
                actions.append(1)
        x, y = next_x, next_y
        poses.append((x, y, direction))
        actions.append(2)
    return poses, actions


def _draw_agent(draw: ImageDraw.ImageDraw, x: float, y: float, direction: int, size: float) -> None:
    forward_x, forward_y = DIRECTION_VECTORS[direction]
    right_x, right_y = -forward_y, forward_x
    nose = (x + forward_x * size, y + forward_y * size)
    back_left = (
        x - forward_x * size * 0.65 - right_x * size * 0.7,
        y - forward_y * size * 0.65 - right_y * size * 0.7,
    )
    back_right = (
        x - forward_x * size * 0.65 + right_x * size * 0.7,
        y - forward_y * size * 0.65 + right_y * size * 0.7,
    )
    draw.polygon([nose, back_left, back_right], fill=AGENT_COLOR, outline=(63, 48, 6))


def _map_base(layout: Layout, cell_size: int, labels: bool = False) -> Image.Image:
    side = layout.size * cell_size
    image = Image.new("RGB", (side, side), WALL_COLOR)
    draw = ImageDraw.Draw(image)
    for y in range(layout.size):
        for x in range(layout.size):
            position = (x, y)
            color = layout.color_at(position)
            x0, y0 = x * cell_size, y * cell_size
            draw.rectangle(
                [x0, y0, x0 + cell_size - 1, y0 + cell_size - 1],
                fill=color,
                outline=GRID_COLOR if cell_size >= 12 else None,
            )

    for room in layout.dead_end_rooms:
        row, col = room
        x_min, x_max = layout.room_bounds[col]
        y_min, y_max = layout.room_bounds[row]
        _outlined_rectangle(
            draw,
            [
                x_min * cell_size,
                y_min * cell_size,
                (x_max + 1) * cell_size - 1,
                (y_max + 1) * cell_size - 1,
            ],
            DEAD_END_COLOR,
            max(2, cell_size // 5),
        )

    if labels:
        room_font = _font(max(11, cell_size - 3), bold=True)
        dead_font = _font(max(8, cell_size // 2 + 1), bold=True)
        for room in sorted(layout.graph):
            center_x, center_y = layout.room_center(room)
            label = "R{},{}".format(*room)
            width, height = draw.textsize(label, font=room_font)
            _outlined_text(
                draw,
                ((center_x + 0.5) * cell_size - width / 2, (center_y + 0.5) * cell_size - height / 2),
                label,
                room_font,
                (250, 250, 250),
                (20, 22, 25),
                2,
            )
            if room in layout.dead_end_rooms:
                dead_label = "DEAD END"
                dead_width, _ = draw.textsize(dead_label, font=dead_font)
                _outlined_text(
                    draw,
                    ((center_x + 0.5) * cell_size - dead_width / 2, (center_y + 2.0) * cell_size),
                    dead_label,
                    dead_font,
                    (255, 230, 230),
                    (80, 12, 12),
                    1,
                )
    return image


def _draw_route(
        image: Image.Image,
        poses: Sequence[Pose],
        upto: int,
        cell_size: int,
        origin: Tuple[int, int] = (0, 0),
) -> None:
    draw = ImageDraw.Draw(image)
    points = []
    previous = None
    for x, y, _ in poses[:upto + 1]:
        position = (x, y)
        if position == previous:
            continue
        points.append((origin[0] + (x + 0.5) * cell_size, origin[1] + (y + 0.5) * cell_size))
        previous = position
    if len(points) > 1:
        draw.line(points, fill=ROUTE_COLOR, width=max(2, cell_size // 4))


def _line_cells(start: Position, end: Position) -> List[Position]:
    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy
    cells = []
    while True:
        cells.append((x0, y0))
        if (x0, y0) == (x1, y1):
            return cells
        doubled = 2 * error
        if doubled > -dy:
            error -= dy
            x0 += sx
        if doubled < dx:
            error += dx
            y0 += sy


def _visible(layout: Layout, agent: Position, target: Position) -> bool:
    cells = _line_cells(agent, target)
    return all(cell in layout.floor_positions for cell in cells[1:-1])


def _local_view(layout: Layout, pose: Pose, tile_size: int = 42) -> Image.Image:
    x, y, direction = pose
    side = 7 * tile_size
    image = Image.new("RGB", (side, side), UNKNOWN_COLOR)
    draw = ImageDraw.Draw(image)
    forward_x, forward_y = DIRECTION_VECTORS[direction]
    right_x, right_y = -forward_y, forward_x
    for view_y in range(7):
        forward = 6 - view_y
        for view_x in range(7):
            right = view_x - 3
            world = (
                x + forward_x * forward + right_x * right,
                y + forward_y * forward + right_y * right,
            )
            if not (0 <= world[0] < layout.size and 0 <= world[1] < layout.size):
                continue
            if not _visible(layout, (x, y), world):
                continue
            color = layout.color_at(world)
            x0, y0 = view_x * tile_size, view_y * tile_size
            draw.rectangle(
                [x0, y0, x0 + tile_size - 1, y0 + tile_size - 1],
                fill=color,
                outline=GRID_COLOR,
            )
    agent_x = (3.5) * tile_size
    agent_y = (6.5) * tile_size
    _draw_agent(draw, agent_x, agent_y, 3, tile_size * 0.34)
    return image


def _room_for_pose(layout: Layout, pose: Pose) -> Optional[Room]:
    return layout.room_by_position.get((pose[0], pose[1]))


def _render_static_map(layout: Layout, poses: Sequence[Pose], output_path: Path) -> None:
    cell_size = 20
    map_image = _map_base(layout, cell_size, labels=True)
    _draw_route(map_image, poses, len(poses) - 1, cell_size)
    draw = ImageDraw.Draw(map_image)
    start_x, start_y = layout.start_position
    _draw_agent(
        draw,
        (start_x + 0.5) * cell_size,
        (start_y + 0.5) * cell_size,
        layout.start_direction,
        cell_size * 0.42,
    )

    canvas = Image.new("RGB", (900, 700), (18, 21, 25))
    canvas.paste(map_image, (28, 52))
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.text((28, 14), "Branching Rooms: exact logical map", font=_font(25, True), fill=TEXT_COLOR)
    panel_x = 680
    canvas_draw.text((panel_x, 62), "Map key", font=_font(19, True), fill=TEXT_COLOR)
    key_rows = [
        (AGENT_COLOR, "Start pose"),
        (ROUTE_COLOR, "Scripted route"),
        (DOOR_COLOR, "Doorway"),
        (DEAD_END_COLOR, "Dead-end outline"),
    ]
    y = 102
    for color, label in key_rows:
        canvas_draw.rectangle([panel_x, y + 2, panel_x + 18, y + 20], fill=color)
        canvas_draw.text((panel_x + 28, y), label, font=_font(14), fill=TEXT_COLOR)
        y += 34
    y += 12
    lines = [
        "31 x 31 grid",
        "693 reachable cells",
        "16 rooms",
        "5 dead ends",
        "3 actions:",
        "left / right / forward",
        "Start: (11, 11), east",
    ]
    for line in lines:
        canvas_draw.text((panel_x, y), line, font=_font(14), fill=MUTED_TEXT_COLOR)
        y += 24
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path))


def _render_animation_frame(
        layout: Layout,
        map_base: Image.Image,
        poses: Sequence[Pose],
        actions: Sequence[Optional[int]],
        index: int,
) -> Image.Image:
    width, height = 940, 570
    map_origin = (24, 58)
    cell_size = 16
    frame = Image.new("RGB", (width, height), (18, 21, 25))
    frame.paste(map_base, map_origin)
    _draw_route(frame, poses, index, cell_size, map_origin)
    draw = ImageDraw.Draw(frame)
    x, y, direction = poses[index]
    _draw_agent(
        draw,
        map_origin[0] + (x + 0.5) * cell_size,
        map_origin[1] + (y + 0.5) * cell_size,
        direction,
        cell_size * 0.45,
    )

    panel_x = 550
    draw.text((24, 17), "Branching Rooms scripted walkthrough", font=_font(24, True), fill=TEXT_COLOR)
    draw.text((panel_x, 58), "7x7 egocentric layout", font=_font(18, True), fill=TEXT_COLOR)
    draw.text((panel_x, 82), "Schematic of the agent's partial view", font=_font(12), fill=MUTED_TEXT_COLOR)
    view = _local_view(layout, poses[index], tile_size=48)
    frame.paste(view, (panel_x, 108))

    room = _room_for_pose(layout, poses[index])
    room_text = "corridor" if room is None else "room {},{}".format(*room)
    action = "Start" if actions[index] is None else ACTION_NAMES[actions[index]]
    reached_rooms = {
        _room_for_pose(layout, pose) for pose in poses[:index + 1]
    }
    reached_rooms.discard(None)
    reached_dead_ends = reached_rooms.intersection(layout.dead_end_rooms)
    unique_positions = {(pose[0], pose[1]) for pose in poses[:index + 1]}
    info_y = 462
    draw.text(
        (panel_x, info_y),
        "Step {:>3}/{:<3}   Action: {}".format(index, len(poses) - 1, action),
        font=_font(14, True),
        fill=TEXT_COLOR,
    )
    draw.text(
        (panel_x, info_y + 25),
        "Position: ({:>2}, {:>2})   Heading: {}   {}".format(x, y, DIRECTION_NAMES[direction], room_text),
        font=_font(13),
        fill=MUTED_TEXT_COLOR,
    )
    draw.text(
        (panel_x, info_y + 49),
        "Visited: {} cells   {}/16 rooms   {}/5 dead ends".format(
            len(unique_positions), len(reached_rooms), len(reached_dead_ends)
        ),
        font=_font(13),
        fill=MUTED_TEXT_COLOR,
    )
    return frame


def _write_video(
        layout: Layout,
        poses: Sequence[Pose],
        actions: Sequence[Optional[int]],
        output_path: Path,
        fps: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write the walkthrough video")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 940, 570
    command = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", "{}x{}".format(width, height),
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    map_base = _map_base(layout, 16, labels=False)
    try:
        for index in range(len(poses)):
            frame = _render_animation_frame(layout, map_base, poses, actions, index)
            process.stdin.write(frame.tobytes())
        # Hold the completed route briefly.
        final_frame = _render_animation_frame(layout, map_base, poses, actions, len(poses) - 1)
        for _ in range(fps * 2):
            process.stdin.write(final_frame.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("ffmpeg exited with status {}".format(return_code))


def _write_gif(video_path: Path, gif_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    filter_graph = (
        "fps=7,scale=760:-1:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
    )
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(video_path), "-filter_complex", filter_graph, str(gif_path)],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-source", type=Path, default=ENV_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--no-gif", action="store_true", help="Skip the smaller GIF export.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    layout = Layout(_load_class_literals(args.env_source))
    positions, room_walk = _scripted_positions(layout)
    poses, actions = _poses_and_actions(layout, positions)
    output_dir = args.output_dir
    map_path = output_dir / "branching_rooms_map.png"
    video_path = output_dir / "branching_rooms_walkthrough.mp4"
    gif_path = output_dir / "branching_rooms_walkthrough.gif"
    final_path = output_dir / "branching_rooms_walkthrough_final.png"

    _render_static_map(layout, poses, map_path)
    _write_video(layout, poses, actions, video_path, max(1, args.fps))
    if not args.no_gif:
        _write_gif(video_path, gif_path)
    map_base = _map_base(layout, 16, labels=False)
    final_frame = _render_animation_frame(layout, map_base, poses, actions, len(poses) - 1)
    final_frame.save(str(final_path))

    print("Map: {}".format(map_path))
    print("Video: {}".format(video_path))
    if not args.no_gif:
        print("GIF: {}".format(gif_path))
    print("Final frame: {}".format(final_path))
    print(
        "Traversal: {} actions, {} unique positions, {} rooms, {} dead ends".format(
            len(poses) - 1,
            len({(x, y) for x, y, _ in poses}),
            len(set(room_walk)),
            len(set(room_walk).intersection(layout.dead_end_rooms)),
        )
    )


if __name__ == "__main__":
    main()
