from __future__ import annotations

from collections import deque
from typing import Dict, Set, Tuple

import gymnasium as gym
from gymnasium.envs.registration import register, registry
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor
from minigrid.minigrid_env import MiniGridEnv


BRANCHING_ROOMS_ENV_ID = 'MiniGrid-BranchingRooms-v0'
BRANCHING_ROOMS_HELDOUT_ENV_ID = 'MiniGrid-BranchingRoomsHeldout-v0'

Room = Tuple[int, int]
Position = Tuple[int, int]
RoomEdge = Tuple[Room, Room]


class BranchingRoomsEnv(MiniGridEnv):
    """A fixed exploration maze with loops, bottlenecks, and dead-end room branches."""

    reward_range = (0.0, 0.0)
    size = 31
    room_count_per_axis = 4
    divider_coordinates = (8, 15, 22)
    room_bounds = ((1, 7), (9, 14), (16, 21), (23, 29))
    room_centers = ((4, 4), (11, 4), (18, 4), (26, 4),
                    (4, 11), (11, 11), (18, 11), (26, 11),
                    (4, 18), (11, 18), (18, 18), (26, 18),
                    (4, 26), (11, 26), (18, 26), (26, 26))
    start_room = (1, 1)
    start_position = (11, 11)
    start_direction = 0

    # The central two-by-two loop and right-hand loop permit multiple routes.
    # The five degree-one rooms force the agent to backtrack through familiar space.
    room_edges: Tuple[RoomEdge, ...] = (
        ((1, 1), (1, 2)),
        ((1, 2), (2, 2)),
        ((2, 2), (2, 1)),
        ((2, 1), (1, 1)),
        ((1, 1), (1, 0)),
        ((1, 0), (0, 0)),
        ((1, 1), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 2), (0, 3)),
        ((2, 1), (2, 0)),
        ((2, 0), (3, 0)),
        ((2, 1), (3, 1)),
        ((3, 1), (3, 2)),
        ((1, 2), (1, 3)),
        ((1, 3), (2, 3)),
        ((2, 3), (3, 3)),
        ((2, 2), (2, 3)),
    )

    # Doorways deliberately alternate within rooms so crossing a room is usually
    # required before entering the next branch.
    doorway_positions: Dict[RoomEdge, Position] = {
        ((1, 1), (1, 2)): (15, 13),
        ((1, 2), (2, 2)): (20, 15),
        ((2, 1), (2, 2)): (15, 17),
        ((1, 1), (2, 1)): (10, 15),
        ((1, 0), (1, 1)): (8, 10),
        ((0, 0), (1, 0)): (3, 8),
        ((0, 1), (1, 1)): (13, 8),
        ((0, 1), (0, 2)): (15, 2),
        ((0, 2), (0, 3)): (22, 6),
        ((2, 0), (2, 1)): (8, 20),
        ((2, 0), (3, 0)): (6, 22),
        ((2, 1), (3, 1)): (12, 22),
        ((3, 1), (3, 2)): (15, 27),
        ((1, 2), (1, 3)): (22, 10),
        ((1, 3), (2, 3)): (25, 15),
        ((2, 3), (3, 3)): (28, 22),
        ((2, 2), (2, 3)): (22, 20),
    }

    # A primary floor color plus a small secondary marker gives each room a
    # recognizable local appearance without changing its traversability.
    room_color_pairs = (
        ('red', 'blue'), ('red', 'green'), ('red', 'yellow'), ('red', 'purple'),
        ('green', 'blue'), ('green', 'yellow'), ('green', 'purple'), ('green', 'grey'),
        ('blue', 'yellow'), ('blue', 'purple'), ('blue', 'grey'), ('yellow', 'purple'),
        ('yellow', 'grey'), ('purple', 'grey'), ('grey', 'red'), ('blue', 'red'),
    )

    def __init__(
            self,
            max_steps: int = 1200,
            agent_view_size: int = 7,
            render_mode: str | None = None,
            **kwargs,
    ) -> None:
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            render_mode=render_mode,
            **kwargs,
        )
        self.reward_range = (0.0, 0.0)
        self.action_space = gym.spaces.Discrete(3)
        self._lifetime_visited_positions: Set[Position] = set()
        self._lifetime_visited_rooms: Set[Room] = set()
        self._episode_visited_positions: Set[Position] = set()
        self._episode_visited_poses: Set[Tuple[int, int, int]] = set()
        self._episode_visited_rooms: Set[Room] = set()
        self._reachable_positions: Set[Position] = set()

    @staticmethod
    def _gen_mission() -> str:
        return 'explore every reachable room'

    @staticmethod
    def _canonical_edge(room_a: Room, room_b: Room) -> RoomEdge:
        return tuple(sorted((room_a, room_b)))  # type: ignore[return-value]

    @classmethod
    def room_graph(cls) -> Dict[Room, Set[Room]]:
        graph = {
            (row, col): set()
            for row in range(cls.room_count_per_axis)
            for col in range(cls.room_count_per_axis)
        }
        for room_a, room_b in cls.room_edges:
            graph[room_a].add(room_b)
            graph[room_b].add(room_a)
        return graph

    @classmethod
    def dead_end_rooms(cls) -> Set[Room]:
        return {room for room, neighbors in cls.room_graph().items() if len(neighbors) == 1}

    def _paint_room_floors(self) -> None:
        for row, (y_min, y_max) in enumerate(self.room_bounds):
            for col, (x_min, x_max) in enumerate(self.room_bounds):
                room_index = row * self.room_count_per_axis + col
                primary, secondary = self.room_color_pairs[room_index]
                for x in range(x_min, x_max + 1):
                    for y in range(y_min, y_max + 1):
                        self.grid.set(x, y, Floor(primary))

                center_x, center_y = self.room_centers[room_index]
                for marker_x, marker_y in (
                        (center_x - 1, center_y - 1),
                        (center_x, center_y - 1),
                        (center_x - 1, center_y),
                        (center_x, center_y),
                ):
                    self.grid.set(marker_x, marker_y, Floor(secondary))

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self._paint_room_floors()

        self.grid.wall_rect(0, 0, width, height)
        for divider in self.divider_coordinates:
            self.grid.vert_wall(divider, 0, height)
            self.grid.horz_wall(0, divider, width)

        for edge, doorway in self.doorway_positions.items():
            if self._canonical_edge(*edge) not in {
                    self._canonical_edge(*configured_edge) for configured_edge in self.room_edges
            }:
                raise ValueError(f'Doorway {doorway} does not correspond to a configured room edge {edge}')
            self.grid.set(*doorway, Floor('grey'))

        self.agent_pos = self.start_position
        self.agent_dir = self.start_direction
        self.grid.set(*self.start_position, Floor(self.room_color_pairs[5][1]))

        self._reachable_positions = self._compute_reachable_positions(self.start_position)

    def _compute_reachable_positions(self, start: Position) -> Set[Position]:
        reachable = {start}
        frontier = deque([start])
        while frontier:
            x, y = frontier.popleft()
            for next_position in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                next_x, next_y = next_position
                if next_position in reachable or not (0 <= next_x < self.width and 0 <= next_y < self.height):
                    continue
                cell = self.grid.get(next_x, next_y)
                if cell is None or cell.can_overlap():
                    reachable.add(next_position)
                    frontier.append(next_position)
        return reachable

    @classmethod
    def room_for_position(cls, position: Position) -> Room | None:
        x, y = position
        for row, (y_min, y_max) in enumerate(cls.room_bounds):
            if not y_min <= y <= y_max:
                continue
            for col, (x_min, x_max) in enumerate(cls.room_bounds):
                if x_min <= x <= x_max:
                    return row, col
        return None

    def _record_visit(self) -> Dict[str, object]:
        position = tuple(int(value) for value in self.agent_pos)
        pose = (position[0], position[1], int(self.agent_dir))
        room = self.room_for_position(position)

        new_position = position not in self._episode_visited_positions
        new_lifetime_position = position not in self._lifetime_visited_positions
        new_room = room is not None and room not in self._episode_visited_rooms
        new_lifetime_room = room is not None and room not in self._lifetime_visited_rooms

        self._episode_visited_positions.add(position)
        self._episode_visited_poses.add(pose)
        self._lifetime_visited_positions.add(position)
        if room is not None:
            self._episode_visited_rooms.add(room)
            self._lifetime_visited_rooms.add(room)

        reachable_count = max(len(self._reachable_positions), 1)
        room_count = self.room_count_per_axis ** 2
        return {
            'agent_position': position,
            'agent_direction': int(self.agent_dir),
            'room_id': room,
            'new_position': new_position,
            'new_lifetime_position': new_lifetime_position,
            'new_room': new_room,
            'new_lifetime_room': new_lifetime_room,
            'visited_position_count': len(self._episode_visited_positions),
            'visited_pose_count': len(self._episode_visited_poses),
            'visited_room_count': len(self._episode_visited_rooms),
            'position_coverage': len(self._episode_visited_positions) / reachable_count,
            'pose_coverage': len(self._episode_visited_poses) / (4 * reachable_count),
            'room_coverage': len(self._episode_visited_rooms) / room_count,
            'lifetime_position_coverage': len(self._lifetime_visited_positions) / reachable_count,
            'lifetime_room_coverage': len(self._lifetime_visited_rooms) / room_count,
            'reachable_position_count': len(self._reachable_positions),
            'room_count': room_count,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self._episode_visited_positions = set()
        self._episode_visited_poses = set()
        self._episode_visited_rooms = set()
        info.update(self._record_visit())
        return observation, info

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        info.update(self._record_visit())
        return observation, 0.0, terminated, truncated, info


class BranchingRoomsHeldoutEnv(BranchingRoomsEnv):
    """A topology held out from the fixed Branching Rooms training layout."""

    room_edges: Tuple[RoomEdge, ...] = (
        # Two loops around the starting room use different connections from the
        # training layout. The remaining rooms form five dead-end branches.
        ((1, 1), (1, 0)),
        ((1, 0), (2, 0)),
        ((2, 0), (2, 1)),
        ((2, 1), (1, 1)),
        ((1, 1), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 2), (1, 2)),
        ((1, 2), (1, 1)),
        ((0, 1), (0, 0)),
        ((0, 2), (0, 3)),
        ((0, 3), (1, 3)),
        ((1, 2), (2, 2)),
        ((2, 2), (2, 3)),
        ((2, 3), (3, 3)),
        ((2, 2), (3, 2)),
        ((2, 1), (3, 1)),
        ((3, 1), (3, 0)),
    )

    doorway_positions: Dict[RoomEdge, Position] = {
        ((1, 0), (1, 1)): (8, 12),
        ((1, 0), (2, 0)): (5, 15),
        ((2, 0), (2, 1)): (8, 19),
        ((1, 1), (2, 1)): (13, 15),
        ((0, 1), (1, 1)): (10, 8),
        ((0, 1), (0, 2)): (15, 5),
        ((0, 2), (1, 2)): (19, 8),
        ((1, 1), (1, 2)): (15, 10),
        ((0, 0), (0, 1)): (8, 3),
        ((0, 2), (0, 3)): (22, 4),
        ((0, 3), (1, 3)): (27, 8),
        ((1, 2), (2, 2)): (17, 15),
        ((2, 2), (2, 3)): (22, 18),
        ((2, 3), (3, 3)): (24, 22),
        ((2, 2), (3, 2)): (20, 22),
        ((2, 1), (3, 1)): (10, 22),
        ((3, 0), (3, 1)): (8, 26),
    }


if BRANCHING_ROOMS_ENV_ID not in registry:
    register(
        id=BRANCHING_ROOMS_ENV_ID,
        entry_point='zoo.minigrid.envs.branching_rooms_env:BranchingRoomsEnv',
    )

if BRANCHING_ROOMS_HELDOUT_ENV_ID not in registry:
    register(
        id=BRANCHING_ROOMS_HELDOUT_ENV_ID,
        entry_point='zoo.minigrid.envs.branching_rooms_env:BranchingRoomsHeldoutEnv',
    )
