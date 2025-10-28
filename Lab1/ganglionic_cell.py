from __future__ import annotations
import os
from typing import Dict, Literal, Tuple, List, Union, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils

# Константы ориентации
ORIENTATION_VERTICAL = 'vertical'
ORIENTATION_HORIZONTAL = 'horizontal'

CellTypes = List[Literal["ganglion", "simple", "complex"]]


class ReceptiveField:
    """Базовый класс рецептивного поля с позицией и размером."""
    def __init__(self, position: Tuple[int, int], size: Tuple[int, int]) -> None:
        self.position: np.ndarray = np.array(position, dtype=np.int16)
        self.size: np.ndarray = np.array(size, dtype=np.int16)

    def get_response(self, image: np.ndarray) -> float:
        """Базовый класс возвращает среднее значение изображения."""
        return float(np.mean(image))

    def set_position(self, new_position: Union[Tuple[int, int], np.ndarray]) -> None:
        self.position = np.array(new_position, dtype=np.int16)


class GanglionCell(ReceptiveField):
    def __init__(self, position: Tuple[int, int] = (0, 0),
                 inner_radius: int = 5,
                 outer_radius: int = 11,
                 is_off_center: bool = False) -> None:
        super().__init__(position, (1, 1))
        self.inner_radius: int = inner_radius
        self.outer_radius: int = outer_radius
        self.is_off_center: bool = is_off_center

    def get_response(self, image: np.ndarray) -> float:
        blurred_inner = cv2.GaussianBlur(image, (self.inner_radius | 1, self.inner_radius | 1), sigmaX=0)
        blurred_outer = cv2.GaussianBlur(image, (self.outer_radius | 1, self.outer_radius | 1), sigmaX=0)
        diff = blurred_outer - blurred_inner if self.is_off_center else blurred_inner - blurred_outer
        x, y = self.position
        window = diff[y-2:y+3, x-2:x+3]
        return float(np.mean(window))



class SimpleCell(ReceptiveField):
    def __init__(self, position: Tuple[int, int] = (0, 0),
                 ganglion_layout: Tuple[int, int] = (5, 5),
                 span: int = 3,
                 is_off_type: bool = False,
                 orientation: str = ORIENTATION_VERTICAL) -> None:
        super().__init__(position, ganglion_layout)
        self.span = span
        self.orientation = orientation

        # корректируем размер сетки на нечётное число
        self.size[0] += int(not(self.size[0] & 1))
        self.size[1] += int(not(self.size[1] & 1))

        self.ganglion_cells: List[GanglionCell] = []
        half_x, half_y = self.size[0] // 2, self.size[1] // 2
        for dx in range(-half_x, half_x + 1):
            for dy in range(-half_y, half_y + 1):
                if is_off_type:
                    cell_type = dx != 0 if self.orientation == ORIENTATION_VERTICAL else dy != 0
                else:
                    cell_type = dx == 0 if self.orientation == ORIENTATION_VERTICAL else dy == 0
                cell_pos = (self.position[0] + dx * self.span, self.position[1] + dy * self.span)
                self.ganglion_cells.append(GanglionCell(position=cell_pos, is_off_center=not cell_type))

    def get_response(self, image: np.ndarray) -> float:
        return sum(cell.get_response(image) for cell in self.ganglion_cells)

    def set_position(self, new_position: Union[Tuple[int, int], np.ndarray]) -> None:
        new_pos_arr = np.array(new_position, dtype=np.int16)
        delta = new_pos_arr - self.position
        self.position = new_pos_arr
        for cell in self.ganglion_cells:
            cell.position += delta

class ComplexCell(ReceptiveField):
    def __init__(self, position: Tuple[int, int] = (0, 0),
                 simple_layout: Tuple[int, int] = (5, 5),  # решётка 5x5
                 span: int = 10,  # шаг между центрами решётки
                 ganglion_layout: Tuple[int, int] = (3, 3),
                 ganglion_span: int = 3,
                 is_off_type: bool = False,
                 orientation: str = ORIENTATION_VERTICAL) -> None:
        """
        ComplexCell теперь представляет собой решётку из простых клеток.
        Центры располагаются в узлах сетки (NxN) вокруг центральной точки.
        """
        super().__init__(position, simple_layout)
        self.span = span
        self.orientation = orientation

        # Гарантируем нечётный размер решётки
        self.size[0] += int(not (self.size[0] & 1))
        self.size[1] += int(not (self.size[1] & 1))

        self.simple_cells: List[SimpleCell] = []
        half_x, half_y = self.size[0] // 2, self.size[1] // 2

        # Создаём сетку NxN
        for dx in range(-half_x, half_x + 1):
            for dy in range(-half_y, half_y + 1):
                pos = (self.position[0] + dx * self.span,
                       self.position[1] + dy * self.span)

                # Ориентации чередуются для разнообразия
                local_orientation = (
                    self.orientation
                    if (dx + dy) % 2 == 0
                    else (ORIENTATION_HORIZONTAL if self.orientation == ORIENTATION_VERTICAL
                          else ORIENTATION_VERTICAL)
                )

                self.simple_cells.append(SimpleCell(
                    position=pos,
                    ganglion_layout=ganglion_layout,
                    span=ganglion_span,
                    is_off_type=is_off_type,
                    orientation=local_orientation
                ))

    def get_response(self, image: np.ndarray) -> float:
        # Берём максимум откликов, как у классической complex-клетки
        if not self.simple_cells:
            return 0.0
        return max(abs(cell.get_response(image)) for cell in self.simple_cells)

    def set_position(self, new_position: Union[Tuple[int, int], np.ndarray]) -> None:
        new_pos_arr = np.array(new_position, dtype=np.int16)
        delta = new_pos_arr - self.position
        self.position = new_pos_arr
        for cell in self.simple_cells:
            cell.set_position(tuple(np.array(cell.position) + delta))

class ReceptiveFieldAnalyzer:
    def __init__(self,
                 ganglion_cell: GanglionCell,
                 simple_cell: SimpleCell,
                 complex_cell: ComplexCell,
                 image_size: Tuple[int, int] = (256, 256)) -> None:
        self.image_size = np.array(image_size, dtype=np.int16)
        self.cells: Dict[str, ReceptiveField] = {
            "ganglion": ganglion_cell,
            "simple": simple_cell,
            "complex": complex_cell
        }
        center = tuple(self.image_size // 2)
        for cell in self.cells.values():
            cell.set_position(center)

    def _prepare_image(self) -> np.ndarray:
        return np.zeros(self.image_size, dtype=np.int16)

    def check_point_response(self, cell_type: str, cell: ReceptiveField, field_dims: Tuple[int, int]) -> np.ndarray:
        field_dims_arr = np.array(field_dims, dtype=np.int16)
        if np.any(self.image_size <= field_dims_arr):
            self.image_size = field_dims_arr + 50
        cell.set_position(tuple(self.image_size // 2))

        circle_interval = 10
        result_size = self.image_size if np.any(self.image_size < field_dims_arr * circle_interval) \
            else field_dims_arr * circle_interval + 50
        response_map = np.zeros(result_size, dtype=np.int16)
        base_img = self._prepare_image()

        for x in tqdm(range(field_dims_arr[0]), desc=f"  Прогресс для [{cell_type}]", unit="px"):
            for y in range(field_dims_arr[1]):
                coords = (self.image_size[1] // 2 + x - field_dims_arr[0] // 2,
                          self.image_size[0] // 2 + y - field_dims_arr[1] // 2)
                cv2.circle(base_img, center=coords, radius=1, color=255, thickness=-1)
                resp = cell.get_response(base_img)
                cv2.circle(base_img, center=coords, radius=1, color=0, thickness=-1)
                display_coords = (result_size[1] // 2 - (field_dims_arr[0] // 2 - x) * circle_interval,
                                  result_size[0] // 2 - (field_dims_arr[1] // 2 - y) * circle_interval)
                cv2.circle(response_map, center=display_coords, radius=2, color=int(resp), thickness=-1)
        return response_map

    def rotate_line_response(self, cell: ReceptiveField, length: int = 100, step_angle: int = 5):
        base_img = self._prepare_image()
        h, w = base_img.shape
        responses = []
        angles = list(range(0, 360, step_angle))

        for angle_deg in angles:
            img = base_img.copy()
            theta = np.deg2rad(angle_deg)
            dx = round(length * np.cos(theta))
            dy = round(length * np.sin(theta))
            pt1 = (int(w // 2 + dx), int(h // 2 + dy))
            pt2 = (int(w // 2 - dx), int(h // 2 - dy))

            # ограничим координаты, чтобы не выходили за изображение
            pt1 = (np.clip(pt1[0], 0, w - 1), np.clip(pt1[1], 0, h - 1))
            pt2 = (np.clip(pt2[0], 0, w - 1), np.clip(pt2[1], 0, h - 1))

            cv2.line(img, pt1, pt2, color=255, thickness=1)
            responses.append(cell.get_response(img))

        return responses, angles



    def shift_line_response(self, cell: ReceptiveField, max_shift: int = 10):
        h, w = self.image_size
        shifts = list(range(-max_shift, max_shift))
        responses = []
        
        for shift in shifts:
            img = self._prepare_image()
            x = w // 2 + shift
            cv2.line(img, (x, 0), (x, h), color=255, thickness=1)
            responses.append(cell.get_response(img))
        
        return responses, shifts

    def check_circle_response(self, cell: ReceptiveField, max_radius: int = 30):
        img = self._prepare_image()
        h, w = img.shape
        responses = []
        for radius in range(max_radius):
            cv2.circle(img, (w // 2, h // 2), radius, color=255, thickness=-1)
            responses.append(cell.get_response(img))
            cv2.circle(img, (w // 2, h // 2), radius, color=0, thickness=-1)
        return responses

    def run_all(self,
                point_field: Tuple[int, int] = (13, 13),
                length: int = 100,
                step_angle: int = 10,
                max_shift: int = 10,
                max_radius: int = 30,
                cell_types: Optional[CellTypes] = None) -> None:
        output_folder = 'aoutput'
        os.makedirs(output_folder, exist_ok=True)

        for cell_type, cell in self.cells.items():
            if cell_types is None or cell_type in cell_types:
                map_img = self.check_point_response(cell_type, cell, point_field)
                plt.title(cell_type)
                plt.imshow(map_img)
                plt.savefig(f"{output_folder}/{cell_type}_point.png")
                plt.show()

                norm_img = cv2.normalize(map_img, None, 75, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                norm_img[norm_img == norm_img[0, 0]] = 0
                norm_img = utils.stretch_image(norm_img, anchors=(0, 255), min_gap=10)
                cv2.imwrite(f"{output_folder}/{cell_type}_point_norm.png", norm_img)

                res, angles = self.rotate_line_response(cell, length, step_angle)
                plt.title(f"{cell_type} rotate")
                plt.xticks(np.arange(0, 361, 30))
                plt.plot(angles, res)
                plt.savefig(f"{output_folder}/{cell_type}_rotate.png")
                plt.show()

                res_s, shifts = self.shift_line_response(cell, max_shift)
                plt.title(f"{cell_type} shift")
                plt.plot(shifts, res_s)
                plt.savefig(f"{output_folder}/{cell_type}_shift.png")
                plt.show()

                circle_res = self.check_circle_response(cell, max_radius)
                plt.title(f"{cell_type} circle")
                plt.plot(list(range(max_radius)), circle_res)
                plt.savefig(f"{output_folder}/{cell_type}_circle.png")
                plt.show()


if __name__ == '__main__':
    analyzer = ReceptiveFieldAnalyzer(
        ganglion_cell=GanglionCell(),
        simple_cell=SimpleCell(
            ganglion_layout=(5, 5),
            span=3,
            is_off_type=False,
            orientation=ORIENTATION_VERTICAL
        ),
        complex_cell=ComplexCell(
            simple_layout=(5, 5),     # решётка NxN
            span=10,                  # шаг между центрами решётки
            ganglion_layout=(3, 3),   # размер сетки внутри простых клеток
            ganglion_span=3,
            is_off_type=False,
            orientation=ORIENTATION_VERTICAL
        )
    )

    types: CellTypes = ['ganglion', 'simple', 'complex']
    analyzer.run_all(
        point_field=(13, 13),
        length=100,
        step_angle=10,
        max_shift=10,
        max_radius=30,
        cell_types=types
    )

