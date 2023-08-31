import concurrent.futures as mp
import PIL.ImageQt as ImageQT

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from threading import Thread
from typing import *
from typing_extensions import TypedDict
import numpy as np
import os

openslide_path = os.path.abspath("../../openslide/bin")
os.add_dll_directory(openslide_path)
from openslide import OpenSlide


class slide_view(QGraphicsView):
    sendPixmap = pyqtSignal(QGraphicsPixmapItem)

    def __init__(self, *args):
        super(slide_view, self).__init__(*args)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setMouseTracking(True)

        self.slide: OpenSlide = None
        self.filepath = None
        self.width = self.scene().width()
        self.height = self.scene().height()
        self.panning: bool = False
        self.pan_start: QPointF = QPointF()
        self.cur_scaling_factor: float = 0.0
        self.max_scaling_factor: float = 0.0
        self.relative_scaling_factor: float = 0.0
        self.cur_downsample: int = 0
        self.cur_level = 0
        self.dim_count = 0
        self.down_sample_factors = {}
        self.dimensions = {}
        self.threads = []
        self.image_blocks = {}
        self.fused_image = QPixmap()
        self.painter = QPainter(self.fused_image)
        self.pixmap_item = QGraphicsPixmapItem()
        self.patch_dims = [0, 0]
        self.pos = QPointF(0, 0)
        self.level_zoom = 1
        self.mouse_pos = QPointF(0, 0)
        self.grid_points = {}

    def fitInView(self, rect, aspect_ratio_mode=Qt.AspectRatioMode.KeepAspectRatio):
        if not self.filepath:
            RuntimeError("There was no slide set!")
        super(slide_view, self).fitInView(rect, aspect_ratio_mode)

    def load_slide(self, filepath: str, width: int = None, height: int = None):
        """
        Loads a new _slide. Needs an update_size after loading a new image.
        :param filepath: path of the _slide data. The data type is based on the OpenSlide library and can handle:
                         Aperio (.svs, .tif), Hamamatsu (.vms, .vmu, .ndpi), Leica (.scn), MIRAX (.mrxs),
                         Philips (.tiff), Sakura (.svslide), Trestle (.tif), Ventana (.bif, .tif),
                         Generic tiled TIFF (.tif) (see https://openslide.org)
        :type filepath: str
        :param width: width of the GraphicsView
        :type width: int
        :param height: height of the GraphicView
        :type height: int
        :return: /
        """
        self.slide = OpenSlide(filepath)
        self.filepath = filepath
        if not width or not height:
            self.width = self.scene().views()[0].viewport().width()
            self.height = self.scene().views()[0].viewport().height()

        rect = QRectF(QPointF(0, 0), QSizeF(self.width, self.height))

        self.fused_image = QPixmap(self.width * 4, self.height * 4)
        self.pixmap_item.setPixmap(self.fused_image)
        self.pixmap_item.setOffset(-self.width, -self.height)

        self.image_blocks = {i: QPixmap(self.width, self.height) for i in range(16)}
        self.image_blocks = np.array(list(self.image_blocks.values()))
        self.image_blocks = self.image_blocks.reshape([4, 4])

        self.dimensions = np.array(self.slide.level_dimensions)
        self.dim_count = self.slide.level_count
        self.patch_dims = [self.dimensions[self.dim_count - 1][0], self.dimensions[self.dim_count - 1][1]]

        self.down_sample_factors = [self.slide.level_downsamples[level] for level in range(self.slide.level_count)]

        self.fitInView(rect)

        self.cur_scaling_factor = max(self.dimensions[0][0] / self.width, self.dimensions[0][1] / self.height)
        self.max_scaling_factor = self.cur_scaling_factor
        self.cur_level = self.slide.get_best_level_for_downsample(self.cur_scaling_factor)
        self.relative_scaling_factor = self.cur_scaling_factor / self.down_sample_factors[self.cur_level]
        self.pos = self.pixmap_item.pos()

        self.cur_downsample = self.slide.level_downsamples[self.cur_level]
        self.grid_points = {0: 0, 1: 0, 2: self.width * self.cur_downsample, 3: self.height * self.cur_downsample}

        self.check_for_update(QPointF(0, 0), True)
        self.sendPixmap.emit(self.pixmap_item)

    def check_for_update(self, move, zoomed):
        self.width = self.scene().views()[0].viewport().width()
        self.height = self.scene().views()[0].viewport().height()

        max_threads = 16
        sqrt_thread_count = int(np.sqrt(max_threads))

        new_patches = [False for r in range(max_threads)]

        self.cur_downsample = self.slide.level_downsamples[self.cur_level]

        print(self.mouse_pos)
        grid_width = self.grid_points[2] - self.grid_points[0]
        grid_height = self.grid_points[3] - self.grid_points[1]

        while self.mouse_pos.x() > self.grid_points[2]:
            self.pixmap_item.moveBy(self.width * self.level_zoom, 0)
            self.grid_points[0] += grid_width
            self.grid_points[2] += grid_width
            new_patches[3] = True
            new_patches[7] = True
            new_patches[11] = True
            new_patches[15] = True

            self.image_blocks = np.roll(self.image_blocks, -1, axis=0)
        while self.mouse_pos.x() < self.grid_points[0]:
            self.pixmap_item.moveBy(-self.width * self.level_zoom, 0)
            self.grid_points[0] -= grid_width
            self.grid_points[2] -= grid_width
            new_patches[0] = True
            new_patches[4] = True
            new_patches[8] = True
            new_patches[12] = True

            self.image_blocks = np.roll(self.image_blocks, 1, axis=0)
        while self.mouse_pos.y() > self.grid_points[3]:
            self.pixmap_item.moveBy(0, self.height * self.level_zoom)
            self.grid_points[1] += grid_height
            self.grid_points[3] += grid_height
            new_patches[12] = True
            new_patches[13] = True
            new_patches[14] = True
            new_patches[15] = True

            self.image_blocks = np.roll(self.image_blocks, -1, axis=1)
        while self.mouse_pos.y() < self.grid_points[1]:
            self.pixmap_item.moveBy(0, -self.height * self.level_zoom)
            self.grid_points[1] -= grid_height
            self.grid_points[3] -= grid_height
            new_patches[0] = True
            new_patches[1] = True
            new_patches[2] = True
            new_patches[3] = True

            self.image_blocks = np.roll(self.image_blocks, 1, axis=1)

        if zoomed:
            new_patches = [True for r in range(16)]

        if True: #any(new_patches):
            block_width = int(self.width)
            block_height = int(self.height)
            block_offset_width = int(self.width * self.down_sample_factors[self.cur_level])
            block_offset_height = int(self.height * self.down_sample_factors[self.cur_level])

            image_offset_width = int(self.width * self.down_sample_factors[self.cur_level])
            image_offset_height = int(self.height * self.down_sample_factors[self.cur_level])

            offset_mouse_pos = QPointF(self.grid_points[0], self.grid_points[1]) - \
                               QPointF(image_offset_width, image_offset_height)

            self.fused_image = QPixmap(4 * self.width, 4 * self.height)
            self.painter = QPainter(self.fused_image)

            with mp.ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(self.process_image_block, i, offset_mouse_pos, block_width,
                                           block_height, block_offset_width, block_offset_height, sqrt_thread_count,
                                           self.cur_level, new_patches[i])
                           for i in range(max_threads)]

                mp.wait(futures)

                pen = QPen()
                pen.setWidth(40)
                pen.setColor(QColor('red'))
                self.painter.setPen(pen)
                self.painter.drawPoint(QPointF(self.mouse_pos.x()/self.cur_scaling_factor + self.width + 50,
                                               self.mouse_pos.y()/self.cur_scaling_factor + self.height + 50))

                self.painter.end()

            self.pixmap_item.setPixmap(self.fused_image)
            self.level_zoom = self.down_sample_factors[self.cur_level] / self.cur_scaling_factor
            self.pixmap_item.setScale(self.level_zoom)

    def process_image_block(self, block_index, mouse_pos, block_width, block_height, block_offset_width,
                            block_offset_height, sqrt_threads, level, generate_new):
        idx_width = block_index % sqrt_threads
        idx_height = block_index // sqrt_threads

        block_location = (
            idx_width * block_offset_width,
            idx_height * block_offset_height
        )
        if generate_new:
            image = self.slide.read_region(
                (int(mouse_pos.x() + block_location[0]), int(mouse_pos.y() + block_location[1])),
                level,
                (block_width, block_height)
            )

            self.image_blocks[idx_width, idx_height] = QPixmap.fromImage(ImageQT.ImageQt(image))

        self.painter.drawPixmap(idx_width * block_width,
                                idx_height * block_height,
                                block_width, block_height, self.image_blocks[idx_width, idx_height])

        # print(f"block_id: {idx_width}, {idx_height} block_Coordinates: {int(mouse_pos.x() + block_location[0])}, "
        #       f"{int(mouse_pos.y() + block_location[1])}, {int(mouse_pos.x() + block_location[0]) + block_width * self.down_sample_factors[level]},"
        #       f" {int(mouse_pos.y() + block_location[1]) + block_height * self.down_sample_factors[level]}")

    def update_grid(self, new_level) -> bool:
        if new_level != self.cur_level:
            zoom_adjustment = QPointF(self.width * self.down_sample_factors[self.cur_level] / 2,
                                      self.height * self.down_sample_factors[self.cur_level] / 2)
            old_grid_width = (self.grid_points[2]-self.grid_points[0])
            old_grid_height = (self.grid_points[3]-self.grid_points[1])
            self.pixmap_item.setPos(QPointF(0, 0))
            self.level_zoom = 0.5

            if new_level < self.cur_level:
                self.mouse_pos -= zoom_adjustment
                grid_width = old_grid_width/2
                grid_height = old_grid_height/2
                self.grid_points[2] += grid_width
                self.grid_points[3] += grid_height
                if self.grid_points[2] - self.mouse_pos.x() < 0:
                    self.grid_points[0] += grid_width,
                    self.grid_points[2] += grid_width
                if self.grid_points[3] - self.mouse_pos.y() < 0:
                    self.grid_points[1] += grid_height
                    self.grid_points[3] += grid_height
            elif new_level > self.cur_level:
                self.mouse_pos += zoom_adjustment
                grid_width = old_grid_width * 2
                grid_height = old_grid_height * 2
                self.grid_points[2] += grid_width
                self.grid_points[3] += grid_height
                # if int(self.grid_points[0]) % int(grid_width) > 1:
                #     self.grid_points[0] -= old_grid_width
                #     self.grid_points[2] -= old_grid_width
                # if int(self.grid_points[3]) % int(grid_height) > 1:
                #     self.grid_points[1] -= old_grid_height
                #     self.grid_points[3] -= old_grid_height
            self.cur_level = new_level
            return True
        else:
            return False

    def wheelEvent(self, event: QWheelEvent):
        """
        Scales the image and moves into the mouse position
        :param event: event to initialize the function
        :type event: QWheelEvent
        :return: /
        """
        scale_factor = 1.1 if event.angleDelta().y() <= 0 else 1 / 1.1
        new_scaling_factor = min(max(self.cur_scaling_factor * scale_factor, 1), self.max_scaling_factor)

        if new_scaling_factor == self.cur_scaling_factor:
            return
        else:
            self.cur_scaling_factor = new_scaling_factor
            level = self.slide.get_best_level_for_downsample(self.cur_scaling_factor)

        scale_jump = self.update_grid(level)

        mouse_zoom_adjust = QPointF(self.width * (self.cur_scaling_factor/scale_factor - self.cur_scaling_factor)/2,
                                    self.height * (self.cur_scaling_factor/scale_factor - self.cur_scaling_factor)/2)
        print(mouse_zoom_adjust)
        self.mouse_pos += mouse_zoom_adjust
        self.relative_scaling_factor = self.cur_scaling_factor / self.down_sample_factors[self.cur_level]
        self.pixmap_item.setScale(self.down_sample_factors[self.cur_level] / self.cur_scaling_factor)
        old_level_zoom = self.level_zoom
        self.level_zoom = self.down_sample_factors[self.cur_level] / self.cur_scaling_factor
        self.pixmap_item.moveBy(self.width * (old_level_zoom - self.level_zoom),
                                self.height * (old_level_zoom - self.level_zoom))

        # new_pos = self.mapToScene(event.position().toPoint())
        # move = QPointF(self.width / 2, self.height / 2) - new_pos
        #
        # self.pixmap_item.moveBy(move.x() * self.scale,
        #                         move.y() * self.scale)

        # self.pos = self.pixmap_item.pos() / self.scale
        #
        # new_pos = QPointF((new_pos.x() / self.width - 0.5) *
        #                   self.dimensions[self.dim_count - self.cur_level - 1][0] * self.relative_scaling_factor,
        #                   (new_pos.y() / self.height - 0.5) *
        #                   self.dimensions[self.dim_count - self.cur_level - 1][1] * self.relative_scaling_factor)

        # self.mouse_pos += QPointF(self.width / 2 * (old_scaling_factor - self.cur_scaling_factor),
        #                           self.height / 2 * (old_scaling_factor - self.cur_scaling_factor))
        # self.mouse_pos += new_pos

        self.check_for_update(QPointF(0, 0), scale_jump)

    def mousePressEvent(self, event: QMouseEvent):
        """
        Enables panning of the image
        :param event: event to initialize the function
        :type event: QMouseEvent
        :return: /
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.panning = True
            self.pan_start = self.mapToScene(event.pos())
        super(QGraphicsView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        Disables panning of the image
        :param event: event to initialize the function
        :type event: QMouseEvent
        :return: /
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.panning = False
        super(QGraphicsView, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Realizes panning, if activated
        :param event: event to initialize the function
        :type event: QMouseEvent
        :return: /
        """
        if self.panning:
            new_pos = self.mapToScene(event.pos())
            move = self.pan_start - new_pos
            self.pixmap_item.moveBy(-move.x(), -move.y())
            self.pan_start = new_pos
            self.pos = self.pixmap_item.pos() / self.level_zoom

            move = QPointF(move.x() * self.cur_scaling_factor,
                           move.y() * self.cur_scaling_factor)
            self.mouse_pos += move
            self.check_for_update(move, False)
        super(QGraphicsView, self).mouseMoveEvent(event)
