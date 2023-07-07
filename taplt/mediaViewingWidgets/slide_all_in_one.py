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
        self.scale = 1
        self.mouse_pos = QPointF(0, 0)

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

        self.check_for_update(QPointF(0, 0), True)
        self.sendPixmap.emit(self.pixmap_item)

    def check_for_update(self, move, zoomed):
        self.width = self.scene().views()[0].viewport().width()
        self.height = self.scene().views()[0].viewport().height()

        self.mouse_pos = QPointF(np.clip(self.mouse_pos.x(), 0, self.dimensions[0][0]),
                                 np.clip(self.mouse_pos.y(), 0, self.dimensions[0][1]))

        # TODO: Problem -> The Mouse pos is used to calculate the next patch. However, this leads to slight
        # inconsistencies in the produced image. A static grid approach would probably be the best solution
        print(self.mouse_pos)

        max_threads = 16
        sqrt_thread_count = int(np.sqrt(max_threads))

        new_patches = [False for r in range(max_threads)]

        if self.pos.x() < - self.width:
            self.pixmap_item.moveBy(self.width * self.scale, 0)
            new_patches[3] = True
            new_patches[7] = True
            new_patches[11] = True
            new_patches[15] = True

            self.image_blocks = np.roll(self.image_blocks, -1, axis=0)
        if self.pos.x() >= self.width:
            self.pixmap_item.moveBy(-self.width * self.scale, 0)
            new_patches[0] = True
            new_patches[4] = True
            new_patches[8] = True
            new_patches[12] = True

            self.image_blocks = np.roll(self.image_blocks, 1, axis=0)
        if self.pos.y() < -self.height:
            self.pixmap_item.moveBy(0, self.height * self.scale)
            new_patches[12] = True
            new_patches[13] = True
            new_patches[14] = True
            new_patches[15] = True

            self.image_blocks = np.roll(self.image_blocks, -1, axis=1)
        if self.pos.y() >= self.height:
            self.pixmap_item.moveBy(0, -self.height * self.scale)
            new_patches[0] = True
            new_patches[1] = True
            new_patches[2] = True
            new_patches[3] = True

            self.image_blocks = np.roll(self.image_blocks, 1, axis=1)

        if zoomed:
            new_patches = [True for r in range(16)]

        if any(new_patches):

            block_width = int(self.width)
            block_height = int(self.height)
            block_offset_width = int(self.width * self.down_sample_factors[self.cur_level])
            block_offset_height = int(self.height * self.down_sample_factors[self.cur_level])

            image_offset_width = int(self.width * self.down_sample_factors[self.cur_level])
            image_offset_height = int(self.height * self.down_sample_factors[self.cur_level])

            offset_mouse_pos = self.mouse_pos - QPointF(image_offset_width, image_offset_height)

            self.fused_image = QPixmap(4 * self.width, 4 * self.height)
            self.painter = QPainter(self.fused_image)

            with mp.ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(self.process_image_block, i, offset_mouse_pos, block_width,
                                           block_height, block_offset_width, block_offset_height, sqrt_thread_count,
                                           self.cur_level, new_patches[i])
                           for i in range(max_threads)]

                mp.wait(futures)

                self.painter.end()

            self.pixmap_item.setPixmap(self.fused_image)
            self.scale = self.down_sample_factors[self.cur_level] / self.cur_scaling_factor
            self.pixmap_item.setScale(self.scale)

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



    def wheelEvent(self, event: QWheelEvent):
        """
        Scales the image and moves into the mouse position
        :param event: event to initialize the function
        :type event: QWheelEvent
        :return: /
        """
        old_scaling_factor = self.cur_scaling_factor
        old_scale = self.scale

        scale_factor = 1.1 if event.angleDelta().y() <= 0 else 1 / 1.1
        new_scaling_factor = min(max(self.cur_scaling_factor * scale_factor, 1), self.max_scaling_factor)

        if new_scaling_factor == self.cur_scaling_factor:
            return

        self.cur_scaling_factor = new_scaling_factor
        scale_jump = False
        level = self.slide.get_best_level_for_downsample(self.cur_scaling_factor)
        if level < self.cur_level:
            scale_jump = True
            self.mouse_pos += QPointF(self.width * self.down_sample_factors[self.cur_level]/2,
                                      self.height * self.down_sample_factors[self.cur_level]/2)
        elif level > self.cur_level:
            scale_jump = True
            self.mouse_pos -= QPointF(self.width * self.down_sample_factors[level] / 2,
                                      self.height * self.down_sample_factors[level] / 2)
        self.cur_level = level
        self.relative_scaling_factor = self.cur_scaling_factor/self.down_sample_factors[self.cur_level]
        self.pixmap_item.setScale(self.down_sample_factors[self.cur_level]/self.cur_scaling_factor)
        self.scale = self.down_sample_factors[self.cur_level]/self.cur_scaling_factor
        self.pixmap_item.moveBy(self.width * (old_scale - self.scale),
                                self.height * (old_scale - self.scale))

        new_pos = self.mapToScene(event.position().toPoint())
        move = QPointF(self.width/2, self.height/2) - new_pos

        self.pixmap_item.moveBy(move.x() * self.scale,
                                move.y() * self.scale)

        self.pos = self.pixmap_item.pos() / self.scale

        new_pos = QPointF((new_pos.x() / self.width - 0.5) *
                          self.dimensions[self.dim_count - self.cur_level - 1][0] * self.relative_scaling_factor,
                          (new_pos.y() / self.height - 0.5) *
                          self.dimensions[self.dim_count - self.cur_level - 1][1] * self.relative_scaling_factor)

        # self.mouse_pos += QPointF(self.width / 2 * (old_scaling_factor - self.cur_scaling_factor),
        #                           self.height / 2 * (old_scaling_factor - self.cur_scaling_factor))
        self.mouse_pos += new_pos

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
            self.pos = self.pixmap_item.pos() / self.scale

            move = QPointF(move.x() * self.cur_scaling_factor,
                           move.y() * self.cur_scaling_factor)
            self.mouse_pos += move
            self.check_for_update(move, False)
        super(QGraphicsView, self).mouseMoveEvent(event)
