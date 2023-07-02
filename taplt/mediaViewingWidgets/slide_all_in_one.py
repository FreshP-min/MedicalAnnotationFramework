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
        self.mouse_pos: QPointF = QPointF()
        self.down_sample_factors = {}
        self.dimensions = {}
        self.threads = []
        self.image_blocks = {}
        self.fused_image = QPixmap()
        self.painter = QPainter(self.fused_image)
        self.pixmap_item = QGraphicsPixmapItem()
        self.patch_dims = [0, 0]
        self.scene_mouse_pos: QPointF = QPointF()

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
        self.mouse_pos = QPointF(0, 0)
        self.scene_mouse_pos = QPointF(0, 0)
        if not width or not height:
            self.width = self.scene().views()[0].viewport().width()
            self.height = self.scene().views()[0].viewport().height()

        rect = QRectF(QPointF(0, 0), QSizeF(self.width, self.height))

        self.fused_image = QPixmap(self.width * 4, self.height * 4)
        self.pixmap_item.setPixmap(self.fused_image)

        self.dimensions = np.array(self.slide.level_dimensions)
        self.dim_count = self.slide.level_count
        self.patch_dims = [self.dimensions[self.dim_count - 1][0], self.dimensions[self.dim_count - 1][1]]

        self.down_sample_factors = [self.slide.level_downsamples[level] for level in range(self.slide.level_count)]

        self.fitInView(rect)

        self.cur_scaling_factor = max(self.dimensions[0][0] / self.width, self.dimensions[0][1] / self.height)
        self.max_scaling_factor = self.cur_scaling_factor
        self.cur_level = self.slide.get_best_level_for_downsample(self.cur_scaling_factor)
        self.relative_scaling_factor = self.cur_scaling_factor / self.down_sample_factors[self.cur_level]
        self.pixmap_item.moveBy(-self.width * (self.down_sample_factors[self.cur_level] / self.cur_scaling_factor),
                                -self.height * (self.down_sample_factors[self.cur_level] / self.cur_scaling_factor))

        max_threads = 16
        sqrt_thread_count = int(np.sqrt(max_threads))

        block_width = int(self.width)
        block_height = int(self.height)
        block_offset_width = int(self.width * self.down_sample_factors[self.cur_level])
        block_offset_height = int(self.height * self.down_sample_factors[self.cur_level])

        image_offset_width = int(self.width * self.down_sample_factors[self.cur_level])
        image_offset_height = int(self.height * self.down_sample_factors[self.cur_level])

        offset_mouse_pos = self.mouse_pos - QPointF(image_offset_width, image_offset_height)

        self.painter = QPainter(self.fused_image)

        with mp.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(self.process_image_block, i, offset_mouse_pos, block_width,
                                       block_height, block_offset_width, block_offset_height, sqrt_thread_count, False)
                       for i in range(max_threads)]

            mp.wait(futures)

            self.painter.end()

            self.pixmap_item.setPixmap(self.fused_image)
            self.pixmap_item.setScale(self.down_sample_factors[self.cur_level] / self.cur_scaling_factor)
            self.sendPixmap.emit(self.pixmap_item)

    def check_for_update(self, move, zoomed):
        self.width = self.scene().views()[0].viewport().width()
        self.height = self.scene().views()[0].viewport().height()

        self.cur_level = self.slide.get_best_level_for_downsample(self.cur_scaling_factor)
        self.relative_scaling_factor = self.cur_scaling_factor / self.down_sample_factors[self.cur_level]

        new_patches = []

        max_threads = 16
        sqrt_thread_count = int(np.sqrt(max_threads))

        if self.mouse_pos.x() >= self.width:
            self.mouse_pos.x() - self.width
            new_patches.extend([3, 7, 11, 15])
        if self.mouse_pos.x() < self.width:
            self.mouse_pos.x() + self.width
            new_patches.extend([0, 4, 8, 12])

        if self.mouse_pos.y() >= self.height:
            self.mouse_pos.y() - self.height
            new_patches.extend([0, 1, 2, 3])
        if self.mouse_pos.y() < self.height:
            self.mouse_pos.y() + self.height
            new_patches.extend([12, 13, 14, 15])

        new_patches = [*set(new_patches)]

       # if len(new_patches) == 0:
        original = self.fused_image.copy()
        self.painter = QPainter(self.fused_image)

        self.painter.drawPixmap(-move.x(),
                                -move.y(),
                                original.width(), original.height(), original)

        self.painter.end()
        self.pixmap_item.pixmap().fill(0)
        self.pixmap_item.setPixmap(self.fused_image)
        self.pixmap_item.setScale(self.down_sample_factors[self.cur_level] / self.cur_scaling_factor)
        #self.sendImage.emit(self.pixmap_item)


        # block_width = int(self.width)
        # block_height = int(self.height)
        # block_offset_width = int(self.width * self.down_sample_factors[self.cur_level])
        # block_offset_height = int(self.height * self.down_sample_factors[self.cur_level])
        #
        # image_offset_width = int(self.width * self.down_sample_factors[self.cur_level])
        # image_offset_height = int(self.height * self.down_sample_factors[self.cur_level])
        #
        # offset_mouse_pos = self.mouse_pos - QPointF(image_offset_width, image_offset_height)
        #
        # self.painter = QPainter(self.fused_image)
        #
        # with mp.ThreadPoolExecutor(max_workers=max_threads) as executor:
        #     futures = [executor.submit(self.process_image_block, i, offset_mouse_pos, block_width,
        #                                block_height, block_offset_width, block_offset_height, sqrt_thread_count, move)
        #                for i in range(max_threads)]
        #
        #     mp.wait(futures)
        #
        #     self.painter.end()
        #
        #     self.pixmap_item.setPixmap(self.fused_image)
        #     self.pixmap_item.setScale(self.down_sample_factors[self.cur_level] / self.cur_scaling_factor)
        #     self.imageFinished.emit()

    def process_image_block(self, block_index, mouse_pos, block_width, block_height, block_offset_width,
                            block_offset_height, sqrt_threads, moved):
        idx_width = block_index % sqrt_threads
        idx_height = block_index // sqrt_threads

        if not moved:

            block_location = (
                idx_width * block_offset_width,
                idx_height * block_offset_height
            )

            image = self.slide.read_region(
                (int(mouse_pos.x() + block_location[0]), int(mouse_pos.y() + block_location[1])),
                self.cur_level,
                (block_width, block_height)
            )

            self.image_blocks[block_index] = QPixmap.fromImage(ImageQT.ImageQt(image))

        self.painter.drawPixmap(idx_width * block_width - self.scene_mouse_pos.x(),
                                idx_height * block_height - self.scene_mouse_pos.y(),
                                block_width, block_height, self.image_blocks[block_index])



    def wheelEvent(self, event: QWheelEvent):
        """
        Scales the image and moves into the mouse position
        :param event: event to initialize the function
        :type event: QWheelEvent
        :return: /
        """
        old_scaling_factor = self.cur_scaling_factor
        old_scale = self.down_sample_factors[self.cur_level] / self.cur_scaling_factor

        scale_factor = 1.1 if event.angleDelta().y() <= 0 else 1 / 1.1
        new_scaling_factor = min(max(self.cur_scaling_factor * scale_factor, 1), self.max_scaling_factor)

        if new_scaling_factor == self.cur_scaling_factor:
            return

        self.cur_scaling_factor = new_scaling_factor
        self.cur_level = self.slide.get_best_level_for_downsample(self.cur_scaling_factor)
        old_relative_scaling_factor = self.relative_scaling_factor
        self.relative_scaling_factor = self.cur_scaling_factor/self.down_sample_factors[self.cur_level]
        self.pixmap_item.setScale(self.down_sample_factors[self.cur_level]/self.cur_scaling_factor)
        new_scale = self.down_sample_factors[self.cur_level]/self.cur_scaling_factor
        pixmap_pos = self.pixmap_item.pos()
        self.pixmap_item.setPos(QPointF(-self.width * (self.down_sample_factors[self.cur_level] / self.cur_scaling_factor) -
                                        (self.width * 2 - self.width * 2 * new_scale),
                                        -self.height * (self.down_sample_factors[self.cur_level] / self.cur_scaling_factor) -
                                        (self.height * 2 - self.height * 2 * new_scale)))
        # self.pixmap_item.moveBy(-self.width * (self.down_sample_factors[self.cur_level] / self.cur_scaling_factor),
        #                         -self.height * (self.down_sample_factors[self.cur_level] / self.cur_scaling_factor))

        new_pos = self.mapToScene(event.position().toPoint())

        self.scene_mouse_pos += QPointF(new_pos.x() - self.width/2, new_pos.y() - self.height/2)
        self.scene_mouse_pos += QPointF((self.width - self.width*scale_factor)/2,
                                        (self.height - self.height*scale_factor)/2)

        new_pos = QPointF((new_pos.x()/self.width - 0.5) *
                          self.dimensions[self.dim_count - self.cur_level - 1][0] * self.relative_scaling_factor,
                          (new_pos.y()/self.height - 0.5) *
                          self.dimensions[self.dim_count - self.cur_level - 1][1] * self.relative_scaling_factor)

        self.mouse_pos += QPointF(self.width/2 * (old_scaling_factor - self.cur_scaling_factor),
                                  self.height/2 * (old_scaling_factor - self.cur_scaling_factor))
        self.mouse_pos += new_pos
        #self.check_for_update(QPointF(0,0), True)

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
            self.scene_mouse_pos += move
            move = QPointF(move.x()/self.width*self.dimensions[self.dim_count - self.cur_level - 1][0],
                                      move.y()/self.height*self.dimensions[self.dim_count - self.cur_level - 1][1])
            self.pan_start = new_pos
            self.mouse_pos += move
            #self.check_for_update(move, False)
        super(QGraphicsView, self).mouseMoveEvent(event)
