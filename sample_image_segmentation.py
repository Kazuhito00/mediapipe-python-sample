#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import argparse
from typing import List, Any

import cv2
import numpy as np
import mediapipe as mp  # type:ignore
from mediapipe.tasks import python  # type:ignore
from mediapipe.tasks.python import vision  # type:ignore

from utils import CvFpsCalc
from utils.download_file import download_file


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--model",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=0,
        help='''
        0:SelfieSegmenter(square)
        1:SelfieSegmenter(landscape)
        2:HairSegmenter
        3:SelfieMulticlass(256x256)
        4:DeepLab-V3
        ''',
    )

    args = parser.parse_args()

    return args


def main() -> None:
    # 引数解析
    args = get_args()

    cap_device: int = args.device
    cap_width: int = args.width
    cap_height: int = args.height

    model: int = args.model

    if args.video is not None:
        cap_device = args.video

    model_url: List[str] = [
        'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
        'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite',
        'https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite',
        'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
        'https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/latest/deeplab_v3.tflite',
    ]

    # ダウンロードファイル名生成
    model_name: str = model_url[model].split('/')[-1]
    quantize_type: str = model_url[model].split('/')[-3]
    split_name: List[str] = model_name.split('.')
    model_name = split_name[0] + '_' + quantize_type + '.' + split_name[1]

    # 重みファイルダウンロード
    model_path: str = os.path.join('model', model_name)
    if not os.path.exists(model_path):
        download_file(url=model_url[model], save_path=model_path)

    # カメラ準備
    cap: cv2.VideoCapture = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Segmenter生成
    base_options: python.BaseOptions = python.BaseOptions(
        model_asset_path=model_path)
    options: vision.ImageSegmenterOptions = vision.ImageSegmenterOptions(
        base_options=base_options, output_category_mask=True)
    segmenter: vision.ImageSegmenter = vision.ImageSegmenter.create_from_options(
        options)  # type:ignore

    # デバッグ用カラーテーブル
    colortable = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Red
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 128, 128),  # Gray
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Light Blue
        (128, 255, 0),  # Lime
        (255, 128, 128),  # Light Red
        (128, 128, 0),  # Olive
        (0, 128, 128),  # Teal
        (128, 0, 128),  # Maroon
        (64, 64, 64),  # Dark Gray
    ]

    # FPS計測モジュール
    cvFpsCalc: CvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps: float = cvFpsCalc.get()

        # カメラキャプチャ
        ret: bool
        frame: Any
        ret, frame = cap.read()
        if not ret:
            break

        # 推論実施
        rgb_frame: mp.Image = mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA),
        )
        segmentation_result: vision.SegmentationResult = segmenter.segment(
            rgb_frame)

        # 後処理
        category_mask = segmentation_result.category_mask
        category_mask = category_mask.numpy_view()

        # 描画
        debug_image: Any = copy.deepcopy(frame)
        debug_image = draw_debug(
            debug_image,
            category_mask,
            display_fps,
            colortable,
        )

        # 画面反映
        cv2.imshow('MediaPipe Segmentation Demo', debug_image)

        # キー処理(ESC：終了)
        key: int = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_debug(
    image: Any,
    category_mask: Any,
    display_fps: float,
    color_table,
) -> Any:

    # 255を除く最大値を取得
    temp_category_mask = category_mask[category_mask < 255]
    if temp_category_mask.size > 0:
        max_value = np.max(temp_category_mask) + 1
    else:
        max_value = 0

    # セグメンテーション色分け
    for index in range(0, max_value):
        mask = np.where(category_mask == index, 0, 1)

        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (
            color_table[index][2],
            color_table[index][1],
            color_table[index][0],
        )

        # 重畳表示
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
        mask_image = np.where(mask, image, bg_image)
        image = cv2.addWeighted(image, 0.5, mask_image, 0.5, 1.0)

    # FPS
    cv2.putText(
        image,
        "FPS:" + str(display_fps),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return image


if __name__ == '__main__':
    main()
