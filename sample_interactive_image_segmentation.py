#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import argparse
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
import mediapipe as mp  # type:ignore
from mediapipe.tasks import python  # type:ignore
from mediapipe.tasks.python import vision  # type:ignore
from mediapipe.tasks.python.components import containers  # type:ignore

from utils import CvFpsCalc
from utils.download_file import download_file


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        default='asset/hedgehog01.jpg',
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=[0],
        default=0,
        help='''
        0:MagicTouch
        ''',
    )

    args = parser.parse_args()

    return args


def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    param['x'] = x
    param['y'] = y


def main() -> None:
    # 引数解析
    args = get_args()

    image_path: str = args.image
    model: int = args.model

    model_url: List[str] = [
        'https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/latest/magic_touch.tflite',
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

    # 画像準備
    image: np.ndarray = cv2.imread(image_path)

    # Interactive Segmenter生成
    region_of_interest = vision.InteractiveSegmenterRegionOfInterest
    normalized_keypoint = containers.keypoint.NormalizedKeypoint

    base_options: python.BaseOptions = python.BaseOptions(
        model_asset_path=model_path)
    options: vision.ImageSegmenterOptions = vision.ImageSegmenterOptions(
        base_options=base_options, output_category_mask=True)
    segmenter: vision.InteractiveSegmenter = vision.InteractiveSegmenter.create_from_options(
        options)  # type:ignore

    # デバッグ用カラーテーブル
    colortable: List[Tuple[int, int, int]] = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Red
    ]

    # FPS計測モジュール
    cvFpsCalc: CvFpsCalc = CvFpsCalc(buffer_len=10)

    # マウスコールバック準備
    window_name: str = 'MediaPipe Interactive Segmentation Demo'
    mouse_param: Dict[str, Any] = {'x': 0, 'y': 0, 'l_button_click': False}

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name,
        mouse_callback,
        mouse_param,
    )  # type:ignore

    while True:
        display_fps: float = cvFpsCalc.get()

        image_width: int = image.shape[1]
        image_height: int = image.shape[0]

        # 推論実施
        target_x: float = mouse_param['x'] / image_width
        target_y: float = mouse_param['y'] / image_height
        roi = region_of_interest(
            format=region_of_interest.Format.KEYPOINT,
            keypoint=normalized_keypoint(target_x, target_y),
        )
        rgb_frame: mp.Image = mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGBA),
        )
        segmentation_result: vision.ImageSegmenterResult = segmenter.segment(
            rgb_frame, roi)

        # 後処理
        category_mask = segmentation_result.category_mask
        category_mask = category_mask.numpy_view()

        # 描画
        debug_image: np.ndarray = copy.deepcopy(image)
        debug_image = draw_debug(
            debug_image,
            category_mask,
            display_fps,
            colortable,
        )

        # 画面反映
        cv2.imshow(window_name, debug_image)

        # キー処理(ESC：終了)
        key: int = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


def draw_debug(
    image: np.ndarray,
    category_mask: Optional[np.ndarray],
    display_fps: float,
    color_table: List[Tuple[int, int, int]],
) -> np.ndarray:
    if category_mask is not None:
        # 255を除く最大値を取得
        max_value: int = np.max(category_mask[category_mask < 255]) + 1

        # セグメンテーション色分け
        for index in range(0, max_value):
            mask: np.ndarray = np.where(category_mask == index, 0, 1)

            bg_image: np.ndarray = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (
                color_table[index][2],
                color_table[index][1],
                color_table[index][0],
            )

            # 重畳表示
            mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
            mask_image: np.ndarray = np.where(mask, image, bg_image)
            image = cv2.addWeighted(image, 0.5, mask_image, 0.5, 1.0)

    # FPS
    cv2.putText(
        image,
        "FPS:" + str(display_fps),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return image


if __name__ == '__main__':
    main()
