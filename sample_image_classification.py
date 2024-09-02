#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import argparse
from typing import List, Any

import cv2
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
        choices=[0, 1, 2, 3],
        default=0,
        help='''
        0:EfficientNet-Lite0(int8)
        1:EfficientNet-Lite0(float 32)
        2:EfficientNet-Lite2(int8)
        3:EfficientNet-Lite2(float 32)
        ''',
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=5,
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
    max_results: int = args.max_results

    if args.video is not None:
        cap_device = args.video

    model_url: List[str] = [
        'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/int8/latest/efficientnet_lite0.tflite',
        'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite',
        'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/float32/latest/efficientnet_lite2.tflite',
        'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/float32/latest/efficientnet_lite2.tflite',
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

    # Classifier生成
    base_options: python.BaseOptions = python.BaseOptions(
        model_asset_path=model_path)
    options: vision.ImageClassifierOptions = vision.ImageClassifierOptions(
        base_options=base_options, max_results=max_results)
    classifier: vision.ImageClassifier = vision.ImageClassifier.create_from_options(
        options)  # type:ignore

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
        classification_result: vision.ClassificationResult = classifier.classify(
            rgb_frame)

        # 描画
        debug_image: Any = copy.deepcopy(frame)
        debug_image = draw_debug(
            debug_image,
            classification_result,
            display_fps,
        )

        # 画面反映
        cv2.imshow('MediaPipe Image Classification Demo', debug_image)

        # キー処理(ESC：終了)
        key: int = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_debug(
    image: Any,
    classification_result,  # type:ignore
    display_fps: float,
) -> Any:
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

    # カテゴリー名
    classification_info = classification_result.classifications[0]
    for index, category_info in enumerate(classification_info.categories):
        category_name: str = category_info.category_name
        score: float = category_info.score
        cv2.putText(
            image,
            category_name + ":" + str(round(score, 3)),
            (10, 40 + (25 * (index + 1))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return image


if __name__ == '__main__':
    main()
