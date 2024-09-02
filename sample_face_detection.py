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
        choices=[0],
        default=0,
        help='''
        0:BlazeFace (short-range)
        1:BlazeFace (full-range)  # 公式準備中で未公開
        2:BlazeFace Sparse (full-range)  # 公式準備中で未公開
        ''',
    )

    args = parser.parse_args()

    return args


def main() -> None:
    # 引数解析
    args: argparse.Namespace = get_args()

    cap_device: int = args.device
    cap_width: int = args.width
    cap_height: int = args.height

    model: int = args.model

    if args.video is not None:
        cap_device = args.video

    model_url: List[str] = [
        'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite',
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

    # Face Detector生成
    base_options: python.BaseOptions = python.BaseOptions(
        model_asset_path=model_path)
    options: vision.FaceDetectorOptions = vision.FaceDetectorOptions(
        base_options=base_options, )
    detector: vision.FaceDetector = vision.FaceDetector.create_from_options(
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
        detection_result: vision.ObjectDetectionResult = detector.detect(
            rgb_frame)

        # 描画
        debug_image: Any = copy.deepcopy(frame)
        debug_image = draw_debug(
            debug_image,
            detection_result,
            display_fps,
        )

        # 画面反映
        cv2.imshow('MediaPipe Face Detection Demo', debug_image)

        # キー処理(ESC：終了)
        key: int = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_debug(
    image: Any,
    detection_result,  # type:ignore
    display_fps: float,
) -> Any:
    image_width: int = image.shape[1]
    image_height: int = image.shape[0]

    for detection_info in detection_result.detections:
        # バウンディングボックス
        x1: int = detection_info.bounding_box.origin_x
        y1: int = detection_info.bounding_box.origin_y
        x2: int = x1 + detection_info.bounding_box.width
        y2: int = y1 + detection_info.bounding_box.height
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # スコア
        score: float = detection_info.categories[0].score
        cv2.putText(
            image,
            str(round(score, 3)),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        for keypoint in detection_info.keypoints:
            keypoint_x: int = int(image_width * keypoint.x)
            keypoint_y: int = int(image_height * keypoint.y)
            cv2.circle(image, (keypoint_x, keypoint_y), 5, (0, 255, 0), -1,
                       cv2.LINE_AA)

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
