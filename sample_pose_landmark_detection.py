#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import argparse
from typing import List, Any, Dict, Tuple, Union

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
    parser.add_argument('--unuse_mirror', action='store_true')
    parser.add_argument(
        "--model",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help='''
        0:Pose landmarker(lite)
        1:Pose landmarker(Full)
        2:Pose landmarker(Heavy)
        ''',
    )
    parser.add_argument('--use_output_segmentation_masks', action='store_true')
    parser.add_argument('--use_world_landmark', action='store_true')

    args = parser.parse_args()

    return args


def main() -> None:
    # 引数解析
    args = get_args()

    cap_device: Union[int, str] = args.device
    cap_width: int = args.width
    cap_height: int = args.height
    unuse_mirror: bool = args.unuse_mirror
    use_world_landmark: bool = args.use_world_landmark
    model: int = args.model
    use_output_segmentation_masks: bool = args.use_output_segmentation_masks

    if args.video is not None:
        cap_device = args.video

    model_url: List[str] = [
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
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

    # PoseLandmarker生成
    base_options: python.BaseOptions = python.BaseOptions(
        model_asset_path=model_path)
    options: vision.PoseLandmarkerOptions = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=use_output_segmentation_masks,
    )
    detector: vision.PoseLandmarker = vision.PoseLandmarker.create_from_options(
        options)  # type:ignore

    # FPS計測モジュール
    cvFpsCalc: CvFpsCalc = CvFpsCalc(buffer_len=10)

    # World座標プロット
    if use_world_landmark:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")  # type:ignore
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    while True:
        display_fps: float = cvFpsCalc.get()

        # カメラキャプチャ
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            break
        if not unuse_mirror:
            frame = cv2.flip(frame, 1)  # ミラー表示

        # 推論実施
        rgb_frame: mp.Image = mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA),
        )
        detection_result: vision.HandLandmarkerResult = detector.detect(
            rgb_frame)

        # 外接矩形計算
        bboxes: List[List[int]] = calc_bounding_rect(frame, detection_result)

        # 描画
        debug_image: np.ndarray = copy.deepcopy(frame)
        debug_image = draw_debug(
            debug_image,
            detection_result,
            bboxes,
            display_fps,
        )

        # 画面反映
        cv2.imshow('MediaPipe Pose Landmarks Detection Demo', debug_image)

        # 描画（ワールド座標）
        if use_world_landmark:
            draw_world_landmarks(
                plt,
                ax,
                detection_result,
            )

        # キー処理(ESC：終了)
        key: int = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


def calc_bounding_rect(
        image: np.ndarray,
        detection_result: vision.HandLandmarkerResult) -> List[List[int]]:
    image_width, image_height = image.shape[1], image.shape[0]

    bboxes: List[List[int]] = []
    for pose_landmarks in detection_result.pose_landmarks:
        landmark_array: np.ndarray = np.empty((0, 2), int)
        for landmark in pose_landmarks:
            landmark_x: int = min(int(landmark.x * image_width),
                                  image_width - 1)
            landmark_y: int = min(int(landmark.y * image_height),
                                  image_height - 1)

            landmark_point: np.ndarray = np.array((landmark_x, landmark_y))

            landmark_array = np.append(landmark_array, [landmark_point],
                                       axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)
        bboxes.append([x, y, x + w, y + h])

    return bboxes


def draw_debug(
    image: np.ndarray,
    detection_result: vision.HandLandmarkerResult,  # type:ignore
    bboxes: List[List[int]],
    display_fps: float,
) -> np.ndarray:
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_draw_info: Dict[
        int,
        Dict[str, Union[str, Tuple[int, int, int]]],
    ] = {
        0: {  # 鼻
            'name': 'NOSE',
            'color': (0, 255, 0)  # 緑
        },
        1: {  # 左目（内側）
            'name': 'LEFT_EYE_INNER',
            'color': (255, 0, 0)  # 赤
        },
        2: {  # 左目
            'name': 'LEFT_EYE',
            'color': (0, 0, 255)  # 青
        },
        3: {  # 左目（外側）
            'name': 'LEFT_EYE_OUTER',
            'color': (255, 255, 0)  # 黄
        },
        4: {  # 右目（内側）
            'name': 'RIGHT_EYE_INNER',
            'color': (0, 255, 255)  # シアン
        },
        5: {  # 右目
            'name': 'RIGHT_EYE',
            'color': (255, 0, 255)  # マゼンタ
        },
        6: {  # 右目（外側）
            'name': 'RIGHT_EYE_OUTER',
            'color': (128, 128, 128)  # グレー
        },
        7: {  # 左耳
            'name': 'LEFT_EAR',
            'color': (255, 128, 0)  # オレンジ
        },
        8: {  # 右耳
            'name': 'RIGHT_EAR',
            'color': (128, 0, 255)  # 紫
        },
        9: {  # 口（左）
            'name': 'MOUTH_LEFT',
            'color': (0, 128, 255)  # ライトブルー
        },
        10: {  # 口（右）
            'name': 'MOUTH_RIGHT',
            'color': (128, 255, 0)  # ライム
        },
        11: {  # 左肩
            'name': 'LEFT_SHOULDER',
            'color': (255, 128, 128)  # ライトレッド
        },
        12: {  # 右肩
            'name': 'RIGHT_SHOULDER',
            'color': (128, 128, 0)  # オリーブ
        },
        13: {  # 左肘
            'name': 'LEFT_ELBOW',
            'color': (0, 128, 128)  # ティール
        },
        14: {  # 右肘
            'name': 'RIGHT_ELBOW',
            'color': (128, 0, 128)  # マルーン
        },
        15: {  # 左手首
            'name': 'LEFT_WRIST',
            'color': (64, 64, 64)  # ダークグレー
        },
        16: {  # 右手首
            'name': 'RIGHT_WRIST',
            'color': (192, 192, 192)  # シルバー
        },
        17: {  # 左小指
            'name': 'LEFT_PINKY',
            'color': (255, 69, 0)  # レッドオレンジ
        },
        18: {  # 右小指
            'name': 'RIGHT_PINKY',
            'color': (75, 0, 130)  # インディゴ
        },
        19: {  # 左人差し指
            'name': 'LEFT_INDEX',
            'color': (173, 255, 47)  # グリーンイエロー
        },
        20: {  # 右人差し指
            'name': 'RIGHT_INDEX',
            'color': (220, 20, 60)  # クリムゾン
        },
        21: {  # 左親指
            'name': 'LEFT_THUMB',
            'color': (255, 0, 0)  # 赤
        },
        22: {  # 右親指
            'name': 'RIGHT_THUMB',
            'color': (0, 0, 255)  # 青
        },
        23: {  # 左腰
            'name': 'LEFT_HIP',
            'color': (0, 255, 0)  # 緑
        },
        24: {  # 右腰
            'name': 'RIGHT_HIP',
            'color': (255, 255, 0)  # 黄
        },
        25: {  # 左膝
            'name': 'LEFT_KNEE',
            'color': (0, 255, 255)  # シアン
        },
        26: {  # 右膝
            'name': 'RIGHT_KNEE',
            'color': (255, 0, 255)  # マゼンタ
        },
        27: {  # 左足首
            'name': 'LEFT_ANKLE',
            'color': (128, 128, 128)  # グレー
        },
        28: {  # 右足首
            'name': 'RIGHT_ANKLE',
            'color': (255, 128, 0)  # オレンジ
        },
        29: {  # 左かかと
            'name': 'LEFT_HEEL',
            'color': (128, 0, 255)  # 紫
        },
        30: {  # 右かかと
            'name': 'RIGHT_HEEL',
            'color': (0, 128, 255)  # ライトブルー
        },
        31: {  # 左足指先
            'name': 'LEFT_FOOT_INDEX',
            'color': (128, 255, 0)  # ライム
        },
        32: {  # 右足指先
            'name': 'RIGHT_FOOT_INDEX',
            'color': (255, 128, 128)  # ライトレッド
        }
    }

    line_info_list: List[List[int]] = [
        [0, 1],  # 鼻から左目（内側）
        [1, 2],  # 左目（内側）から左目
        [2, 3],  # 左目から左目（外側）
        [3, 7],  # 左目（外側）から左耳
        [0, 4],  # 鼻から右目（内側）
        [4, 5],  # 右目（内側）から右目
        [5, 6],  # 右目から右目（外側）
        [6, 8],  # 右目（外側）から右耳
        [9, 10],  # 口（左）から口（右）
        [11, 12],  # 左肩から右肩
        [11, 13],  # 左肩から左肘
        [13, 15],  # 左肘から左手首
        [15, 17],  # 左手首から左小指
        [15, 19],  # 左手首から左人差し指
        [15, 21],  # 左手首から左親指
        [12, 14],  # 右肩から右肘
        [14, 16],  # 右肘から右手首
        [16, 18],  # 右手首から右小指
        [16, 20],  # 右手首から右人差し指
        [16, 22],  # 右手首から右親指
        [23, 24],  # 左腰から右腰
        [23, 25],  # 左腰から左膝
        [25, 27],  # 左膝から左足首
        [27, 29],  # 左足首から左かかと
        [29, 31],  # 左かかとから左足指先
        [24, 26],  # 右腰から右膝
        [26, 28],  # 右膝から右足首
        [28, 30],  # 右足首から右かかと
        [30, 32],  # 右かかとから右足指先
        [11, 23],  # 左肩から左腰
        [12, 24]  # 右肩から右腰
    ]

    # セグメンテーション
    if detection_result.segmentation_masks is not None:
        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        mask = np.stack((segmentation_mask, ) * 3, axis=-1) > 0.5
        bg_resize_image = np.zeros(image.shape, dtype=np.uint8)
        bg_resize_image[:] = (0, 255, 0)
        image = np.where(mask, image, bg_resize_image)

    for pose_landmarks, _, bbox in zip(
            detection_result.pose_landmarks,
            detection_result.pose_world_landmarks,
            bboxes,
    ):
        # 各ランドマーク情報整理
        landmark_dict: Dict[int, List[Union[int, float]]] = {}
        for index, landmark in enumerate(pose_landmarks):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue
            landmark_x: int = min(int(landmark.x * image_width),
                                  image_width - 1)
            landmark_y: int = min(int(landmark.y * image_height),
                                  image_height - 1)
            landmark_dict[index] = [landmark_x, landmark_y, landmark.z]

        # 接続線描画
        for line_info in line_info_list:
            cv2.line(image, tuple(landmark_dict[line_info[0]][:2]),
                     tuple(landmark_dict[line_info[1]][:2]), (220, 220, 220),
                     3, cv2.LINE_AA)  # type:ignore

        # 各ランドマーク描画
        for index, landmark in landmark_dict.items():
            cv2.circle(image, (landmark[0], landmark[1]), 5,
                       landmark_draw_info[index]['color'], -1,
                       cv2.LINE_AA)  # type:ignore

        # 外接矩形描画
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (255, 255, 255), 2)

    # FPS
    if detection_result.segmentation_masks is None:
        color = (0, 255, 0)
    else:
        color = (255, 255, 255)
    cv2.putText(
        image,
        "FPS:" + str(display_fps),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )

    return image


def draw_world_landmarks(
    plt: Any,
    ax: Any,
    detection_result: vision.HandLandmarkerResult,
) -> None:
    for _, pose_world_landmarks in zip(
            detection_result.pose_landmarks,
            detection_result.pose_world_landmarks,
    ):
        # 各ランドマーク情報整理
        landmark_dict: Dict[int, List[float]] = {}
        for index, landmark in enumerate(pose_world_landmarks):
            landmark_dict[index] = [landmark.x, landmark.y, landmark.z]

        face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        right_arm_index_list = [11, 13, 15, 17, 19, 21]
        left_arm_index_list = [12, 14, 16, 18, 20, 22]
        right_body_side_index_list = [11, 23, 25, 27, 29, 31]
        left_body_side_index_list = [12, 24, 26, 28, 30, 32]
        shoulder_index_list = [11, 12]
        waist_index_list = [23, 24]

        # 顔
        face_x, face_y, face_z = [], [], []
        for index in face_index_list:
            point = landmark_dict[index]
            face_x.append(point[0])
            face_y.append(point[2])
            face_z.append(point[1] * (-1))

        # 右腕
        right_arm_x, right_arm_y, right_arm_z = [], [], []
        for index in right_arm_index_list:
            point = landmark_dict[index]
            right_arm_x.append(point[0])
            right_arm_y.append(point[2])
            right_arm_z.append(point[1] * (-1))

        # 左腕
        left_arm_x, left_arm_y, left_arm_z = [], [], []
        for index in left_arm_index_list:
            point = landmark_dict[index]
            left_arm_x.append(point[0])
            left_arm_y.append(point[2])
            left_arm_z.append(point[1] * (-1))

        # 右半身
        right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
        for index in right_body_side_index_list:
            point = landmark_dict[index]
            right_body_side_x.append(point[0])
            right_body_side_y.append(point[2])
            right_body_side_z.append(point[1] * (-1))

        # 左半身
        left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
        for index in left_body_side_index_list:
            point = landmark_dict[index]
            left_body_side_x.append(point[0])
            left_body_side_y.append(point[2])
            left_body_side_z.append(point[1] * (-1))

        # 肩
        shoulder_x, shoulder_y, shoulder_z = [], [], []
        for index in shoulder_index_list:
            point = landmark_dict[index]
            shoulder_x.append(point[0])
            shoulder_y.append(point[2])
            shoulder_z.append(point[1] * (-1))

        # 腰
        waist_x, waist_y, waist_z = [], [], []
        for index in waist_index_list:
            point = landmark_dict[index]
            waist_x.append(point[0])
            waist_y.append(point[2])
            waist_z.append(point[1] * (-1))

        ax.cla()
        ax.set_xlim3d(-1.0, 1.0)
        ax.set_ylim3d(-1.0, 1.0)
        ax.set_zlim3d(-1.0, 1.0)

        ax.scatter(face_x, face_y, face_z)
        ax.plot(right_arm_x, right_arm_y, right_arm_z)
        ax.plot(left_arm_x, left_arm_y, left_arm_z)
        ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
        ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
        ax.plot(shoulder_x, shoulder_y, shoulder_z)
        ax.plot(waist_x, waist_y, waist_z)

    plt.pause(.001)

    return


if __name__ == '__main__':
    main()
