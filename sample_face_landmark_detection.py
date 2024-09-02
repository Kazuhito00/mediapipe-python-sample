#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import argparse
from typing import List, Any, Dict, Tuple, Union

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
        0:FaceLandscapeer
        ''',
    )
    parser.add_argument("--num_faces", type=int, default=1)
    parser.add_argument("--unuse_output_face_blendshapes", action="store_true")
    parser.add_argument("--unuse_output_facial_transformation_matrixes",
                        action="store_true")

    args = parser.parse_args()

    return args


def main() -> None:
    # 引数解析
    args: argparse.Namespace = get_args()

    cap_device: int = args.device
    cap_width: int = args.width
    cap_height: int = args.height

    model: int = args.model
    num_faces = args.num_faces
    unuse_output_face_blendshapes = args.unuse_output_face_blendshapes
    unuse_output_facial_transformation_matrixes = args.unuse_output_facial_transformation_matrixes

    use_output_face_blendshapes = not unuse_output_face_blendshapes
    use_output_facial_transformation_matrixes = not unuse_output_facial_transformation_matrixes

    if args.video is not None:
        cap_device = args.video

    model_url: List[str] = [
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
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
    options: vision.FaceLandmarkerOptions = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=use_output_face_blendshapes,
        output_facial_transformation_matrixes=
        use_output_facial_transformation_matrixes,
        num_faces=num_faces,
    )
    detector: vision.FaceLandmarker = vision.FaceLandmarker.create_from_options(
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
        detection_result: vision.FaceLandmarkerResult = detector.detect(
            rgb_frame)

        # 描画
        debug_image: Any = copy.deepcopy(frame)
        debug_image = draw_debug(
            debug_image,
            detection_result,
            display_fps,
        )

        # 画面反映
        cv2.imshow('MediaPipe Face Landmark Detection Demo', debug_image)

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

    # ランドマーク表示
    landmark_dict: Dict[int, List[Union[int, float]]] = {}
    for face_landmarks in detection_result.face_landmarks:
        for index, face_landmark in enumerate(face_landmarks):
            face_landmark_x: int = int(image_width * face_landmark.x)
            face_landmark_y: int = int(image_height * face_landmark.y)
            cv2.circle(image, (face_landmark_x, face_landmark_y), 1,
                       (0, 255, 0), -1, cv2.LINE_AA)

            landmark_dict[index] = [face_landmark_x, face_landmark_y]

        # 左眉毛(55：内側、46：外側)
        cv2.line(image, landmark_dict[55], landmark_dict[65], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[65], landmark_dict[52], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[52], landmark_dict[53], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[53], landmark_dict[46], (0, 255, 0), 2)

        # 右眉毛(285：内側、276：外側)
        cv2.line(image, landmark_dict[285], landmark_dict[295], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[295], landmark_dict[282], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[282], landmark_dict[283], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[283], landmark_dict[276], (0, 255, 0), 2)

        # 左目 (133：目頭、246：目尻)
        cv2.line(image, landmark_dict[133], landmark_dict[173], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[173], landmark_dict[157], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[157], landmark_dict[158], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[158], landmark_dict[159], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[159], landmark_dict[160], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[160], landmark_dict[161], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[161], landmark_dict[246], (0, 255, 0), 2)

        cv2.line(image, landmark_dict[246], landmark_dict[163], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[163], landmark_dict[144], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[144], landmark_dict[145], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[145], landmark_dict[153], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[153], landmark_dict[154], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[154], landmark_dict[155], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[155], landmark_dict[133], (0, 255, 0), 2)

        # 右目 (362：目頭、466：目尻)
        cv2.line(image, landmark_dict[362], landmark_dict[398], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[398], landmark_dict[384], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[384], landmark_dict[385], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[385], landmark_dict[386], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[386], landmark_dict[387], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[387], landmark_dict[388], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[388], landmark_dict[466], (0, 255, 0), 2)

        cv2.line(image, landmark_dict[466], landmark_dict[390], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[390], landmark_dict[373], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[373], landmark_dict[374], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[374], landmark_dict[380], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[380], landmark_dict[381], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[381], landmark_dict[382], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[382], landmark_dict[362], (0, 255, 0), 2)

        # 口 (308：右端、78：左端)
        cv2.line(image, landmark_dict[308], landmark_dict[415], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[415], landmark_dict[310], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[310], landmark_dict[311], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[311], landmark_dict[312], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[312], landmark_dict[13], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[13], landmark_dict[82], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[82], landmark_dict[81], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[81], landmark_dict[80], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[80], landmark_dict[191], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[191], landmark_dict[78], (0, 255, 0), 2)

        cv2.line(image, landmark_dict[78], landmark_dict[95], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[95], landmark_dict[88], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[88], landmark_dict[178], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[178], landmark_dict[87], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[87], landmark_dict[14], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[14], landmark_dict[317], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[317], landmark_dict[402], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[402], landmark_dict[318], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[318], landmark_dict[324], (0, 255, 0), 2)
        cv2.line(image, landmark_dict[324], landmark_dict[308], (0, 255, 0), 2)

        # 左目：中心
        cv2.circle(image, landmark_dict[468], 2, (0, 0, 255), -1)
        # 左目：目頭側
        cv2.circle(image, landmark_dict[469], 2, (0, 0, 255), -1)
        # 左目：上側
        cv2.circle(image, landmark_dict[470], 2, (0, 0, 255), -1)
        # 左目：目尻側
        cv2.circle(image, landmark_dict[471], 2, (0, 0, 255), -1)
        # 左目：下側
        cv2.circle(image, landmark_dict[472], 2, (0, 0, 255), -1)
        # 右目：中心
        cv2.circle(image, landmark_dict[473], 2, (0, 0, 255), -1)
        # 右目：目尻側
        cv2.circle(image, landmark_dict[474], 2, (0, 0, 255), -1)
        # 右目：上側
        cv2.circle(image, landmark_dict[475], 2, (0, 0, 255), -1)
        # 右目：目頭側
        cv2.circle(image, landmark_dict[476], 2, (0, 0, 255), -1)
        # 右目：下側
        cv2.circle(image, landmark_dict[477], 2, (0, 0, 255), -1)

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
