#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
from typing import List, Any
import time

import cv2
import mediapipe as mp  # type:ignore
from mediapipe.tasks import python  # type:ignore
from mediapipe.tasks.python import vision  # type:ignore

from utils.download_file import download_file


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image01",
        type=str,
        default='asset/hedgehog01.jpg',
    )
    parser.add_argument(
        "--image02",
        type=str,
        default='asset/hedgehog02.jpg',
    )

    parser.add_argument(
        "--model",
        type=int,
        choices=[0, 1],
        default=0,
        help='''
        0:MobileNet-V3 (small)
        1:MobileNet-V3 (large)
        ''',
    )

    parser.add_argument(
        "--unuse_l2_normalize",
        action="store_true",
    )
    parser.add_argument(
        "--unuse_quantize",
        action="store_true",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    # 引数解析
    args: argparse.Namespace = get_args()

    image01_path: str = args.image01
    image02_path: str = args.image02

    model: int = args.model
    unuse_l2_normalize: bool = args.unuse_l2_normalize
    unuse_quantize: bool = args.unuse_quantize

    use_l2_normalize: bool = not unuse_l2_normalize
    use_quantize: bool = not unuse_quantize

    model_url: List[str] = [
        'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite',
        'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/latest/mobilenet_v3_large.tflite',
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

    # Image Embedder生成
    base_options: python.BaseOptions = python.BaseOptions(
        model_asset_path=model_path)
    options: vision.ImageEmbedderOptions = vision.ImageEmbedderOptions(
        base_options=base_options,
        l2_normalize=use_l2_normalize,
        quantize=use_quantize,
    )
    embedder: vision.ImageEmbedder = vision.ImageEmbedder.create_from_options(
        options)  # type:ignore

    # 画像準備
    image01: Any = cv2.imread(image01_path)
    image02: Any = cv2.imread(image02_path)

    # 処理時間計測開始
    start_time: float = time.time()

    # 推論実施
    rgb_frame01: mp.Image = mp.Image(
        image_format=mp.ImageFormat.SRGBA,
        data=cv2.cvtColor(image01, cv2.COLOR_BGR2RGBA),
    )
    rgb_frame02: mp.Image = mp.Image(
        image_format=mp.ImageFormat.SRGBA,
        data=cv2.cvtColor(image02, cv2.COLOR_BGR2RGBA),
    )
    embedding_result01: vision.ImageEmbedderResult = embedder.embed(
        rgb_frame01)
    embedding_result02: vision.ImageEmbedderResult = embedder.embed(
        rgb_frame02)

    # 類似度計算
    similarity: float = vision.ImageEmbedder.cosine_similarity(
        embedding_result01.embeddings[0],
        embedding_result02.embeddings[0],
    )

    # 処理時間計測終了
    end_time: float = time.time()
    elapsed_time: int = int((end_time - start_time) * 1000)

    print()
    print('MediaPipe Image Embedder Demo')
    print('  Similarity:', similarity)
    print('  Processing time:', elapsed_time, 'ms')


if __name__ == '__main__':
    main()
