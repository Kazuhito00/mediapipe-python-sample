> [!IMPORTANT]
> MediaPipe レガシーソリューションのサポートは、2023年3月1日で終了しています。<br>
> 従来のソリューションのサンプルは [_legacy](_legacy)ディレクトリに移動しました。<br>
> MediaPipeは後方互換を保っており、現パッケージでもレガシーソリューションのサンプルを実行出来ます。<br>

# mediapipe-python-sample
[google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe)のPythonパッケージのサンプルスクリプト集です。<br>
2024/9/1時点でPython実装のある以下15機能について用意しています。
* [物体検出（Object Detection）](https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector?hl=ja)
* [画像分類（Image Classification）](https://ai.google.dev/mediapipe/solutions/vision/image_classifier?hl=ja)
* [画像セグメンテーション（Image Segmentation）](https://ai.google.dev/mediapipe/solutions/vision/image_segmenter?hl=ja)
* [インタラクティブ セグメンテーション（Interactive segmentation）](https://ai.google.dev/mediapipe/solutions/vision/interactive_segmenter?hl=ja)
* [手検出（Hand Landmark detection）](https://ai.google.dev/mediapipe/solutions/vision/hand_landmarker?hl=ja)
* [手のジェスチャー認識（Gesture Recognition）](https://ai.google.dev/mediapipe/solutions/vision/gesture_recognizer?hl=ja)
* [画像の埋め込み表現（Image Embedding）](https://ai.google.dev/mediapipe/solutions/vision/image_embedder?hl=ja)
* [顔検出（Face Detection）](https://ai.google.dev/mediapipe/solutions/vision/face_detector?hl=ja)
* [顔のランドマーク検出（Face Landmark Detection）](https://ai.google.dev/mediapipe/solutions/vision/face_landmarker?hl=ja)
* [顔のスタイル変換（Face Stylization）](https://ai.google.dev/mediapipe/solutions/vision/face_stylizer?hl=ja)
* [姿勢推定（Pose Landmark Detection）](https://ai.google.dev/mediapipe/solutions/vision/pose_landmarker?hl=ja)
* [テキスト分類（Text Classification）](https://ai.google.dev/mediapipe/solutions/text/text_classifier?hl=ja)
* [テキストの埋め込み表現（Text Embedding）](https://ai.google.dev/mediapipe/solutions/text/text_embedder?hl=ja)
* [テキスト言語分類（Language Detector）](https://ai.google.dev/mediapipe/solutions/text/language_detector?hl=ja)
* [音分類（Audio Classification）](https://ai.google.dev/mediapipe/solutions/audio/audio_classifier?hl=ja)

# Requirement 
* mediapipe 0.10.14 or later
* opencv-python 4.10.0.84 or later
* tqdm 4.66.5 or later　※重みファイルダウンロードに使用
* requests 2.32.3 or later　※重みファイルダウンロードに使用
* scipy 1.14.1 or later　※音分類（Audio Classification）サンプルを実行する場合のみ
* numpy 1.26.4　※NumPyは1.x系

```
pip install -r requirements.txt
```

# Demo
デモの実行方法は以下です。

### 物体検出（Object Detection）
```bash
python sample_object_detection.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --video<br>
動画パスの指定 ※指定時はカメラより優先<br>
デフォルト：None
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
使用モデル[0, 1, 2, 3, 4, 5, 6, 7]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
[COCOデータセット](https://cocodataset.org/#home)でトレーニングされた重みで、サポートされているラベルは[labelmap.txt](https://storage.googleapis.com/mediapipe-tasks/object_detector/labelmap.txt)<br>
デフォルト：0<br>
  * 0:EfficientDet-Lite0(int8)
  * 1:EfficientDet-Lite0(float 16)
  * 2:EfficientDet-Lite0(float 32)
  * 3:EfficientDet-Lite2(int8)
  * 4:EfficientDet-Lite2(float 16)
  * 5:EfficientDet-Lite2float 32）
  * 6:SSDMobileNet-V2(int8)
  * 7:SSDMobileNet-V2(float 32)
* --score_threshold<br>
スコア閾値<br>
デフォルト：0.5
</details>
<img src="https://github.com/user-attachments/assets/c006ff20-f3a4-413e-9f37-99f7faaa07bb" loading="lazy" width="250px">

### 画像分類（Image Classification）
```bash
python sample_image_classification.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --video<br>
動画パスの指定 ※指定時はカメラより優先<br>
デフォルト：None
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
使用モデル[0, 1, 2, 3]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
[ImageNet](https://www.image-net.org/)でトレーニングされた重みで、サポートされているラベルは[labels.txt](https://storage.googleapis.com/mediapipe-tasks/image_classifier/labels.txt)<br>
デフォルト：0<br>
  * 0:EfficientNet-Lite0(int8)
  * 1:EfficientNet-Lite0(float 32)
  * 2:EfficientNet-Lite2(int8)
  * 3:EfficientNet-Lite2(float 32)
* --max_results<br>
結果出力数<br>
デフォルト：5
</details>
<img src="https://github.com/user-attachments/assets/fd74c89c-9d03-4862-84dc-210a35208017" loading="lazy" width="250px">

### 画像セグメンテーション（Image Segmentation）
```bash
python sample_image_segmentation.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --video<br>
動画パスの指定 ※指定時はカメラより優先<br>
デフォルト：None
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
使用モデル[0, 1, 2, 3, 4]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
デフォルト：0<br>
  * 0:SelfieSegmenter(square)
  * 1:SelfieSegmenter(landscape)
  * 2:HairSegmenter
  * 3:SelfieMulticlass(256x256)
  * 4:DeepLab-V3
</details>
<img src="https://github.com/user-attachments/assets/f28b6ece-26a5-4ddc-be70-1bd27dfefd3e" loading="lazy" width="250px"> <img src="https://github.com/user-attachments/assets/25c048b4-860c-45cc-a20a-5a0bc1756d7c" loading="lazy" width="250px"> <img src="https://github.com/user-attachments/assets/2f9dc84d-6c96-4c29-a5d3-553a75d1e89e" loading="lazy" width="250px">


### インタラクティブ セグメンテーション（Interactive segmentation）
```bash
python sample_interactive_image_segmentation.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --image<br>
画像パスの指定<br>
デフォルト：asset/hedgehog01.jpg
* --model<br>
使用モデル[0]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
デフォルト：0<br>
  * 0:MagicTouch
</details>
<img src="https://github.com/user-attachments/assets/bccf93ba-c28d-4003-a643-65bc925f7a77" loading="lazy" width="250px">

### 手検出（Hand Landmark detection）
```bash
python sample_hand_landmarks_detection.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --video<br>
動画パスの指定 ※指定時はカメラより優先<br>
デフォルト：None
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --unuse_mirror<br>
ミラー表示不使用<br>
デフォルト：指定なし
* --model<br>
使用モデル[0]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
デフォルト：0<br>
  * 0:HandLandmarker (full)
* --num_hands<br>
検出数<br>
デフォルト：2
* --use_world_landmark<br>
ワールド座標表示<br>
デフォルト：指定なし
</details>
<img src="https://github.com/user-attachments/assets/b6db982a-fb64-490b-abbf-6d15de141f1b" loading="lazy" width="250px"> <img src="https://github.com/user-attachments/assets/5fd708c0-4394-4a2d-b9fc-cab38b5278f0" loading="lazy" width="250px">

### 手のジェスチャー認識（Gesture Recognition）
```bash
python sample_hand_gesture_recognition.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --video<br>
動画パスの指定 ※指定時はカメラより優先<br>
デフォルト：None
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --unuse_mirror<br>
ミラー表示不使用<br>
デフォルト：指定なし
* --model<br>
使用モデル[0]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
認識ジェスチャーは「Closed fist」「Open palm」「Pointing up」「Thumbs down」「Thumbs up」「Victory」「Love」「Unknown」<br>
デフォルト：0<br>
  * 0:HandGestureClassifier
</details>
<img src="https://github.com/user-attachments/assets/02de2c97-e29a-4246-bf4a-70b91a09032b" loading="lazy" width="250px">

### 画像の埋め込み表現（Image Embedding）
```bash
python sample_image_embedding.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --image01<br>
画像パス1の指定<br>
デフォルト：asset/hedgehog01.jpg
* --image02<br>
画像パス2の指定<br>
デフォルト：asset/hedgehog02.jpg
* --model<br>
使用モデル[0, 1]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
デフォルト：0<br>
  * 0:MobileNet-V3 (small)
  * 1:MobileNet-V3 (large)
* --unuse_l2_normalize<br>
特徴ベクトルを L2 ノルムで正規化しない<br>
デフォルト：指定なし
* --unuse_quantize<br>
特徴ベクトルを スカラー量子化によってバイトに量子化しない<br>
デフォルト：指定なし
</details>
<img src="https://github.com/user-attachments/assets/cc93d136-4103-4120-9987-5a40de4d44ef" loading="lazy" width="500px"><br>
<img src="https://github.com/user-attachments/assets/399eba04-00bf-47be-a728-0904d1863b6e" loading="lazy" width="500px"><br><br>
<img src="https://github.com/user-attachments/assets/1a4c725e-f6c8-4ef8-9cb8-455b6c29fb82" loading="lazy" width="500px"><br><img src="https://github.com/user-attachments/assets/298291fe-eb51-48d2-bccf-14581cd8f1bd" loading="lazy" width="500px">

### 顔検出（Face Detection）
```bash
python sample_face_landmark_detection.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --video<br>
動画パスの指定 ※指定時はカメラより優先<br>
デフォルト：None
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
使用モデル[0]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
デフォルト：0<br>
  * 0:BlazeFace (short-range)
</details>
<img src="https://github.com/user-attachments/assets/6264ab0a-7d7a-4fe1-9aeb-0a8768499fde" loading="lazy" width="250px">

### 顔のランドマーク検出（Face Landmark Detection）
```bash
python sample_face_landmark_detection.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --video<br>
動画パスの指定 ※指定時はカメラより優先<br>
デフォルト：None
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
使用モデル[0]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
デフォルト：0<br>
  * 0:FaceLandscapeer
* --num_faces<br>
検出数<br>
デフォルト：1
* --unuse_output_face_blendshapes<br>
顔のブレンドシェイプを出力しない<br>
デフォルト：指定なし
* --unuse_output_facial_transformation_matrixes<br>
顔変換行列を出力しない<br>
デフォルト：指定なし
</details>
<img src="https://github.com/user-attachments/assets/c965b40b-9592-48c3-9fb0-d8b48050c0b2" loading="lazy" width="250px">

### 顔のスタイル変換（Face Stylization）
```bash
python sample_face_stylization.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --video<br>
動画パスの指定 ※指定時はカメラより優先<br>
デフォルト：None
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
使用モデル[0, 1, 2]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
デフォルト：0<br>
  * 0:Color sketch
  * 1:Color ink
  * 2:Oil painting
</details>
<img src="https://github.com/user-attachments/assets/113a4277-47e0-450d-adae-fb16c9a89590" loading="lazy" width="250px"> <img src="https://github.com/user-attachments/assets/c252903f-ba96-4c54-9c35-6c34cd2a01f0" loading="lazy" width="250px"> <img src="https://github.com/user-attachments/assets/ec5b8b93-b602-4c93-bdc5-0a3e3a15d0fe" loading="lazy" width="250px">

### 姿勢推定（Pose Landmark Detection）
```bash
python sample_pose_landmark_detection.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --video<br>
動画パスの指定 ※指定時はカメラより優先<br>
デフォルト：None
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --unuse_mirror<br>
ミラー表示不使用<br>
デフォルト：指定なし
* --model<br>
使用モデル[0, 1, 2]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
デフォルト：0<br>
  * 0:Pose landmarker(lite)
  * 1:Pose landmarker(Full)
  * 2:Pose landmarker(Heavy)
* --use_output_segmentation_masks<br>
セグメンテーションを実施<br>
デフォルト：指定なし
* --use_world_landmark<br>
ワールド座標表示を実施<br>
デフォルト：指定なし
</details>
<img src="https://github.com/user-attachments/assets/a05f4b92-d99a-4098-ad5e-437dc2d8d6a6" loading="lazy" width="250px"> <img src="https://github.com/user-attachments/assets/89991b1f-5cae-4eff-a2b6-6f3580906ddc" loading="lazy" width="250px"> <img src="https://github.com/user-attachments/assets/f1a20ece-24f2-4312-a7ea-86f2704a7244" loading="lazy" width="250px"> 

### テキスト分類（Text Classification）
```bash
python sample_text_classification.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --input_text<br>
入力テキスト<br>
デフォルト：I'm looking forward to what will come next.
* --model<br>
使用モデル[0, 1]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
デフォルト：0<br>
  * 0:BERT-classifier
  * 1:Average word embedding
</details>
<img src="https://github.com/user-attachments/assets/d37d68a6-3539-4cee-a744-8a9f14356228" loading="lazy" width="500px">

### テキストの埋め込み表現（Text Embedding）
```bash
python sample_text_embedding.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --input_text01<br>
入力テキスト1<br>
デフォルト：I'm feeling so good
* --input_text02<br>
入力テキスト2<br>
デフォルト：I'm okay I guess
* --model<br>
使用モデル[0]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
デフォルト：0<br>
  * 0:Universal Sentence Encoder
* --unuse_l2_normalize<br>
特徴ベクトルを L2 ノルムで正規化しない<br>
デフォルト：指定なし
* --use_quantize<br>
特徴ベクトルを スカラー量子化によってバイトに量子化する<br>
デフォルト：指定なし
</details>
<img src="https://github.com/user-attachments/assets/ae23df66-d0bf-4794-a60a-7fe8ef6d7897" loading="lazy" width="500px">

### テキスト言語分類（Language Detector）
```bash
python sample_text_language_detection.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --input_text<br>
入力テキスト<br>
デフォルト：分久必合合久必分
* --model<br>
使用モデル[0, 1]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
デフォルト：0<br>
  * 0:Language Detector
</details>
<img src="https://github.com/user-attachments/assets/4b391e27-a96e-478d-aec7-1e9dfd668893" loading="lazy" width="500px">

### 音分類（Audio Classification）
```bash
python sample_audio_classification.py
```
<details>
<summary>コマンドライン引数オプション</summary>
 
* --input_audio<br>
入力音声ファイルのパス<br>
デフォルト：asset/hyakuninisshu_02.wav
* --model<br>
使用モデル[0]　※対象モデルの重みがmodelディレクトリ内に無い場合ダウンロードを実行<br>
デフォルト：0<br>
  * 0:YamNet
* --max_results<br>
結果出力数<br>
デフォルト：5
</details>
<img src="https://github.com/user-attachments/assets/c247e7b8-2576-4a97-9d3f-876e14a05809" loading="lazy" width="500px">

# Reference
* [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
mediapipe-python-sample is under [Apache-2.0 License](LICENSE).

# License(Image, Video, Audio)
サンプル実行用に格納している画像などは以下を利用しています。
* [ぱくたそ](https://www.pakutaso.com)様：[トゲトゲのサボテンとハリネズミ](https://www.pakutaso.com/20190257050post-19488.html)
* [ぱくたそ](https://www.pakutaso.com)様：[人間の靴にはまり込むハリネズ](https://www.pakutaso.com/20171041289post-13677.html)
* [ぱくたそ](https://www.pakutaso.com)様：[靴にすっぽり隠れるハリネズミ](https://www.pakutaso.com/20171039289post-13676.html)
* [NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)様：「[猫カフェのネコ（３）](https://www2.nhk.or.jp/archives/movies/?id=D0002161325_00000)」
* [NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)様：「[寅さんの像　アップ](https://www2.nhk.or.jp/archives/movies/?id=D0002022189_00000)」
* [NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)様：「[音声 　百人一首　二](https://www2.nhk.or.jp/archives/movies/?id=D0002110102_00000)」
