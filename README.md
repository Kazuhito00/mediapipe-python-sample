# mediapipe-python-sample
[MediaPipe](https://github.com/google/mediapipe)のPythonパッケージのサンプルです。
2020/12/09時点でPython実装のある以下3機能について用意しています。
* [Hands](https://google.github.io/mediapipe/solutions/hands)<br>
![suwkm-avmbx](https://user-images.githubusercontent.com/37477845/101514487-a59d8500-39c0-11eb-8346-d3c9ab917ea6.gif)<br>
* [Pose](https://google.github.io/mediapipe/solutions/pose)<br>
![z9e49-wa894](https://user-images.githubusercontent.com/37477845/101512555-7ab23180-39be-11eb-814c-9fad59e0cf9a.gif)<br>
* [Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh)<br>
![8w6n9-lavib](https://user-images.githubusercontent.com/37477845/101512592-869df380-39be-11eb-8a80-241e272cc195.gif)

# Requirement 
* mediapipe 0.8.0 or later
* OpenCV 3.4.2 or later

mediapipeはpipでインストールできます。
```bash
pip install mediapipe
```

# Demo
デモの実行方法は以下です。
```bash
python sample_face.py
```
```bash
python sample_hand.py
```
```bash
python sample_pose.py
```
デモ実行時には、以下のオプションが指定可能です。<br>
また、「image」ディレクトリの画像を差し替えることによって重畳画像を変更できます。<br>
（複数枚格納した場合はアニメーションを行い、1枚であれば固定画像となります）

* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --min_detection_confidence<br>
検出信頼値の閾値<br>
デフォルト：0.5(sample_hand.pyのみ0.7)
* --min_tracking_confidence<br>
トラッキング信頼値の閾値<br>
デフォルト：0.5

# Reference
* [MediaPipe](https://github.com/google/mediapipe)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
mediapipe-python-sample is under [Apache-2.0 License](LICENSE).

また、女性の画像は[フリー素材ぱくたそ](https://www.pakutaso.com)様の写真を利用しています。
