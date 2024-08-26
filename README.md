



## データセット

GeoNetデータセットは以下のリンクよりダウンロードできます．

[GeoNet](https://tarun005.github.io/GeoNet/)

`GeoNet/data/`というフォルダを新たに作成し，その中にダウンロードしたデータセットを入れてください．


## モデルの学習

GeoImNetを用いてモデルを学習する場合，以下のスクリプトを実行します．

```
bash jobs/GeoImNet.sh <source> <target> <Path for GeoImNet dataset>
```

例として，USAのデータで学習し，Asiaのデータでテストをしたい場合は以下のようになります．
```
bash jobs/GeoImNet.sh usa asia ./data/GeoImNet/
```

また，GeoPlacesを用いる場合は以下のスクリプトを実行します．
```
bash jobs/GeoPlaces.sh <source> <target> <Path for GeoPlaces dataset>
```

## 学習済モデルのテスト

学習済モデルを用いてテストを行う場合は，次のように入力してください．

```
python eval.py --nClasses 600 --checkpoint best_model.pth.tar --data_dir <Path for GeoImNet dataset>  --dataset GeoImNet --target asia
```

