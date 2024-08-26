# GeoNetデータセットでの




## データセット

GeoNetデータセットは以下のリンクよりダウンロードできます．

[GeoNet](https://tarun005.github.io/GeoNet/)

`GeoNet/data/`というフォルダを新たに作成し，その中にダウンロードしたデータセットを入れてください．


## モデルの学習

GeoImNetを用いてモデルを学習する場合，以下のスクリプトを実行します．

```
bash jobs/GeoImNet.sh <source> <target> <Path for GeoImNet dataset>
```

例として，USAのデータで学習し，Asiaのデータでテストをしたい場合は，
```
bash jobs/GeoImNet.sh usa asia ./data/GeoImNet/
```
となります．

また，GeoPlacesを用いる場合は以下のようになります．
```
bash jobs/GeoPlaces.sh <source> <target> <Path for GeoPlaces dataset>
```

## Testing using trained model.

To directly test our trained model, download the models available at the following links.

 Method        | Trained Model  |
| ------------- |:-----|
| DomainNet | [Link](https://drive.google.com/drive/folders/1JpWG_Pdbt2G6PBAv7Ed-vWjB5Ct5-Qqp?usp=sharing) |
| CUB-200 |   [Link](https://drive.google.com/drive/folders/1akY4kZSz7ML5TkY15NhIDYKDttTXV-ye?usp=sharing)   |

##### CUB-200 dataset
```
python eval.py --nClasses 200 --checkpoint drawing_cub.pth.tar --data_dir <Path for cub2011 dataset> --batch_size 64 --dataset cub2011 --target cub
```

##### DomainNet
```
python eval.py --nClasses 345 --checkpoint real_clipart.pth.tar --data_dir <Path for domainNet dataset>  --dataset domainNet --target clipart
```

If you find MemSAC useful for your work please cite:
```
@article{kalluri2022memsac
  author    = {Kalluri, Tarun and Sharma, Astuti and Chandraker, Manmohan},
  title     = {MemSAC: Memory Augmented Sample Consistency for Large Scale Domain Adaptation},
  journal   = {ECCV},
  year      = {2022},
}
```
