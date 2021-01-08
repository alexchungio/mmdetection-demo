# mmdetection-voc


## Dataset

```
python preproces_voc.py
```

## Config

reference [officer tutorial](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html)


```shell script
vim config/pascal_voc/reninanet_r50_fpn_1x_voc0712.py
```

## Train
```shell script
python voc_train.py  --config ./configs/pascal_voc/custom_retinanet_r50_fpn_1x_voc0712.py
```

## Test
```shell script
python voc_test.py
```

## View logs

```shell script
python tools/analyze_logs.py plot_curve ./outputs/retinanet_r50_fpn/20210107_171450.log.json --keys loss_cls loss_bbox loss --out losses.pdf --legend cls_loss bbox_loss total_loss --title loss

```

## TODO


## Reference
* <https://mmdetection.readthedocs.io/en/latest/tutorials/config.html>

